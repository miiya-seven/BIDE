from __future__ import annotations

import json
import os
import urllib.request
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

ANSWER_SYSTEM = "Answer from the supplied authoritative Raw evidence only. Return JSON only."


class ModelResponseError(RuntimeError):
    """A syntactically valid API response that contains no usable answer."""


def _chat_content(raw: str) -> str:
    """Validate an OpenAI-compatible response without leaking credentials.

    Some gateways return HTTP 200 with an ``error`` object or a refusal-style
    message that has no ``content`` key.  A bare KeyError made those transport
    failures indistinguishable from a Scheme-B planning failure.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelResponseError(f"chat_response_not_json:{exc.msg}") from exc
    choices = payload.get("choices") if isinstance(payload, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    message = first_choice.get("message") if first_choice else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(
            str(block.get("text") or "")
            for block in content if isinstance(block, dict)
        )
    if isinstance(content, str) and content.strip():
        return content
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        # Provider messages are untrusted free text and may echo request
        # headers.  Stable categorical fields are sufficient for diagnosis.
        detail = str(error.get("code") or error.get("type") or "provider_error")
    elif first_choice:
        message_keys = ",".join(sorted(map(str, message))) if isinstance(message, dict) else "none"
        detail = f"finish_reason={first_choice.get('finish_reason') or 'unknown'};message_keys={message_keys}"
    else:
        top_keys = ",".join(sorted(map(str, payload))) if isinstance(payload, dict) else type(payload).__name__
        detail = f"missing_choices_or_message_content;top_keys={top_keys[:160]}"
    raise ModelResponseError(f"chat_response_without_content:{detail[:240]}")


class ModelClients:
    def __init__(self, config_path: Path):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        configured_env = config.get("env") or {}
        def configured(name: str, default: str = "") -> str:
            value = os.getenv(name)
            if value is not None:
                return value
            raw = configured_env.get(name)
            return str(raw) if raw is not None else default

        # Config values are inputs, not process-global environment mutations.
        # Mutating os.environ here made one client silently change subsequent
        # QuestionIntent/provider resolution in the same process.
        self.llm_base = configured("OPENAI_BASE_URL")
        self.llm_key = configured("OPENAI_API_KEY")
        self.llm_credential_source = "environment_or_local_config" if self.llm_key else "missing"
        self.llm_model = configured("LLM_MODEL", str(config.get("generation_model") or "gpt-5.4-mini"))
        self.external_network_mode = configured("AB_EXTERNAL_NETWORK_MODE", "inherit").strip().casefold()
        seed_value = configured("LLM_SEED", "").strip()
        self.llm_seed = int(seed_value) if seed_value else None
        self.memory_llm_base = configured("MEMORY_BUILD_OPENAI_BASE_URL", self.llm_base)
        self.memory_llm_key = self.llm_key
        self.memory_llm_model = configured("MEMORY_BUILD_MODEL", self.llm_model)
        self.embedding_base = configured("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1")
        self.embedding_key = configured("EMBEDDING_API_KEY", "dummy")
        self.embedding_model = configured("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.reranker_base = configured("RERANKER_BASE_URL", "http://127.0.0.1:8002/v1")
        self.reranker_key = configured("RERANKER_API_KEY", "dummy")
        self.reranker_model = configured("RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")
        # A dead local reranker must not hold an entire resumable retrieval
        # chunk for ten minutes.  Callers provide a deterministic lexical
        # fallback after this bounded network timeout.
        self.reranker_timeout = float(os.getenv("RERANKER_TIMEOUT_SECONDS", "60"))

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            f"{self.embedding_base.rstrip('/')}/embeddings",
            data=json.dumps({"model": self.embedding_model, "input": texts}, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.embedding_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))) as response:
            payload = json.loads(response.read().decode())
        return [row["embedding"] for row in sorted(payload["data"], key=lambda row: row["index"])]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        base = self.reranker_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        request = urllib.request.Request(
            f"{base}/score",
            data=json.dumps({"model": self.reranker_model, "text_1": query, "text_2": documents}, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.reranker_key}"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.reranker_timeout) as response:
            payload = json.loads(response.read().decode())
        return [float(row["score"]) for row in sorted(payload["data"], key=lambda row: row["index"])]

    def _chat_json(
        self, system: str, prompt: str, max_tokens: int, *,
        base_url: str | None = None, api_key: str | None = None, model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        base_url = base_url or self.llm_base
        api_key = api_key or self.llm_key
        model = model or self.llm_model
        use_openai_sdk = os.getenv("LLM_USE_OPENAI_SDK", "").strip().casefold() in {"1", "true", "yes", "on"}
        body = {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
        # Keep the SDK request minimal for the configured provider-compatible route. The
        # prompt already contains the JSON contract; some gateway routes have
        # returned empty content when response_format/reasoning was included.
        if not use_openai_sdk:
            body.update({"temperature": 0.0, "response_format": {"type": "json_object"}})
            if self.llm_seed is not None:
                body["seed"] = self.llm_seed
        effective_reasoning_effort = (reasoning_effort if reasoning_effort is not None
                                      else os.getenv("LLM_REASONING_EFFORT", "")).strip()
        if effective_reasoning_effort and not use_openai_sdk:
            body["reasoning_effort"] = effective_reasoning_effort
        attempts = max(1, int(os.getenv("MODEL_TRANSPORT_MAX_ATTEMPTS", "4")))
        base_delay = max(0.0, float(os.getenv("MODEL_TRANSPORT_RETRY_BASE_SECONDS", "1")))
        last_error: Exception | None = None
        base_url = base_url.rstrip("/")
        endpoint = base_url + ("/chat/completions" if base_url.endswith("/v1") else "/v1/chat/completions")
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                endpoint, data=json.dumps(body, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, method="POST")
            try:
                if use_openai_sdk:
                    try:
                        from openai import OpenAI
                        sdk_client = OpenAI(
                            api_key=api_key,
                            base_url=base_url,
                            timeout=float(os.getenv("LLM_SOCKET_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "180"))),
                            max_retries=0,
                        )
                        sdk_response = sdk_client.chat.completions.create(**body)
                        if hasattr(sdk_response, "model_dump_json"):
                            raw = sdk_response.model_dump_json()
                        else:
                            raw = json.dumps(sdk_response.model_dump(mode="json"), ensure_ascii=False)
                        content = _chat_content(raw)
                    except Exception as exc:
                        raise ModelResponseError(f"sdk_transport_failed:{type(exc).__name__}:{str(exc)[:240]}") from None
                else:
                    try:
                        # In the managed runtime, Python's direct socket path
                        # can hang while curl is explicitly allow-listed and
                        # reaches the same OpenAI-compatible endpoint.  Use
                        # curl first for direct mode; retain urllib for
                        # inherited/proxied environments.
                        if self.external_network_mode == "direct":
                            curl_command = [
                                "curl", "-sS", "--fail-with-body", "--max-time",
                                str(int(float(os.getenv("LLM_CURL_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "180"))))),
                                "-X", "POST", endpoint,
                                "-H", "Content-Type: application/json",
                                "-H", f"Authorization: Bearer {api_key}",
                                "--data-binary", json.dumps(body, ensure_ascii=False),
                            ]
                            raw = subprocess.check_output(
                                curl_command, text=True, stderr=subprocess.PIPE,
                                timeout=float(os.getenv("LLM_CURL_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "180"))) + 10)
                            content = _chat_content(raw)
                        else:
                            with urllib.request.urlopen(request, timeout=float(os.getenv("LLM_SOCKET_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "180")))) as response:
                                content = _chat_content(response.read().decode())
                    except (urllib.error.URLError, OSError, subprocess.SubprocessError):
                        # Some managed runners deny Python sockets while allowing
                        # the approved curl transport. Empty HTTP-200 messages are
                        # transport failures too and are retried here rather than
                        # consuming a semantic Planner/Answer repair attempt.
                        try:
                            curl_command = [
                                "curl", "-sS", "--max-time", str(int(float(os.getenv("LLM_CURL_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "180")))))
                            ]
                            curl_command.extend(["-X", "POST",
                                endpoint,
                                "-H", "Content-Type: application/json", "-H", f"Authorization: Bearer {api_key}",
                                "--data-binary", json.dumps(body, ensure_ascii=False),
                            ])
                            raw = subprocess.check_output(curl_command, text=True, stderr=subprocess.DEVNULL,
                               timeout=float(os.getenv("LLM_CURL_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "180"))) + 10)
                        except (subprocess.SubprocessError, OSError) as exc:
                            # CalledProcessError repr contains argv, including the
                            # Authorization header. Never retain it in the causal
                            # chain or surface it in a run receipt/traceback.
                            raise ModelResponseError(f"curl_transport_failed:{type(exc).__name__}") from None
                        content = _chat_content(raw)
                payload = json.loads(content)
                if not isinstance(payload, dict):
                    raise ModelResponseError("chat_content_json_not_object")
                return payload
            except json.JSONDecodeError as exc:
                last_error = ModelResponseError(f"chat_content_not_json:{exc.msg}")
            except (ModelResponseError, subprocess.SubprocessError, TimeoutError, OSError) as exc:
                last_error = exc
            if attempt < attempts and base_delay:
                time.sleep(min(30.0, base_delay * (2 ** (attempt - 1))))
        if isinstance(last_error, ModelResponseError):
            raise last_error
        raise ModelResponseError(f"chat_transport_failed:{type(last_error).__name__ if last_error else 'unknown'}") from None

    def demand_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Parse the question into soft semantic target hypotheses. Never alter supplied hard fields and never answer the question. Return JSON only.",
            prompt, 500,
        )

    def select_evidence_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Select memory evidence that directly helps answer the question. Rank by answer-bearing relevance, not topical similarity. Preserve complementary evidence needed for comparison, counting, time, cause, attitude, or shared-activity questions. Return JSON only.",
            prompt, 320,
        )

    def focus_evidence_json(self, prompt: str, system: str) -> dict[str, Any]:
        """Select a soft attention subset while retaining the caller's full reserve Packet."""
        return self._chat_json(system, prompt, 480)

    def locate_answer_facts_json(self, prompt: str, system: str) -> dict[str, Any]:
        """Locate exact Raw spans for answer-stage attention without producing an answer."""
        return self._chat_json(system, prompt, 1400)

    def select_fact_closure_json(self, prompt: str, *, system: str, max_tokens: int) -> dict[str, Any]:
        """Select permitted closure evidence without Answer or fallback authority."""
        return self._chat_json(system, prompt, max_tokens)

    def extract_locator_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Extract a lightweight semantic locator graph from anchored dialogue evidence. Use only stated information. Normalize predicates to the supplied ontology, preserve uncertainty, and attach every fact to its exact raw_id. Do not answer any question. Return JSON only.",
            prompt, 1800,
        )

    def compile_locator_query_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Compile a memory question into the same lightweight locator ontology used by the supplied facts. Do not answer the question. Preserve relational references as variables, identify answer-bearing target terms, and return JSON only.",
            prompt, 700,
        )

    def select_locator_facts_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Select the locator facts that satisfy the complete query plan. Require correct entity, relation, target event/object, time scope, and answer field. Topic similarity alone is insufficient. Do not answer the question. Return JSON only.",
            prompt, 500,
        )

    def extract_unified_frames_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json("Extract memory frames using exactly the supplied unified schema. Never invent role or predicate names. Return JSON only.", prompt, 2200)

    def reconcile_unified_states_json(self, prompt: str, system: str, max_tokens: int) -> dict[str, Any]:
        return self._chat_json(system, prompt, max_tokens)

    def compile_unified_query_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json("Compile the question into query frames using exactly the same supplied unified schema. Do not answer. Return JSON only.", prompt, 1200)

    def extract_unified_v2_memory_json(self, prompt: str, *, max_tokens: int | None = None) -> dict[str, Any]:
        return self._chat_json(
            "Extract Scheme-B-v2 homogeneous memory. Use only supplied Raw evidence and return JSON only.",
            prompt, max_tokens or max(3200, int(os.getenv("MEMORY_BUILD_LLM_MAX_TOKENS", "3200"))),
            base_url=self.memory_llm_base, api_key=self.memory_llm_key, model=self.memory_llm_model,
        )

    def compile_unified_v2_query_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Compile a Scheme-B-v2 QueryGraph with the complete executable AnswerUnit Projection Contract. "
            "Do not answer the question. Return JSON only.",
            prompt, int(os.getenv("BNEXT_QUERY_MAX_TOKENS", "3200")),
        )

    def answer_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        return self._chat_json(system or ANSWER_SYSTEM, prompt, 320)

    def materialize_requirement_ledger_json(self, prompt: str, system: str) -> dict[str, Any]:
        """Build an answer-blind Demand-to-Requirement ledger from one Raw packet."""
        return self._chat_json(system, prompt, 2600)

    def extract_state_lifecycle_json(self, prompt: str, system: str) -> dict[str, Any]:
        """Extract one exhaustive, question-independent session lifecycle ledger."""
        return self._chat_json(
            system, prompt,
            int(os.getenv("STATE_LIFECYCLE_MAX_TOKENS", "6000")),
        )

    def propose_answer_item_delta_json(self, prompt: str, system: str) -> dict[str, Any]:
        return self._chat_json(system, prompt, 1400)

    def verify_answer_item_defect_json(self, prompt: str, system: str) -> dict[str, Any]:
        return self._chat_json(system, prompt, 700)

    def replay_temporal_item_json(self, prompt: str, system: str) -> dict[str, Any]:
        return self._chat_json(system, prompt, 900)

    def answer_bnext_json(self, prompt: str, system: str) -> dict[str, Any]:
        """One complete B-next answer operation with an auditable execution trace."""
        return self._chat_json(system, prompt, 1200)

    def revise_bnext_answer_json(self, prompt: str, system: str) -> dict[str, Any]:
        """Finalize one draft against the same packet; never compare answer arms."""
        return self._chat_json(system, prompt, 900)

    def plan_bnext_evidence_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Build one exhaustive demand-to-evidence plan for a long-term-memory question. "
            "The natural-language question is authoritative. Bind every answer unit to exact quoted Raw evidence, "
            "keep recurring events separate, and explicitly exclude plausible competitors. Do not write a final answer. "
            "Return exactly the requested JSON object.",
            prompt,
            5000,
        )

    def verify_bnext_evidence_plan_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Independently verify and, when necessary, rewrite one demand-to-evidence plan. "
            "The natural-language question and exact Raw text are authoritative; the proposed plan and QueryGraph are hypotheses. "
            "A grounded quote is not sufficient unless it binds the exact requested subject, event, role, modifier, time scope, "
            "selection rule and shared variable. Return one corrected plan in the requested JSON schema, never an answer or a vote.",
            prompt,
            5000,
        )

    def build_bnext_operation_ledger_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Materialize one operation-specific Event, Member or Temporal Ledger from authoritative Raw evidence. "
            "Use the natural-language question as authoritative, copy exact Raw quotes, preserve occurrence identity and answer-unit "
            "bindings, and return only the requested JSON. Do not generate a final answer.",
            prompt,
            int(os.getenv("BNEXT_LEDGER_MAX_TOKENS", "10000")),
        )

    def adjudicate_answers_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Conservatively adjudicate candidate answers from supplied Raw evidence. Return JSON only with selected_index (integer 0 or 1), used_raw_ids, baseline_status, residual_status, and reason. Never generate a third answer.",
            prompt,
            420,
        )

    def bind_answer_claims_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Bind answer-bearing claims from authoritative Raw evidence. Match the exact query subject, event/relation, object, modifiers, phase, and time scope before extracting an answer. Return JSON only with answer, used_raw_ids, bindings, claims, and confidence.",
            prompt,
            700,
        )

    def claim_ledger_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Build an exhaustive atomic answer-claim ledger from the supplied authoritative Raw evidence. Every claim must have an exact Raw quote. Audit baseline and candidate coverage against the same ledger. Return exactly the requested JSON fields and no prose.",
            prompt,
            2200,
        )

    def verify_claim_ledger_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Independently verify a typed memory claim ledger against the exact question and quoted Raw evidence. Do not see or compare any prior answer. Approve only when entity roles, event identity, modifiers, time/phase, selection rule, answer slot and closure are jointly proven. Return JSON only.",
            prompt,
            1000,
        )

    def collection_ledger_json(self, prompt: str) -> dict[str, Any]:
        return self._chat_json(
            "Exhaustively enumerate the exact typed collection requested by the QueryGraph from authoritative Raw evidence. Every distinct member must have a canonical key and an exact Raw quote. Do not answer outside the requested JSON schema.",
            prompt,
            3200,
        )
