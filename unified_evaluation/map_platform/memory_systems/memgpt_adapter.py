from __future__ import annotations

import os
import re
import time
from typing import Any

from letta_client import Letta

from map_platform.datasets.schema import UnifiedSample
from map_platform.memory_systems.base import (
    AnswerRecord,
    BaseMemorySystemAdapter,
    MemoryEntry,
    OrganizationState,
    PromptRecord,
    RetrievalResult,
    RetrievedMemory,
)
from map_platform.utils.serialization import to_jsonable


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _as_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if hasattr(payload, "dict"):
        return payload.dict()
    return {"value": payload}


def _flatten_assistant_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            raw = _as_dict(item)
            text = raw.get("text") or raw.get("content")
            if text:
                parts.append(str(text).strip())
        return "\n".join(part for part in parts if part)
    return str(content or "").strip()


def _letta_model_handle(value: str, *, env_key: str) -> str:
    raw = str(value or "").strip()
    override = os.getenv(env_key)
    if override:
        raw = override.strip()
    providers = {
        "anthropic",
        "azure",
        "bedrock",
        "google",
        "groq",
        "letta",
        "ollama",
        "openai",
        "openai-proxy",
        "together",
        "vllm",
    }
    if "/" in raw and raw.split("/", 1)[0].lower() in providers:
        provider, model_name = raw.split("/", 1)
        return f"{provider}/{model_name}"
    model_name = raw.rstrip("/").split("/")[-1] if raw else ""
    provider = os.getenv("LETTA_MODEL_PROVIDER", "openai").strip() or "openai"
    return f"{provider}/{model_name}" if model_name else raw


def _model_name_from_handle(value: str) -> str:
    raw = str(value or "").strip()
    return raw.split("/", 1)[1] if "/" in raw else raw


def _qwen_namespace_alias(value: str) -> str | None:
    raw = str(value or "").strip()
    if raw.startswith("openai/Qwen"):
        return f"Qwen/{raw.split('/', 1)[1]}"
    if raw.startswith("Qwen3-"):
        return f"Qwen/{raw}"
    return None


def _looks_like_model_not_found(exc: Exception) -> bool:
    text = str(exc).lower()
    return "404" in text and "model" in text and ("does not exist" in text or "notfound" in text or "not found" in text)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _letta_passage_list_limit() -> int:
    return _env_int("LETTA_PASSAGE_LIST_LIMIT", 2000)


class LettaAdapter(BaseMemorySystemAdapter):
    system_name = "letta"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Letta | None = None
        self._agent_id: str | None = None
        self._agent_state: dict[str, Any] = {}
        self._archival_passages: list[dict[str, Any]] = []

    def reset(self, sample_id: str) -> None:
        super().reset(sample_id)
        self._agent_id = None
        self._agent_state = {}
        self._archival_passages = []
        self._debug_state["sdk"] = {"service": self._service_meta()}

    def build_memory(self, history: list[dict], sample: UnifiedSample) -> None:
        self._debug_state["active_call"] = {
            "stage": "ensure_agent",
            "model": self._letta_generation_model(),
            "embedding": self._letta_embedding_model(),
            "service": self._service_meta(),
        }
        self._ensure_agent(sample)
        ingest_rows: list[dict[str, Any]] = []
        passage_groups = self._archival_passage_groups(history)
        for group_index, group in enumerate(passage_groups, start=1):
            passage_text = "\n\n".join(item["text"] for item in group if item["text"])
            if not passage_text:
                continue
            self._debug_state["active_call"] = {
                "stage": "archival_passage_create",
                "agent_id": self._agent_id,
                "group_index": group_index,
                "num_turns": len(group),
                "turn_ids": [item["turn_id"] for item in group],
            }
            response = self._client_instance().agents.passages.create(
                self._agent_id,
                text=passage_text,
                tags=[
                    "benchmark_history",
                    f"sample:{sample.sample_id}",
                    *[f"session:{item['session_id']}" for item in group if item["session_id"]],
                    *[f"turn:{item['turn_id']}" for item in group if item["turn_id"]],
                ],
            )
            ingest_rows.append(
                {
                    "group_index": group_index,
                    "num_turns": len(group),
                    "turn_ids": [item["turn_id"] for item in group],
                    "response": self._compact_passage_response(response),
                }
            )
        self._debug_state["active_call"] = {"stage": "refresh_after_ingest", "agent_id": self._agent_id}
        self._refresh_remote_state()
        self._memory_entries = self._build_memory_entries()
        core_ids = [entry.entry_id for entry in self._memory_entries if entry.metadata.get("memory_bucket") == "core"]
        archive_ids = [entry.entry_id for entry in self._memory_entries if entry.metadata.get("memory_bucket") == "archive"]
        self._org_state = OrganizationState(
            entries=list(self._memory_entries),
            levels={"core": core_ids, "archive": archive_ids},
            priorities={entry_id: (1.0 if entry_id in core_ids else 0.6) for entry_id in core_ids + archive_ids},
            raw={
                "sdk": "letta",
                "agent_id": self._agent_id,
                "service": self._service_meta(),
            },
        )
        self._debug_state["archival_ingest"] = {
            "mode": "direct_passage_create",
            "num_passages": len(ingest_rows),
            "rows": ingest_rows[:20],
            "rows_truncated": max(len(ingest_rows) - 20, 0),
        }

    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        assert self._agent_id is not None
        self._debug_state["active_call"] = {
            "stage": "archival_passages_list_search",
            "agent_id": self._agent_id,
            "top_k": top_k,
        }
        passages = self._client_instance().agents.passages.list(self._agent_id, search=query, limit=top_k)
        raw_items = [_as_dict(item) for item in passages]
        retrieval_mode = "remote_archival_search"
        if not raw_items:
            if os.getenv("LETTA_ALLOW_LOCAL_RETRIEVAL_FALLBACK", "false").strip().lower() not in {
                "1", "true", "yes", "on"
            }:
                raise RuntimeError(
                    "Letta archival search returned no results; set LETTA_ALLOW_LOCAL_RETRIEVAL_FALLBACK=true "
                    "only for non-formal diagnostics."
                )
            raw_items = self._local_archival_search(query, top_k)
            retrieval_mode = "local_archival_search_fallback_non_formal"
        retrieved: list[RetrievedMemory] = []
        for rank, raw in enumerate(raw_items, start=1):
            retrieved.append(
                RetrievedMemory(
                    entry_id=f"archive:{raw.get('id') or rank}",
                    content=str(raw.get("text") or "").strip(),
                    score=raw.get("score"),
                    rank=rank,
                    source_ids=self._passage_source_ids(raw),
                    metadata={
                        "memory_bucket": "archive",
                        "tags": raw.get("tags"),
                        **(raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}),
                        "retrieval_mode": retrieval_mode,
                    },
                )
            )
        self._retrieval_result = RetrievalResult(
            query=query,
            retrieved_entries=retrieved,
            top_k=top_k,
            raw={"sdk": "letta", "agent_id": self._agent_id, "retrieval_mode": retrieval_mode, "results": raw_items},
        )
        return self._retrieval_result

    def build_prompt(self, query: str, retrieved: RetrievalResult, sample: UnifiedSample) -> PromptRecord:
        core_entries = [entry for entry in self._memory_entries if entry.metadata.get("memory_bucket") == "core"]
        system_prompt = "\n".join(
            f"[{entry.metadata.get('label') or 'core'}]\n{entry.content}" for entry in core_entries
        ) or None
        memory_context = self._format_retrieved_memory_block(retrieved.retrieved_entries, sample=sample)
        full_prompt = (
            "You are answering a benchmark question.\n"
            f"{self._answering_instructions()}\n"
            f"Core Memory:\n{system_prompt or ''}\n"
            f"Archival Retrieval:\n{memory_context}\n"
            f"Question: {query}\n"
            "Answer:"
        )
        injected_ids = [item.entry_id for item in retrieved.retrieved_entries] + [entry.entry_id for entry in core_entries]
        injection_positions = {item.entry_id: "middle_context" for item in retrieved.retrieved_entries}
        injection_positions.update({entry.entry_id: "system_prompt" for entry in core_entries})
        self._prompt_record = PromptRecord(
            system_prompt=system_prompt,
            user_prompt=f"Question: {query}",
            memory_context=memory_context,
            full_prompt=full_prompt,
            injected_entry_ids=injected_ids,
            token_count=self._estimate_token_count(full_prompt),
            injection_positions=injection_positions,
            raw={
                "system": self.system_name,
                "sdk": "letta",
                "prompt_source": "reconstructed_from_agent_state",
                "agent_id": self._agent_id,
            },
        )
        return self._prompt_record

    def generate_answer(self, prompt: PromptRecord, sample: UnifiedSample) -> AnswerRecord:
        assert self._agent_id is not None
        started = time.perf_counter()
        answer_mode = os.getenv("LETTA_ANSWER_MODE", "agent").strip().lower()
        if answer_mode in {"local", "local_prompt", "prompt"}:
            gateway = self._resolve_generation_gateway(fallback_provider=self.llm_provider_name or "openai")
            response = None
            errors: list[str] = []
            for model in self._local_answer_model_candidates():
                try:
                    response = gateway.complete_text(
                        prompt=prompt.full_prompt,
                        model=model,
                        temperature=0.0,
                    )
                    break
                except Exception as exc:
                    errors.append(f"{model}: {type(exc).__name__}: {exc}")
                    if not _looks_like_model_not_found(exc):
                        raise
            if response is None:
                raise RuntimeError("Local Letta answer model failed: " + " | ".join(errors))
            self._answer_record = AnswerRecord(
                answer=response.text,
                raw_response={
                    **response.to_dict(),
                    "system": self.system_name,
                    "answer_mode": answer_mode,
                    "model_attempt_errors": errors,
                },
                latency=response.latency,
                token_usage={
                    "prompt_tokens_est": prompt.token_count,
                    "completion_tokens_est": len(response.text.split()),
                },
            )
            return self._answer_record
        self._debug_state["active_call"] = {
            "stage": "agent_message_create",
            "agent_id": self._agent_id,
            "model": self._letta_generation_model(),
        }
        response = self._client_instance().agents.messages.create(
            self._agent_id,
            input=prompt.full_prompt,
            max_steps=int(os.getenv("LETTA_QUERY_STEPS", "8")),
            override_model=self._letta_generation_model(),
        )
        if os.getenv("LETTA_REFRESH_AFTER_ANSWER", "false").strip().lower() in {"1", "true", "yes", "on"}:
            self._refresh_remote_state()
        usage = _as_dict(getattr(response, "usage", None))
        self._answer_record = AnswerRecord(
            answer=_strip_reasoning_tags(self._extract_answer_text(response)),
            raw_response=to_jsonable(_as_dict(response)),
            latency=time.perf_counter() - started,
            token_usage={
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "context_tokens": usage.get("context_tokens"),
                "step_count": usage.get("step_count"),
            },
        )
        return self._answer_record

    def _client_instance(self) -> Letta:
        if self._client is None:
            base_url = os.getenv("LETTA_BASE_URL")
            api_key = os.getenv("LETTA_API_KEY")
            environment = (os.getenv("LETTA_ENVIRONMENT") or ("local" if base_url else "cloud")).strip().lower()
            project_id = os.getenv("LETTA_PROJECT_ID") or None
            if not base_url and not api_key:
                raise RuntimeError("Letta 适配器需要 LETTA_BASE_URL 或 LETTA_API_KEY。")
            kwargs: dict[str, Any] = {
                "environment": environment,
            }
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
            if project_id:
                kwargs["project_id"] = project_id
            self._client = Letta(**kwargs)
        return self._client

    def _ensure_agent(self, sample: UnifiedSample) -> None:
        if self._agent_id is not None:
            return
        self._debug_state["active_call"] = {
            "stage": "agent_create",
            "model": self._letta_generation_model(),
            "embedding": self._letta_embedding_model(),
            "sample_id": sample.sample_id,
        }
        create_kwargs: dict[str, Any] = {
            "name": f"benchmark_{_safe_component(sample.sample_id)}_{os.getpid()}_{int(time.time())}",
            "include_base_tools": True,
            "include_default_source": False,
            "memory_blocks": [
                {
                    "label": "persona",
                    "value": (
                        "You are a benchmark QA assistant. Maintain useful memory faithfully, prefer recent updates, "
                        "and answer questions directly with grounded information."
                    ),
                    "limit": 4000,
                },
                {
                    "label": "benchmark_profile",
                    "value": (
                        f"sample_id={sample.sample_id}\n"
                        f"dataset={sample.dataset}\n"
                        f"task_type={sample.task_type}"
                    ),
                    "limit": 4000,
                },
            ],
            "metadata": {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "task_type": sample.task_type,
            },
        }
        llm_config = self._letta_llm_config()
        embedding_config = self._letta_embedding_config()
        if llm_config:
            create_kwargs["llm_config"] = llm_config
        else:
            create_kwargs["model"] = self._letta_generation_model()
        if embedding_config:
            create_kwargs["embedding_config"] = embedding_config
        else:
            create_kwargs["embedding"] = self._letta_embedding_model()
        agent = self._client_instance().agents.create(**create_kwargs)
        self._agent_id = str(getattr(agent, "id"))
        self._agent_state = to_jsonable(_as_dict(agent))

    def _turn_as_archival_passage(self, turn: dict[str, Any], index: int) -> str:
        text = str(turn.get("text") or "").strip()
        if not text:
            return ""
        speaker = str(turn.get("speaker") or "unknown").strip()
        turn_id = str(turn.get("turn_id") or f"turn_{index}").strip()
        session_id = str(turn.get("session_id") or "").strip()
        timestamp = str(turn.get("timestamp") or "").strip()
        header_parts = [
            f"turn_id={turn_id}",
            f"session_id={session_id}" if session_id else "",
            f"session_date={timestamp}" if timestamp else "",
            f"speaker={speaker}" if speaker else "",
        ]
        header = "; ".join(part for part in header_parts if part)
        return (
            f"[Source: {header}]\n"
            f"  {self._strip_redundant_speaker_prefix(text, speaker=speaker)}"
        )

    def _archival_passage_groups(self, history: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
        group_size = max(int(os.getenv("LETTA_PASSAGE_TURNS", "8")), 1)
        rows: list[dict[str, str]] = []
        for index, turn in enumerate(history, start=1):
            text = self._turn_as_archival_passage(turn, index)
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "turn_id": str(turn.get("turn_id") or f"turn_{index}").strip(),
                    "session_id": str(turn.get("session_id") or "").strip(),
                }
            )
        return [rows[start : start + group_size] for start in range(0, len(rows), group_size)]

    def _letta_generation_model(self) -> str:
        return _letta_model_handle(self.generation_model, env_key="LETTA_MODEL")

    def _letta_embedding_model(self) -> str:
        return _letta_model_handle(self.embedding_model, env_key="LETTA_EMBEDDING_MODEL")

    def _local_answer_model_candidates(self) -> list[str]:
        primary = (
            os.getenv("LETTA_LOCAL_ANSWER_MODEL")
            or os.getenv("LETTA_ANSWER_MODEL")
            or self.generation_model
            or os.getenv("LLM_MODEL")
            or ""
        ).strip()
        candidates: list[str] = []
        for model in (primary, os.getenv("LLM_MODEL", "").strip(), self.generation_model):
            if model and model not in candidates:
                candidates.append(model)
            alias = _qwen_namespace_alias(model)
            if alias and alias not in candidates:
                candidates.append(alias)
        return candidates

    def _letta_llm_config(self) -> dict[str, Any] | None:
        endpoint = (
            os.getenv("LETTA_MODEL_ENDPOINT")
            or os.getenv("LETTA_LLM_ENDPOINT")
            or os.getenv("LETTA_OPENAI_BASE_URL")
        )
        if not endpoint:
            return None
        handle = self._letta_generation_model()
        return {
            "model": os.getenv("LETTA_MODEL_NAME") or _model_name_from_handle(handle),
            "handle": handle,
            "model_endpoint_type": os.getenv("LETTA_MODEL_ENDPOINT_TYPE", "openai"),
            "model_endpoint": endpoint,
            "context_window": _env_int("LETTA_CONTEXT_WINDOW", 30000),
            "max_tokens": _env_int("LETTA_MAX_TOKENS", 4096),
            "temperature": float(os.getenv("LETTA_TEMPERATURE", "0")),
            "enable_reasoner": os.getenv("LETTA_ENABLE_REASONER", "false").strip().lower() in {"1", "true", "yes", "on"},
        }

    def _letta_embedding_config(self) -> dict[str, Any] | None:
        endpoint = (
            os.getenv("LETTA_EMBEDDING_ENDPOINT")
            or os.getenv("LETTA_EMBEDDING_BASE_URL")
            or os.getenv("LETTA_OPENAI_BASE_URL")
        )
        if not endpoint:
            return None
        handle = self._letta_embedding_model()
        return {
            "embedding_model": os.getenv("LETTA_EMBEDDING_MODEL_NAME") or _model_name_from_handle(handle),
            "handle": handle,
            "embedding_endpoint_type": os.getenv("LETTA_EMBEDDING_ENDPOINT_TYPE", "openai"),
            "embedding_endpoint": endpoint,
            "embedding_dim": _env_int("LETTA_EMBEDDING_DIM", 1536),
            "embedding_chunk_size": _env_int("LETTA_EMBEDDING_CHUNK_SIZE", 300),
            "batch_size": _env_int("LETTA_EMBEDDING_BATCH_SIZE", 32),
        }

    def _refresh_remote_state(self) -> None:
        assert self._agent_id is not None
        try:
            agent_state = self._client_instance().agents.retrieve(self._agent_id, include=["agent.blocks"])
        except TypeError as exc:
            if "include" not in str(exc):
                raise
            agent_state = self._client_instance().agents.retrieve(self._agent_id)
        self._agent_state = to_jsonable(_as_dict(agent_state))
        passages = self._client_instance().agents.passages.list(self._agent_id, limit=_letta_passage_list_limit())
        self._archival_passages = [to_jsonable(_as_dict(item)) for item in passages]

    def _build_memory_entries(self) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        for index, block in enumerate(self._agent_state.get("blocks") or [], start=1):
            raw = _as_dict(block)
            entries.append(
                MemoryEntry(
                    entry_id=f"core:{raw.get('id') or index}",
                    content=str(raw.get("value") or "").strip(),
                    raw=raw,
                    created_at=raw.get("created_at"),
                    updated_at=raw.get("updated_at"),
                    source_ids=[str(raw.get("id") or "")],
                    metadata={
                        "memory_bucket": "core",
                        "label": raw.get("label"),
                        "limit": raw.get("limit"),
                    },
                )
            )
        for index, passage in enumerate(self._archival_passages, start=1):
            entries.append(
                MemoryEntry(
                    entry_id=f"archive:{passage.get('id') or index}",
                    content=str(passage.get("text") or "").strip(),
                    raw=passage,
                    created_at=passage.get("created_at"),
                    updated_at=passage.get("updated_at"),
                    source_ids=self._passage_source_ids(passage),
                    metadata={
                        "memory_bucket": "archive",
                        "tags": passage.get("tags"),
                        **(passage.get("metadata") if isinstance(passage.get("metadata"), dict) else {}),
                    },
                )
            )
        return entries

    def _extract_answer_text(self, response: Any) -> str:
        messages = getattr(response, "messages", None) or []
        for message in reversed(messages):
            if getattr(message, "message_type", None) != "assistant_message":
                continue
            text = _flatten_assistant_content(getattr(message, "content", ""))
            if text:
                return text
        return ""

    def _local_archival_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_terms = _search_terms(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for passage in self._archival_passages:
            text = str(passage.get("text") or "")
            text_terms = _search_terms(text)
            if not text_terms:
                continue
            overlap = query_terms & text_terms
            score = len(overlap) / max(1, len(query_terms))
            phrase_bonus = 0.5 if any(term in text.lower() for term in query_terms if len(term) > 4) else 0.0
            final_score = score + phrase_bonus
            if final_score > 0:
                item = dict(passage)
                item["score"] = final_score
                scored.append((final_score, item))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def _passage_source_ids(self, passage: dict[str, Any]) -> list[str]:
        source_ids: list[str] = []
        for tag in passage.get("tags") or []:
            text = str(tag)
            if text.startswith("turn:"):
                source_ids.append(text.split(":", 1)[1])
        if not source_ids and passage.get("id"):
            source_ids.append(str(passage.get("id")))
        return source_ids

    def _compact_passage_response(self, response: Any) -> dict[str, Any]:
        raw = to_jsonable(_as_dict(response))
        value = raw.get("value")
        if isinstance(value, list):
            raw["value"] = [self._compact_passage_item(item) for item in value[:3] if isinstance(item, dict)]
            raw["value_truncated"] = max(len(value) - 3, 0)
        return raw

    def _compact_passage_item(self, item: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "id": item.get("id"),
            "text": str(item.get("text") or "")[:240],
            "tags": item.get("tags"),
            "created_at": item.get("created_at"),
        }
        embedding = item.get("embedding")
        if isinstance(embedding, list):
            compact["embedding_dim"] = len(embedding)
        embedding_config = item.get("embedding_config")
        if isinstance(embedding_config, dict):
            compact["embedding_config"] = {
                "handle": embedding_config.get("handle"),
                "embedding_model": embedding_config.get("embedding_model"),
                "embedding_endpoint": embedding_config.get("embedding_endpoint"),
                "embedding_dim": embedding_config.get("embedding_dim"),
            }
        return compact

    def _service_meta(self) -> dict[str, Any]:
        return {
            "environment": os.getenv("LETTA_ENVIRONMENT") or ("local" if os.getenv("LETTA_BASE_URL") else "cloud"),
            "base_url": os.getenv("LETTA_BASE_URL"),
            "project_id": os.getenv("LETTA_PROJECT_ID"),
            "model": self._letta_generation_model(),
            "embedding": self._letta_embedding_model(),
        }

    def validate_setup(self) -> dict[str, Any]:
        if not os.getenv("LETTA_BASE_URL") and not os.getenv("LETTA_API_KEY"):
            return {
                "system": self.system_name,
                "ready": False,
                "provider": "letta",
                "message": "缺少 LETTA_BASE_URL 或 LETTA_API_KEY。",
            }
        try:
            client = self._client_instance()
            if self._letta_llm_config() and self._letta_embedding_config():
                return {
                    "system": self.system_name,
                    "ready": True,
                    "provider": "letta",
                    "message": "使用显式 llm_config/embedding_config，跳过 Letta server 模型注册表校验。",
                }
            models = to_jsonable(client.models.list())
            available_handles = self._available_model_handles(models)
            requested = self._letta_generation_model()
            if available_handles and requested not in available_handles:
                return {
                    "system": self.system_name,
                    "ready": False,
                    "provider": "letta",
                    "message": f"Letta 模型 {requested} 不可用。可用模型: {', '.join(available_handles[:12])}",
                }
            if available_handles == []:
                return {
                    "system": self.system_name,
                    "ready": False,
                    "provider": "letta",
                    "message": "Letta server 当前没有注册任何模型。请先在 Letta 服务端配置 LLM/embedding provider。",
                }
        except Exception as exc:
            return {
                "system": self.system_name,
                "ready": False,
                "provider": "letta",
                "message": f"{type(exc).__name__}: {exc}",
            }
        return {
            "system": self.system_name,
            "ready": True,
            "provider": "letta",
            "message": "",
        }

    def _available_model_handles(self, payload: Any) -> list[str] | None:
        if isinstance(payload, dict):
            for key in ("models", "data", "results", "items", "value"):
                value = payload.get(key)
                if isinstance(value, list):
                    return self._handles_from_model_items(value)
            if payload:
                return self._handles_from_model_items([payload])
            return []
        if isinstance(payload, list):
            return self._handles_from_model_items(payload)
        return None

    def _handles_from_model_items(self, items: list[Any]) -> list[str]:
        handles: list[str] = []
        for item in items:
            raw = _as_dict(item)
            for key in ("handle", "model", "name", "id"):
                value = str(raw.get(key) or "").strip()
                if value:
                    handles.append(value)
                    break
        return sorted(set(handles))


def _strip_reasoning_tags(text: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    return stripped or (text or "").strip()


def _search_terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "at",
        "did",
        "for",
        "go",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "when",
    }
    return {term for term in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(term) > 2 and term not in stopwords}


class MemGPTAdapter(LettaAdapter):
    # Letta 是 MemGPT 的后继实现，这里保留兼容别名。
    system_name = "memgpt"
