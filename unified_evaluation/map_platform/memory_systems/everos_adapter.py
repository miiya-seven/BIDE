from __future__ import annotations

import os
import re
import time
from uuid import uuid4
from typing import Any

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


# =============================================================================
# Official EverOS LoCoMo answering pipeline (ported from
# EverOS/tests/test_locomo.py : ANSWER_PROMPT / _extract_final_answer /
# _answer_one). Enabled by default via EVEROS_OFFICIAL_ANSWER so that EverOS
# answers LoCoMo questions exactly like its own paper pipeline (7-step CoT,
# FINAL ANSWER extraction, rising-temperature retry) instead of the generic
# base.py prompt.
# =============================================================================

EVEROS_ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory 1: [timestamp] - [content]
- Memory 2: [timestamp] - [content]
...

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown -> Memory 2 mentions A's hometown is LA -> Therefore A moved from LA"]
- Inferred facts: [list any facts that require combining information from multiple memories]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references]
- Original reference: [e.g., "last year" from May 2022]
- Calculated actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION CHECK
[If multiple memories contain different information]
- Conflicting information: [describe]
- Resolution: [explain which is most recent/reliable]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included: [list them]
- [ ] All locations included: [list them]
- [ ] All numbers exact: [list them]
- [ ] All frequencies specific: [list them]
- [ ] All dates/times precise: [list them]
- [ ] All proper nouns preserved: [list them]

## STEP 7: ANSWER FORMULATION
[Explain how you're combining the information to answer the question]

## FINAL ANSWER:
[Provide the concise answer with ALL specific details preserved]

---

{context}

Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:
"""


def _everos_extract_final_answer(text: str) -> str:
    """Extract text after 'FINAL ANSWER:' marker (ported from official eval)."""
    marker = "FINAL ANSWER:"
    idx = text.upper().rfind(marker.upper())
    if idx != -1:
        answer = text[idx + len(marker):].strip()
        answer = re.sub(r"^#+\s*", "", answer).strip()
        return answer
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            return line
    return text.strip()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw.isdigit() else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _infer_source_ids(content: str, history: list[dict[str, Any]], max_matches: int = 5) -> list[str]:
    if not content.strip():
        return []
    content_words = set(content.lower().split())
    if not content_words:
        return []
    scored: list[tuple[float, str]] = []
    for turn in history:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        turn_id = str(turn.get("turn_id") or turn.get("source_id") or turn.get("dia_id") or "")
        if not turn_id:
            continue
        turn_words = set(text.lower().split())
        if not turn_words:
            continue
        overlap = len(content_words & turn_words) / max(len(content_words), 1)
        if overlap > 0.15:
            scored.append((overlap, turn_id))
    scored.sort(key=lambda x: -x[0])
    return [tid for _, tid in scored[:max_matches]]


class EverOSAdapter(BaseMemorySystemAdapter):
    system_name = "everos"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._base_url: str = ""
        self._owner_id: str = ""
        self._conv_prefix: str = ""
        self._app_id: str = ""
        self._project_id: str = ""
        self._session: Any = None

    def reset(self, sample_id: str) -> None:
        super().reset(sample_id)
        self._base_url = _env("EVEROS_BASE_URL", "http://localhost:8000").rstrip("/")
        namespace = _safe_component(_env("EVEROS_NAMESPACE_SUFFIX", ""))
        namespace_part = f"_{namespace}" if namespace else ""
        self._conv_prefix = f"map_bench{namespace_part}_{_safe_component(sample_id)}"
        self._owner_id = ""
        self._app_id = f"map_bench{namespace_part}"
        self._project_id = f"{_safe_component(sample_id)}{namespace_part}"
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"Content-Type": "application/json"})
        return self._session

    def _post(self, path: str, data: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        timeout = _env_int("EVEROS_TIMEOUT", 300)
        url = f"{self._base_url}{path}"
        session = self._get_session()
        try:
            resp = session.post(url, json=data, timeout=(10, timeout))
            return resp.status_code, resp.json() if resp.ok else {"status_code": resp.status_code, "text": resp.text[:500]}
        except Exception as e:
            return -1, {"error": str(e)}

    def build_memory(self, history: list[dict[str, Any]], sample: UnifiedSample) -> None:
        sessions = self._group_by_session(history, sample)
        if not sessions:
            return

        first_speaker = next(
            (
                str(turn.get("speaker") or "").strip()
                for turn in history
                if str(turn.get("speaker") or "").strip()
            ),
            "user",
        )
        conv_id = _safe_component(self._sample_id)
        namespace = _safe_component(_env("EVEROS_NAMESPACE_SUFFIX", ""))
        namespace_part = f"_{namespace}" if namespace else ""
        self._owner_id = f"{first_speaker.lower()}_{conv_id}{namespace_part}"
        self._debug_state["owner_id"] = self._owner_id

        if _env_bool("EVEROS_REUSE_EXISTING_MEMORY", False):
            self._debug_state["reused_existing_memory"] = True
            self._set_session_memory_entries(sessions)
            return

        batch_size = _env_int("EVEROS_BATCH_SIZE", 50)

        for sess in sessions:
            session_id = f"{self._conv_prefix}_s{sess['session_idx']}"
            api_messages = self._format_messages(sess["messages"], conv_id, namespace_part)

            for i in range(0, len(api_messages), batch_size):
                batch = api_messages[i:i + batch_size]
                payload = {
                    "session_id": session_id,
                    "messages": batch,
                }
                self._add_scope(payload)
                status, resp = self._post("/api/v1/memory/add", payload)
                if status != 200:
                    error = {
                        "session_id": session_id,
                        "batch_idx": i // batch_size,
                        "status": status,
                        "response": resp,
                    }
                    self._debug_state.setdefault("add_errors", []).append(error)
                    raise RuntimeError(
                        "EverOS memory/add failed; refusing to evaluate with an "
                        f"incomplete memory build: {error}"
                    )

            flush_status, flush_resp = self._post(
                "/api/v1/memory/flush",
                self._scoped_payload({"session_id": session_id}),
            )
            if flush_status != 200:
                error = {
                    "session_id": session_id,
                    "status": flush_status,
                    "response": flush_resp,
                }
                self._debug_state.setdefault("flush_errors", []).append(error)
                raise RuntimeError(
                    "EverOS memory/flush failed; refusing to evaluate with an "
                    f"incomplete memory build: {error}"
                )

        wait_seconds = _env_int("EVEROS_POST_FLUSH_WAIT", 180)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        poll_interval = 10
        poll_max = _env_int("EVEROS_POLL_MAX_ATTEMPTS", 6)
        for _attempt in range(poll_max):
            poll_payload = {
                "query": "test",
                "method": "hybrid",
                "top_k": 1,
                "user_id": self._owner_id,
            }
            self._add_scope(poll_payload)
            status, resp = self._post("/api/v1/memory/search", poll_payload)
            if status == 200 and resp.get("data", {}).get("episodes"):
                break
            time.sleep(poll_interval)

        self._set_session_memory_entries(sessions)

    def _set_session_memory_entries(self, sessions: list[dict[str, Any]]) -> None:
        self._memory_entries = [
            MemoryEntry(
                entry_id=f"everos_session_{sess['session_idx']}",
                content=f"Session {sess['session_idx']}: {len(sess['messages'])} messages",
                raw={"session_idx": sess["session_idx"]},
                source_ids=[str(m.get("turn_id") or m.get("dia_id") or "") for m in sess["messages"]],
            )
            for sess in sessions
        ]

    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        method = _env("EVEROS_SEARCH_METHOD", "hybrid")
        search_top_k = _env_int("EVEROS_RETRIEVE_TOP_K", top_k)
        payload: dict[str, Any] = {
            "query": query,
            "method": method,
            "top_k": search_top_k,
            "user_id": self._owner_id,
            # Official LoCoMo eval does NOT request profiles (episode-only context).
            # Keep configurable; default False to match official pipeline.
            "include_profile": _env_bool("EVEROS_INCLUDE_PROFILE", False),
        }
        self._add_scope(payload)
        status, resp = self._post("/api/v1/memory/search", payload)

        if status != 200:
            self._retrieval_result = RetrievalResult(
                query=query,
                retrieved_entries=[],
                top_k=top_k,
                raw={"error": resp, "status": status},
            )
            return self._retrieval_result

        data = resp.get("data", {})
        retrieved = self._map_search_results(data, sample, search_top_k)

        self._retrieval_result = RetrievalResult(
            query=query,
            retrieved_entries=retrieved,
            top_k=top_k,
            raw={"episodes_count": len(data.get("episodes", [])), "profiles_count": len(data.get("profiles", []))},
        )
        return self._retrieval_result

    def build_prompt(self, query: str, retrieved: RetrievalResult, sample: UnifiedSample) -> PromptRecord:
        """Official EverOS LoCoMo prompt: 7-step CoT over episode context.

        Falls back to base.py generic prompt when EVEROS_OFFICIAL_ANSWER is off.
        """
        if not _env_bool("EVEROS_OFFICIAL_ANSWER", True):
            return super().build_prompt(query, retrieved, sample)
        # Build episode-only context exactly like official _build_context:
        # "N. {subject}: {episode_body}" — one line per retrieved episode.
        lines: list[str] = []
        for idx, item in enumerate(retrieved.retrieved_entries, 1):
            lines.append(f"{idx}. {str(item.content or '').strip()}")
        context = "\n".join(lines)
        full_prompt = EVEROS_ANSWER_PROMPT.format(context=context, question=query)
        self._prompt_record = PromptRecord(
            system_prompt=None,
            user_prompt=f"Question: {query}",
            memory_context=context,
            full_prompt=full_prompt,
            injected_entry_ids=[item.entry_id for item in retrieved.retrieved_entries],
            token_count=self._estimate_token_count(full_prompt),
            injection_positions={item.entry_id: "middle_context" for item in retrieved.retrieved_entries},
            raw={"adapter": self.system_name, "official_answer": True},
        )
        return self._prompt_record

    def generate_answer(self, prompt: PromptRecord, sample: UnifiedSample) -> AnswerRecord:
        """Official EverOS answer generation: rising-temperature retry + FINAL ANSWER extraction.

        gpt-style models occasionally emit the FINAL ANSWER marker then stop without
        a body at temperature 0; retries bump temperature (0.0->0.3->0.6) to break the
        deterministic truncation path (ported from official _answer_one).
        """
        if not _env_bool("EVEROS_OFFICIAL_ANSWER", True):
            return super().generate_answer(prompt, sample)
        raw_answer = ""
        final = ""
        last_latency = 0.0
        last_error: Exception | None = None
        for temp in (0.0, 0.3, 0.6):
            old_extra_body = os.environ.get("LLM_USE_EXTRA_BODY")
            os.environ["LLM_USE_EXTRA_BODY"] = "1"
            try:
                response = self.llm.complete_text(
                    prompt=prompt.full_prompt, model=self.generation_model, temperature=temp
                )
            except Exception as exc:  # noqa: BLE001 - retry next official temperature
                last_error = exc
                if old_extra_body is None:
                    os.environ.pop("LLM_USE_EXTRA_BODY", None)
                else:
                    os.environ["LLM_USE_EXTRA_BODY"] = old_extra_body
                continue
            finally:
                if old_extra_body is None:
                    os.environ.pop("LLM_USE_EXTRA_BODY", None)
                else:
                    os.environ["LLM_USE_EXTRA_BODY"] = old_extra_body
            last_latency = response.latency
            raw_answer = response.text or ""
            final = _everos_extract_final_answer(raw_answer)
            if final.strip():
                break
        if not final.strip() and last_error is not None:
            raise last_error
        self._answer_record = AnswerRecord(
            answer=final,
            raw_response={"raw_cot": raw_answer, "official_answer": True},
            latency=last_latency,
            token_usage={
                "prompt_tokens_est": prompt.token_count,
                "completion_tokens_est": len(raw_answer.split()),
            },
        )
        return self._answer_record

    def validate_setup(self) -> dict[str, Any]:
        try:
            session = self._get_session()
            url = f"{_env('EVEROS_BASE_URL', 'http://localhost:8000').rstrip('/')}/health"
            resp = session.get(url, timeout=5)
            payload = resp.json() if resp.status_code == 200 else {}
            if resp.status_code == 200 and payload.get("status") in {"ok", "healthy"}:
                return {"system": self.system_name, "ready": True, "provider": "everos_http", "message": ""}
        except Exception as e:
            return {
                "system": self.system_name,
                "ready": False,
                "provider": "everos_http",
                "message": f"Cannot connect to EverOS at {_env('EVEROS_BASE_URL', 'http://localhost:8000')}: {e}",
            }
        return {
            "system": self.system_name,
            "ready": False,
            "provider": "everos_http",
            "message": "EverOS health endpoint returned server error",
        }

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _group_by_session(self, history: list[dict[str, Any]], sample: UnifiedSample) -> list[dict[str, Any]]:
        sessions_dict: dict[str, list[dict[str, Any]]] = {}
        for turn in history:
            sid = str(turn.get("session_id") or "default")
            sessions_dict.setdefault(sid, []).append(turn)

        sessions: list[dict[str, Any]] = []
        for idx, (sid, turns) in enumerate(sessions_dict.items(), 1):
            sessions.append({"session_idx": idx, "session_id": sid, "messages": turns})
        return sessions

    def _format_messages(
        self, messages: list[dict[str, Any]], conv_id: str, namespace_part: str = ""
    ) -> list[dict[str, Any]]:
        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            text = str(msg.get("text") or "").strip()
            if not text:
                continue
            speaker = str(msg.get("speaker") or "user").strip()
            timestamp = msg.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    from datetime import datetime, timezone
                    dt = datetime.strptime(timestamp.strip(), "%I:%M %p on %d %B, %Y")
                    ts_ms = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
                except (ValueError, TypeError):
                    ts_ms = int(time.time() * 1000)
            elif isinstance(timestamp, (int, float)):
                ts_ms = int(timestamp) if timestamp > 1e12 else int(timestamp * 1000)
            else:
                ts_ms = int(time.time() * 1000)

            api_messages.append({
                "sender_id": f"{speaker.lower()}_{conv_id}{namespace_part}",
                "sender_name": speaker,
                "role": "user",
                "timestamp": ts_ms,
                "content": [{"type": "text", "text": text}],
            })
        return api_messages

    def _scoped_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = dict(payload)
        self._add_scope(data)
        return data

    def _add_scope(self, payload: dict[str, Any]) -> None:
        if not _env_bool("EVEROS_USE_SCOPE", False):
            return
        payload["app_id"] = _env("EVEROS_APP_ID", self._app_id) or self._app_id
        payload["project_id"] = _env("EVEROS_PROJECT_ID", self._project_id) or self._project_id

    def _map_search_results(
        self, data: dict[str, Any], sample: UnifiedSample, top_k: int
    ) -> list[RetrievedMemory]:
        retrieved: list[RetrievedMemory] = []
        rank = 1
        # Official LoCoMo eval builds context from EPISODES ONLY (subject: episode).
        # In official mode, episodes get the full top_k budget and profiles /
        # atomic_facts are not used as separate evidence entries.
        episodes_only = _env_bool("EVEROS_OFFICIAL_ANSWER", True) and not _env_bool(
            "EVEROS_INCLUDE_PROFILE", False
        )

        for ep in data.get("episodes", []):
            if rank > top_k:
                break
            subject = ep.get("subject", "")
            body = ep.get("episode") or ep.get("summary") or ep.get("content") or ""
            content = f"{subject}: {body}".strip(": ") if subject else body.strip()
            if not content:
                continue

            timestamp = ep.get("timestamp") or ep.get("occurred_at") or ""
            if timestamp:
                content = f"[{timestamp}] {content}"

            source_ids = _infer_source_ids(content, sample.history)
            retrieved.append(RetrievedMemory(
                entry_id=f"everos:episode:{ep.get('id', rank)}",
                content=content,
                score=ep.get("score"),
                rank=rank,
                source_ids=source_ids,
                metadata={
                    "memory_type": "episode",
                    "subject": subject,
                    "timestamp": timestamp,
                    "adapter": self.system_name,
                },
            ))
            rank += 1

        for p in data.get("profiles", []):
            if episodes_only:
                break
            if rank > top_k:
                break
            content = p.get("content") or p.get("summary") or p.get("value") or ""
            if not content.strip():
                continue
            retrieved.append(RetrievedMemory(
                entry_id=f"everos:profile:{p.get('id', rank)}",
                content=content.strip(),
                score=p.get("score"),
                rank=rank,
                source_ids=[],
                metadata={"memory_type": "profile", "adapter": self.system_name},
            ))
            rank += 1

        for ep in data.get("episodes", []):
            if episodes_only:
                break
            for fact in ep.get("atomic_facts", []):
                if rank > top_k:
                    break
                content = fact.get("content") or fact.get("fact") or ""
                if not content.strip():
                    continue
                source_ids = _infer_source_ids(content, sample.history)
                retrieved.append(RetrievedMemory(
                    entry_id=f"everos:fact:{fact.get('id', rank)}",
                    content=content.strip(),
                    score=fact.get("score"),
                    rank=rank,
                    source_ids=source_ids,
                    metadata={"memory_type": "atomic_fact", "adapter": self.system_name},
                ))
                rank += 1
            if rank > top_k:
                break

        return retrieved
