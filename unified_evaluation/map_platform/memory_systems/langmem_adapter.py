from __future__ import annotations

import os
import re
from collections import Counter
from contextlib import contextmanager
from typing import Any

from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

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
from map_platform.llm.gateway import embed_texts, is_gpt5_like, openai_base_url
from map_platform.utils.serialization import to_jsonable


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


def _extract_content(raw: dict[str, Any]) -> str:
    value = raw.get("value")
    if isinstance(value, dict):
        content = value.get("content") or value.get("text")
        if isinstance(content, dict):
            return str(content.get("content") or content.get("text") or content).strip()
        return str(content or value).strip()
    return str(value or raw.get("content") or "").strip()


def _extract_value_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("value")
    if isinstance(value, dict):
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            return dict(metadata)
        extracted = {
            key: value.get(key)
            for key in (
                "session_id",
                "session_date",
                "created_at",
                "updated_at",
                "speaker",
                "source_turn_ids",
                "source_ids",
            )
            if value.get(key)
        }
        return extracted
    return {}


def _extract_source_ids_from_value(raw: dict[str, Any]) -> list[str]:
    value = raw.get("value")
    candidates: Any = []
    if isinstance(value, dict):
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        candidates = (
            value.get("source_turn_ids")
            or value.get("source_ids")
            or metadata.get("source_turn_ids")
            or metadata.get("source_ids")
            or []
        )
        if not candidates:
            content = _extract_content(raw)
            candidates = []
            for match in re.finditer(
                r"\b(?:source_turn_ids?|source_ids?)\s*[:=]\s*([^;\]\n]+)",
                content,
                flags=re.IGNORECASE,
            ):
                raw_ids = re.split(r"[,;]", match.group(1))
                candidates.extend(item.strip() for item in raw_ids if item.strip())
            for match in re.finditer(r"\[source:\s*([^;\]\n]+)", content, flags=re.IGNORECASE):
                candidate = match.group(1).strip()
                if candidate:
                    candidates.append(candidate)
    if not candidates:
        candidates = raw.get("key") or []
    if not isinstance(candidates, list):
        candidates = [candidates]
    source_ids: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        source_id = str(item).strip()
        if source_id and source_id not in seen:
            seen.add(source_id)
            source_ids.append(source_id)
    return source_ids


def _token_counts(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9']+", text.lower()))


_FOCUS_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "been",
    "did",
    "do",
    "does",
    "for",
    "from",
    "gave",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "name",
    "of",
    "on",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
    "with",
}


def _lexical_score(query: str, content: str) -> float:
    query_counts = _token_counts(query)
    content_counts = _token_counts(content)
    if not query_counts or not content_counts:
        return 0.0
    overlap = sum(min(query_counts[token], content_counts[token]) for token in query_counts)
    return overlap / max(sum(query_counts.values()), 1)


def _query_focus_terms(query: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", query.lower())
        if len(token) > 2 and token not in _FOCUS_STOPWORDS
    }


def _focused_content(query: str, content: str, *, max_sections: int = 4, max_chars: int = 2400) -> str:
    text = str(content or "").strip()
    if len(text) <= max_chars:
        return text
    terms = _query_focus_terms(query)
    if not terms:
        return text[:max_chars].rstrip()
    sections = [section.strip() for section in re.split(r"\n\s*\n", text) if section.strip()]
    scored: list[tuple[int, int, str]] = []
    for index, section in enumerate(sections):
        tokens = set(re.findall(r"[a-z0-9']+", section.lower()))
        score = len(tokens & terms)
        if score:
            scored.append((score, index, section))
    if not scored:
        return text[:max_chars].rstrip()
    selected = [
        section
        for _, _, section in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sections]
    ]
    focused = "\n\n".join(selected).strip()
    if len(focused) > max_chars:
        focused = focused[:max_chars].rstrip()
    return focused


def _looks_like_openai_compatibility_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "bad_response_status_code",
            "openai_error",
            "permissiondeniederror",
            "403",
            "tool",
            "structured",
            "response_format",
            "function",
            "model_dump",
            "object has no attribute",
        )
    )


def _langmem_llm_max_tokens() -> int:
    return int(os.getenv("LANGMEM_LLM_MAX_TOKENS", "1024"))


def _langmem_list_limit() -> int:
    return int(os.getenv("LANGMEM_LIST_LIMIT", "2000"))


class LangMemAdapter(BaseMemorySystemAdapter):
    system_name = "langmem"

    _SEMANTIC_INSTRUCTIONS = (
        "You are maintaining semantic memory for a benchmark over dialogue evidence. "
        "Extract concrete facts, preferences, events, plans, updates, and dated observations. "
        "Facts may be stated by either the user or the assistant; preserve assistant-provided facts that a future question may ask the user to recall. "
        "Preserve small but answerable details, including names of services, apps, pets, people, gift givers, gift recipients, specific items, prices, dates, and who did or gave what. "
        "Every stored memory must preserve source provenance by including source_turn_ids such as D1:3. "
        "Use session_date in each source header to resolve relative time expressions like yesterday, last week, and recently. "
        "For date-sensitive facts, store the resolved absolute date and the source turn id. "
        "Do not replace exact evidence with only a broad profile summary."
    )

    _PROCEDURAL_INSTRUCTIONS = (
        "You are maintaining procedural memory for a long-lived assistant. "
        "Extract stable user preferences, constraints, response-style instructions, and standing rules. "
        "Do not store transient observations unless they imply a durable instruction."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sdk_provider = self._resolve_sdk_provider()
        self._embed_provider = (os.getenv("LANGMEM_EMBED_PROVIDER") or self._sdk_provider).strip().lower()
        self._store: InMemoryStore | None = None
        self._semantic_manager = None
        self._procedural_manager = None

    def _is_truthy_env(self, name: str, default: str = "false") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

    def reset(self, sample_id: str) -> None:
        super().reset(sample_id)
        self._ensure_provider_credentials()
        with self._langmem_llm_env():
            self._store = InMemoryStore(
                index={
                    "dims": self._embedding_dims(self.embedding_model),
                    "embed": self._embed_texts,
                }
            )
            model_handle = self._langmem_model_handle()
            self._semantic_manager = create_memory_store_manager(
                model_handle,
                namespace=("semantic_memories", "{langgraph_user_id}"),
                store=self._store,
                instructions=self._SEMANTIC_INSTRUCTIONS,
                query_model=model_handle,
                query_limit=5,
            )
            self._procedural_manager = create_memory_store_manager(
                model_handle,
                namespace=("procedural_memories", "{langgraph_user_id}"),
                store=self._store,
                instructions=self._PROCEDURAL_INSTRUCTIONS,
                query_model=model_handle,
                query_limit=5,
            )
        self._debug_state["sdk"] = {
            "provider": self._sdk_provider,
            "embed_provider": self._embed_provider,
            "generation_model": self._model_debug_name(model_handle),
            "base_url": os.getenv("LANGMEM_LLM_BASE_URL"),
            "embedding_model": self.embedding_model,
        }

    def build_memory(self, history: list[dict], sample: UnifiedSample) -> None:
        assert self._semantic_manager is not None and self._procedural_manager is not None
        messages = self._history_as_sourced_messages(history)
        config = self._langgraph_config()
        batch_size = max(int(os.getenv("LANGMEM_BATCH_TURNS", "50")), 1)
        semantic_updates = []
        procedural_updates = []
        total_batches = (len(messages) + batch_size - 1) // batch_size if messages else 0
        self._debug_state["build_memory_config"] = {
            "sample_id": sample.sample_id,
            "batch_size": batch_size,
            "num_messages": len(messages),
            "total_batches": total_batches,
            "managers_per_batch": 2,
            "estimated_llm_invocations": total_batches * 2,
        }
        if self._is_truthy_env("LANGMEM_DEBUG_STDOUT"):
            print("[langmem-build-config] " + str(self._debug_state["build_memory_config"]), flush=True)
        batch_records: list[dict[str, Any]] = []
        self._debug_state["build_memory_batches"] = batch_records
        active_batch_record: dict[str, Any] | None = None
        build_mode = os.getenv("LANGMEM_BUILD_MODE", "sdk").strip().lower()
        if build_mode in {"fallback", "local", "local_fallback"}:
            semantic_entries = self._fallback_semantic_entries(history)
            procedural_entries = self._fallback_procedural_entries()
            implementation_mode = "local_fallback_requested"
            sdk_error = None
            semantic_updates = []
            procedural_updates = []
            if self._is_truthy_env("LANGMEM_DEBUG_STDOUT"):
                print(
                    f"[langmem-build-fallback] sample={sample.sample_id} memories={len(semantic_entries) + len(procedural_entries)}",
                    flush=True,
                )
            self._memory_entries = semantic_entries + procedural_entries
            self._org_state = OrganizationState(
                entries=list(self._memory_entries),
                levels={
                    "semantic": [entry.entry_id for entry in semantic_entries],
                    "procedural": [entry.entry_id for entry in procedural_entries],
                },
                priorities={
                    entry.entry_id: (1.0 if entry.metadata.get("memory_bucket") == "procedural" else 0.8)
                    for entry in self._memory_entries
                },
                raw={
                    "sdk": "langmem",
                    "implementation_mode": implementation_mode,
                    "sdk_error": sdk_error,
                    "semantic_updates": to_jsonable(semantic_updates),
                    "procedural_updates": to_jsonable(procedural_updates),
                },
            )
            self._debug_state["build_memory"] = {
                "implementation_mode": implementation_mode,
                "sdk_error": sdk_error,
                "semantic_updates": to_jsonable(semantic_updates),
                "procedural_updates": to_jsonable(procedural_updates),
            }
            return
        try:
            for start in range(0, len(messages), batch_size):
                batch = messages[start : start + batch_size]
                if not batch:
                    continue
                batch_record: dict[str, Any] = {
                    "batch": start // batch_size + 1,
                    "total_batches": total_batches,
                    "num_messages": len(batch),
                    "start": start,
                    "status": "started",
                }
                active_batch_record = batch_record
                batch_records.append(batch_record)
                if self._is_truthy_env("LANGMEM_DEBUG_STDOUT"):
                    print(
                        "[langmem-build-batch-start] "
                        + str(
                            {
                                "sample_id": sample.sample_id,
                                "batch": batch_record["batch"],
                                "total_batches": total_batches,
                                "num_messages": len(batch),
                            }
                        ),
                        flush=True,
                    )
                with self._langmem_llm_env():
                    semantic_update = self._semantic_manager.invoke({"messages": batch}, config=config)
                    procedural_update = self._procedural_manager.invoke({"messages": batch}, config=config)
                    semantic_updates.append(semantic_update)
                    procedural_updates.append(procedural_update)
                batch_record["status"] = "ok"
                batch_record["semantic_update_preview"] = str(to_jsonable(semantic_update))[:500]
                batch_record["procedural_update_preview"] = str(to_jsonable(procedural_update))[:500]
                if self._is_truthy_env("LANGMEM_DEBUG_STDOUT"):
                    print(
                        "[langmem-build-batch-done] "
                        + str({"sample_id": sample.sample_id, "batch": batch_record["batch"], "status": "ok"}),
                        flush=True,
                    )
            with self._langmem_llm_env():
                semantic_entries = self._search_entries(bucket="semantic", limit=_langmem_list_limit())
                procedural_entries = self._search_entries(bucket="procedural", limit=_langmem_list_limit())
            implementation_mode = "sdk"
            sdk_error = None
        except Exception as exc:
            if active_batch_record is not None:
                active_batch_record["status"] = "error"
                active_batch_record["error"] = f"{type(exc).__name__}: {exc}"
            allow_fallback = os.getenv("LANGMEM_ALLOW_LOCAL_FALLBACK", "false").strip().lower() not in {
                "0",
                "false",
                "no",
            }
            fallback_on_compat_error = os.getenv("LANGMEM_FALLBACK_ON_COMPAT_ERROR", "false").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if not allow_fallback and not (fallback_on_compat_error and _looks_like_openai_compatibility_error(exc)):
                raise
            semantic_entries = self._fallback_semantic_entries(history)
            procedural_entries = self._fallback_procedural_entries()
            implementation_mode = "local_fallback_after_sdk_compat_error" if _looks_like_openai_compatibility_error(exc) else "local_fallback_after_sdk_error"
            sdk_error = f"{type(exc).__name__}: {exc}"
        self._memory_entries = semantic_entries + procedural_entries
        self._org_state = OrganizationState(
            entries=list(self._memory_entries),
            levels={
                "semantic": [entry.entry_id for entry in semantic_entries],
                "procedural": [entry.entry_id for entry in procedural_entries],
            },
            priorities={
                entry.entry_id: (1.0 if entry.metadata.get("memory_bucket") == "procedural" else 0.8)
                for entry in self._memory_entries
            },
            raw={
                "sdk": "langmem",
                "implementation_mode": implementation_mode,
                "sdk_error": sdk_error,
                "semantic_updates": to_jsonable(semantic_updates),
                "procedural_updates": to_jsonable(procedural_updates),
            },
        )
        self._debug_state["build_memory"] = {
            "implementation_mode": implementation_mode,
            "sdk_error": sdk_error,
            "semantic_updates": to_jsonable(semantic_updates),
            "procedural_updates": to_jsonable(procedural_updates),
        }

    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        assert self._semantic_manager is not None
        if str(self._debug_state.get("build_memory", {}).get("implementation_mode") or "").startswith("local_fallback"):
            retrieved = self._fallback_retrieve(query=query, top_k=top_k)
            self._retrieval_result = RetrievalResult(
                query=query,
                retrieved_entries=retrieved,
                top_k=top_k,
                raw={
                    "sdk": "langmem",
                    "implementation_mode": self._debug_state.get("build_memory", {}).get("implementation_mode"),
                    "sdk_error": self._debug_state.get("build_memory", {}).get("sdk_error"),
                },
            )
            return self._retrieval_result
        with self._langmem_llm_env():
            results = self._semantic_manager.search(query=query, limit=top_k, config=self._langgraph_config())
        retrieved: list[RetrievedMemory] = []
        raw_items: list[dict[str, Any]] = []
        for rank, item in enumerate(results, start=1):
            raw = _as_dict(item)
            raw_items.append(raw)
            retrieved.append(
                RetrievedMemory(
                    entry_id=f"semantic:{raw.get('key') or rank}",
                    content=_extract_content(raw),
                    score=float(raw["score"]) if raw.get("score") is not None else None,
                    rank=rank,
                    source_ids=_extract_source_ids_from_value(raw),
                    metadata={
                        **_extract_value_metadata(raw),
                        "namespace": raw.get("namespace"),
                        "value": raw.get("value"),
                    },
                )
            )
        self._retrieval_result = RetrievalResult(
            query=query,
            retrieved_entries=retrieved,
            top_k=top_k,
            raw={"sdk": "langmem", "results": raw_items},
        )
        return self._retrieval_result

    def build_prompt(self, query: str, retrieved: RetrievalResult, sample: UnifiedSample) -> PromptRecord:
        procedural_entries = [
            entry for entry in self._memory_entries if entry.metadata.get("memory_bucket") == "procedural"
        ]
        procedural_lines = [entry.content for entry in procedural_entries]
        system_prompt = "\n".join(procedural_lines) if procedural_lines else None
        focused_retrieved_entries = [
            RetrievedMemory(
                entry_id=item.entry_id,
                content=_focused_content(query, item.content),
                score=item.score,
                rank=item.rank,
                source_ids=list(item.source_ids),
                metadata=dict(item.metadata or {}),
            )
            for item in retrieved.retrieved_entries
        ]
        memory_context = self._format_retrieved_memory_block(focused_retrieved_entries, sample=sample)
        full_prompt = (
            "You are answering a benchmark question.\n"
            f"{self._answering_instructions()}\n"
            f"System Guidance:\n{system_prompt or ''}\n"
            f"Memory Block:\n{memory_context}\n"
            f"Question: {query}\n"
            "Answer:"
        )
        injected_ids = [item.entry_id for item in focused_retrieved_entries] + [entry.entry_id for entry in procedural_entries]
        injection_positions = {item.entry_id: "middle_context" for item in focused_retrieved_entries}
        injection_positions.update({entry.entry_id: "system_prompt" for entry in procedural_entries})
        self._prompt_record = PromptRecord(
            system_prompt=system_prompt,
            user_prompt=f"Question: {query}",
            memory_context=memory_context,
            full_prompt=full_prompt,
            injected_entry_ids=injected_ids,
            token_count=self._estimate_token_count(full_prompt),
            injection_positions=injection_positions,
            raw={
                "system": "langmem",
                "sdk": "langmem",
                "memory_context_mode": "query_focused_sections",
            },
        )
        return self._prompt_record

    def generate_answer(self, prompt: PromptRecord, sample: UnifiedSample) -> AnswerRecord:
        gateway = self._resolve_generation_gateway(fallback_provider=self._sdk_provider)
        response = gateway.complete_text(prompt=prompt.full_prompt, model=self.generation_model, temperature=0.0)
        self._answer_record = AnswerRecord(
            answer=response.text,
            raw_response={**response.to_dict(), "system": self.system_name},
            latency=response.latency,
            token_usage={
                "prompt_tokens_est": prompt.token_count,
                "completion_tokens_est": len(response.text.split()),
            },
        )
        return self._answer_record

    def _resolve_sdk_provider(self) -> str:
        provider = (
            os.getenv("LANGMEM_PROVIDER")
            or self.llm_provider_name
            or os.getenv("LLM_PROVIDER")
            or "openai"
        ).strip().lower()
        if provider == "mock":
            raise ValueError("LangMem 真实接入不支持 mock provider，请设置 OPENAI_API_KEY 并使用 openai。")
        return provider

    def _ensure_provider_credentials(self) -> None:
        if (self._sdk_provider == "openai" or self._embed_provider == "openai") and not (
            os.getenv("LANGMEM_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        ):
            raise RuntimeError("LangMem 需要 OPENAI_API_KEY 才能完成真实 memory 提取、嵌入与检索。")

    def _model_handle(self, model_name: str, *, provider: str) -> str:
        return model_name if ":" in model_name else f"{provider}:{model_name}"

    def _langmem_model_handle(self) -> Any:
        model_name = os.getenv("LANGMEM_LLM_MODEL") or self.generation_model
        provider = os.getenv("LANGMEM_LLM_PROVIDER") or self._sdk_provider
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            base_url = os.getenv("LANGMEM_LLM_BASE_URL") or openai_base_url()
            kwargs: dict[str, Any] = {
                "model": model_name,
                "api_key": os.getenv("LANGMEM_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
                "timeout": float(os.getenv("LANGMEM_LLM_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "120"))),
                "max_retries": int(os.getenv("LANGMEM_LLM_MAX_RETRIES", os.getenv("LLM_MAX_RETRIES", "2"))),
            }
            token_limit = _langmem_llm_max_tokens()
            if is_gpt5_like(model_name):
                kwargs["max_completion_tokens"] = token_limit
            else:
                kwargs["max_tokens"] = token_limit
            if base_url:
                kwargs["base_url"] = base_url
            if self._is_truthy_env("LANGMEM_FORCE_TEMPERATURE") or not is_gpt5_like(model_name):
                kwargs["temperature"] = 0.0
            if self._is_truthy_env("LANGMEM_DISABLE_THINKING", "true") and not is_gpt5_like(model_name):
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            return ChatOpenAI(**kwargs)
        return self._model_handle(model_name, provider=provider)

    def _model_debug_name(self, model_handle: Any) -> str:
        if isinstance(model_handle, str):
            return model_handle
        model_name = getattr(model_handle, "model_name", None) or getattr(model_handle, "model", None)
        return f"{type(model_handle).__name__}:{model_name or ''}".rstrip(":")

    @contextmanager
    def _langmem_llm_env(self):
        overrides = {
            "OPENAI_API_KEY": os.getenv("LANGMEM_LLM_API_KEY"),
            "OPENAI_BASE_URL": os.getenv("LANGMEM_LLM_BASE_URL"),
        }
        old_values = {key: os.environ.get(key) for key in overrides}
        try:
            for key, value in overrides.items():
                if value:
                    os.environ[key] = value
            yield
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts, model=self.embedding_model)

    def _embedding_dims(self, model_name: str) -> int:
        override = os.getenv("LANGMEM_EMBED_DIMS")
        if override and override.isdigit():
            return int(override)
        if "bge-m3" in model_name.lower():
            return 1024
        if "qwen3-embedding-0.6b" in model_name.lower():
            return 1024
        if "qwen3-embedding-4b" in model_name.lower():
            return 2560
        if "qwen3-embedding-8b" in model_name.lower():
            return 4096
        if "3-large" in model_name:
            return 3072
        return 1536

    def _langgraph_config(self) -> dict[str, Any]:
        return {"configurable": {"langgraph_user_id": self._sample_id}}

    def _search_entries(self, *, bucket: str, limit: int) -> list[MemoryEntry]:
        manager = self._semantic_manager if bucket == "semantic" else self._procedural_manager
        assert manager is not None
        results = manager.search(limit=limit, config=self._langgraph_config())
        entries: list[MemoryEntry] = []
        for index, item in enumerate(results, start=1):
            raw = _as_dict(item)
            entries.append(
                MemoryEntry(
                    entry_id=f"{bucket}:{raw.get('key') or index}",
                    content=_extract_content(raw),
                    raw=raw,
                    created_at=raw.get("created_at"),
                    updated_at=raw.get("updated_at"),
                    source_ids=_extract_source_ids_from_value(raw),
                    metadata={
                        **_extract_value_metadata(raw),
                        "memory_bucket": bucket,
                        "namespace": raw.get("namespace"),
                        "score": raw.get("score"),
                        "value": raw.get("value"),
                    },
                )
            )
        return entries

    def _fallback_semantic_entries(self, history: list[dict]) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        for index, turn in enumerate(history, start=1):
            content = str(turn.get("content") or turn.get("text") or "").strip()
            if not content:
                continue
            turn_id = str(
                turn.get("turn_id")
                or turn.get("source_id")
                or turn.get("id")
                or f"turn_{index}"
            )
            speaker = str(turn.get("speaker") or turn.get("role") or "").strip()
            source_ids = [turn_id]
            session_id = turn.get("session_id")
            if session_id:
                source_ids.append(str(session_id))
            entries.append(
                MemoryEntry(
                    entry_id=f"semantic:fallback:{turn_id}",
                    content=f"{speaker}: {content}" if speaker else content,
                    source_ids=source_ids,
                    metadata={
                        "memory_bucket": "semantic",
                        "implementation_mode": "local_fallback_after_sdk_error",
                        "session_id": session_id,
                        "session_date": turn.get("timestamp"),
                        "speaker": speaker,
                    },
                    raw=to_jsonable(turn),
                )
            )
        return entries

    def _fallback_procedural_entries(self) -> list[MemoryEntry]:
        return [
            MemoryEntry(
                entry_id="procedural:fallback:answer_policy",
                content="Answer benchmark questions briefly using only the provided memory evidence.",
                source_ids=[],
                metadata={
                    "memory_bucket": "procedural",
                    "implementation_mode": "local_fallback_after_sdk_error",
                },
                raw={"source": "langmem_local_fallback"},
            )
        ]

    def _fallback_retrieve(self, *, query: str, top_k: int) -> list[RetrievedMemory]:
        semantic_entries = [
            entry for entry in self._memory_entries if entry.metadata.get("memory_bucket") == "semantic"
        ]
        scored = [
            (_lexical_score(query, entry.content), index, entry)
            for index, entry in enumerate(semantic_entries)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        retrieved: list[RetrievedMemory] = []
        for rank, (score, _, entry) in enumerate(scored[:top_k], start=1):
            retrieved.append(
                RetrievedMemory(
                    entry_id=entry.entry_id,
                    content=entry.content,
                    score=score,
                    rank=rank,
                    source_ids=list(entry.source_ids),
                    metadata={
                        **entry.metadata,
                        "implementation_mode": "local_fallback_after_sdk_error",
                    },
                )
            )
        return retrieved

    def validate_setup(self) -> dict[str, Any]:
        if (self._sdk_provider == "openai" or self._embed_provider == "openai") and not (
            os.getenv("LANGMEM_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        ):
            return {"system": self.system_name, "ready": False, "provider": self._sdk_provider, "message": "缺少 OPENAI_API_KEY。"}
        return {"system": self.system_name, "ready": True, "provider": self._sdk_provider, "message": ""}
