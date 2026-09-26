from __future__ import annotations

import importlib.util
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

from map_platform.datasets.schema import UnifiedSample
from map_platform.llm.gateway import embed_texts, embedding_base_url, embedding_client_kwargs, openai_base_url
from map_platform.memory_systems.base import (
    AnswerRecord,
    BaseMemorySystemAdapter,
    MemoryEntry,
    OrganizationState,
    PromptRecord,
    RetrievalResult,
    RetrievedMemory,
)
from map_platform.utils.live_progress import update_live_progress


class MemoryBankAdapter(BaseMemorySystemAdapter):
    system_name = "memorybank"

    # Verbatim English summary instruction from the official
    # memory_bank/summarize_memory.py snapshot.  The benchmark adapter changes
    # only the input schema and model transport.
    _DAY_SUMMARY_PROMPT = (
        "Please summarize the following dialogue as concisely as possible, "
        "extracting the main themes and key information. If there are multiple "
        "key events, you may summarize them separately. Dialogue content:\n"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Any | None = None
        self._embedding_cache: dict[str, list[float]] = {}

    def reset(self, sample_id: str) -> None:
        super().reset(sample_id)
        self._ensure_openai_ready()
        self._debug_state["official_source"] = {
            "repo": "https://github.com/zhongwanjun/MemoryBank-SiliconFriend",
            "commit": "cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9",
            "local_snapshot": "external_systems/MemoryBank",
            "implementation_mode": "official_source_adapted_to_unified_benchmarks",
            "official_files": [
                "memory_bank/summarize_memory.py",
                "memory_bank/memory_retrieval/local_doc_qa.py",
            ],
            "forgetting": "disabled_as_in_official_default_configuration",
        }

    def build_memory(self, history: list[dict[str, Any]], sample: UnifiedSample) -> None:
        groups = self._group_history(history)
        entries: list[MemoryEntry] = []
        levels: dict[str, list[str]] = {}
        traces: list[dict[str, Any]] = []
        total_groups = len(groups)
        update_live_progress(
            stage="build_memory", system=self.system_name, sample_id=sample.sample_id,
            current_operation="memorybank_group_summary", group=0,
            total_groups=total_groups, memory_entries=0, history_turns=len(history),
        )

        for group_index, (group_id, turns) in enumerate(groups.items(), start=1):
            dialogue_records = self._dialogue_records(turns)
            group_text = self._render_turns(turns)
            memory_entry_ids: list[str] = []
            for index, (dialogue, source_ids) in enumerate(dialogue_records, start=1):
                entry_id = f"{group_id}:memory:{index}"
                entry = MemoryEntry(
                    entry_id=entry_id,
                    content=f"Conversation content on {group_id}: {dialogue}",
                    raw={"group_id": group_id, "kind": "dialogue_memory", "source_text": dialogue},
                    created_at=self._group_timestamp(turns),
                    updated_at=self._group_timestamp(turns),
                    source_ids=source_ids,
                    metadata={
                        "memory_bucket": "episodic",
                        "group_id": group_id,
                        "turn_count": len(turns),
                    },
                )
                entries.append(entry)
                memory_entry_ids.append(entry_id)

            summary_text = self._summarize_group(group_text)
            summary_id = f"{group_id}:summary"
            summary_entry = MemoryEntry(
                entry_id=summary_id,
                content=summary_text,
                raw={"group_id": group_id, "kind": "period_summary", "source_text": group_text},
                created_at=self._group_timestamp(turns),
                updated_at=self._group_timestamp(turns),
                source_ids=self._turn_source_ids(turns),
                metadata={
                    "memory_bucket": "summary",
                    "group_id": group_id,
                    "turn_count": len(turns),
                },
            )
            entries.append(summary_entry)

            levels[group_id] = [summary_id] + memory_entry_ids
            traces.append(
                {
                    "group_id": group_id,
                    "num_turns": len(turns),
                    "num_dialogue_memories": len(dialogue_records),
                    "summary": summary_text,
                }
            )
            update_live_progress(
                stage="build_memory", system=self.system_name, sample_id=sample.sample_id,
                current_operation="memorybank_group_summary", group=group_index,
                total_groups=total_groups, group_id=group_id,
                completed_groups=group_index, memory_entries=len(entries),
            )

        total_entries = len(entries)
        for entry_index, entry in enumerate(entries, start=1):
            self._embed(entry.content)
            update_live_progress(
                stage="build_memory", system=self.system_name, sample_id=sample.sample_id,
                current_operation="memorybank_embedding", embedding=entry_index,
                total_embeddings=total_entries, completed_embeddings=entry_index,
                memory_entries=total_entries,
            )

        self._memory_entries = entries
        self._org_state = OrganizationState(
            entries=list(entries),
            levels=levels,
            raw={
                "system": "memorybank",
                "grouping": "time_bucket",
                "official_algorithm": "dialogue_plus_daily_summary_vector_retrieval",
                "forgetting_enabled": False,
            },
        )
        self._debug_state["build_memory"] = {"groups": traces}

    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._memory_entries:
            semantic = self._similarity(query, entry.content)
            scored.append((semantic, entry))
        scored.sort(key=lambda item: item[0], reverse=True)

        retrieved: list[RetrievedMemory] = []
        raw_rows: list[dict[str, Any]] = []
        for rank, (score, entry) in enumerate(scored[:top_k], start=1):
            retrieved.append(
                RetrievedMemory(
                    entry_id=entry.entry_id,
                    content=entry.content,
                    score=score,
                    rank=rank,
                    source_ids=list(entry.source_ids),
                    metadata={
                        "memory_bucket": entry.metadata.get("memory_bucket"),
                        "group_id": entry.metadata.get("group_id"),
                        "session_date": entry.created_at,
                        "created_at": entry.created_at,
                        "updated_at": entry.updated_at,
                        "semantic_score": score,
                    },
                )
            )
            raw_rows.append(
                {
                    "entry_id": entry.entry_id,
                    "semantic_score": score,
                    "final_score": score,
                }
            )

        self._retrieval_result = RetrievalResult(
            query=query,
            retrieved_entries=retrieved,
            top_k=top_k,
            raw={
                "system": "memorybank",
                "retrieval": "official_style_vector_similarity",
                "results": raw_rows,
            },
        )
        return self._retrieval_result

    def build_prompt(self, query: str, retrieved: RetrievalResult, sample: UnifiedSample) -> PromptRecord:
        history_index = self._history_index(sample)
        summary_lines = [
            self._format_retrieved_memory(item, history_index=history_index)
            for item in retrieved.retrieved_entries
            if item.metadata.get("memory_bucket") == "summary"
        ]
        episodic_lines = [
            self._format_retrieved_memory(item, history_index=history_index)
            for item in retrieved.retrieved_entries
            if item.metadata.get("memory_bucket") != "summary"
        ]
        memory_context = ""
        if summary_lines:
            memory_context += "Period Summaries:\n" + "\n".join(summary_lines) + "\n"
        if episodic_lines:
            memory_context += "Detailed Memories:\n" + "\n".join(episodic_lines)
        full_prompt = (
            "You are answering a benchmark question using a memory bank.\n"
            f"{self._answering_instructions()}\n"
            "Use the memory summaries first, and use detailed memories to resolve specifics.\n"
            f"Memory Block:\n{memory_context.strip()}\n"
            f"Question: {query}\n"
            "Answer:"
        )
        self._prompt_record = PromptRecord(
            system_prompt="Use recent and high-strength memories when conflicts appear.",
            user_prompt=f"Question: {query}",
            memory_context=memory_context.strip(),
            full_prompt=full_prompt,
            injected_entry_ids=[item.entry_id for item in retrieved.retrieved_entries],
            token_count=self._estimate_token_count(full_prompt),
            injection_positions={item.entry_id: "middle_context" for item in retrieved.retrieved_entries},
            raw={"system": self.system_name, "memory_layout": "summary_plus_detail"},
        )
        return self._prompt_record

    def generate_answer(self, prompt: PromptRecord, sample: UnifiedSample) -> AnswerRecord:
        response = self.llm.complete_text(prompt=prompt.full_prompt, model=self.generation_model, temperature=0.0)
        self._answer_record = AnswerRecord(
            answer=response.text,
            raw_response=response.to_dict(),
            latency=response.latency,
            token_usage={"prompt_tokens_est": prompt.token_count, "completion_tokens_est": len(response.text.split())},
        )
        return self._answer_record

    def _group_history(self, history: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for index, turn in enumerate(history, start=1):
            group_id = self._bucket_id(turn, index)
            grouped[group_id].append(turn)
        if not grouped:
            grouped["group_0001"] = []
        return dict(grouped)

    def _bucket_id(self, turn: dict[str, Any], index: int) -> str:
        timestamp = str(turn.get("timestamp") or "").strip()
        date_key = _date_key_from_timestamp(timestamp)
        if date_key:
            return date_key
        session_id = str(turn.get("session_id") or "").strip()
        if session_id:
            return f"session_{session_id}"
        return f"chunk_{((index - 1) // 20) + 1:04d}"

    def _group_timestamp(self, turns: list[dict[str, Any]]) -> str | None:
        for turn in turns:
            timestamp = str(turn.get("timestamp") or "").strip()
            if timestamp:
                return timestamp
        return None

    def _render_turns(self, turns: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for turn in turns:
            speaker = str(turn.get("speaker") or "unknown")
            text = str(turn.get("text") or "").strip()
            if text:
                turn_id = str(turn.get("turn_id") or turn.get("source_id") or "").strip()
                session_id = str(turn.get("session_id") or "").strip()
                timestamp = str(turn.get("timestamp") or "").strip()
                header_parts = []
                if turn_id:
                    header_parts.append(f"turn_id={turn_id}")
                if session_id:
                    header_parts.append(f"session_id={session_id}")
                if timestamp:
                    header_parts.append(f"session_date={timestamp}")
                header = f"[{'; '.join(header_parts)}] " if header_parts else ""
                lines.append(f"{header}{speaker}: {text}")
        return "\n".join(lines)

    def _turn_source_ids(self, turns: list[dict[str, Any]]) -> list[str]:
        return [
            str(turn.get("turn_id") or turn.get("source_id") or "").strip()
            for turn in turns
            if str(turn.get("turn_id") or turn.get("source_id") or "").strip()
        ]

    def _dialogue_records(self, turns: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
        """Map alternating benchmark turns to the official query/response memory unit."""
        records: list[tuple[str, list[str]]] = []
        index = 0
        while index < len(turns):
            pair = turns[index : index + 2]
            rendered = self._render_turns(pair).strip()
            if rendered:
                records.append((rendered, self._turn_source_ids(pair)))
            index += 2
        return records

    def _summarize_group(self, group_text: str) -> str:
        prompt = f"{self._DAY_SUMMARY_PROMPT}{group_text}\nSummarization:"
        self._debug_state["active_call"] = {
            "stage": "llm_group_summary",
            "base_url": os.getenv("MEMORY_BUILD_OPENAI_BASE_URL") or openai_base_url(),
            "model": self._stage_generation_model(stage="build_memory"),
            "api_key_source": api_key_source("OPENAI_API_KEY"),
        }
        response = self._complete_text_for_stage(prompt=prompt, stage="build_memory", temperature=0.0)
        summary = response.text.strip()
        if not summary and os.getenv("MEMORY_EVAL_FORMAL_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError("MemoryBank group summarization returned an empty response")
        return summary or group_text[:240]

    def _similarity(self, left: str, right: str) -> float:
        left_vector = self._embed(left)
        right_vector = self._embed(right)
        numerator = sum(a * b for a, b in zip(left_vector, right_vector))
        left_norm = math.sqrt(sum(a * a for a in left_vector))
        right_norm = math.sqrt(sum(b * b for b in right_vector))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))

    def _embed(self, text: str) -> list[float]:
        key = text.strip()
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached
        self._debug_state["active_call"] = {
            "stage": "embedding_similarity",
            "base_url": embedding_base_url(),
            "model": self.embedding_model,
            "api_key_source": api_key_source("EMBEDDING_API_KEY", fallback_key="OPENAI_API_KEY"),
        }
        vector = embed_texts([text], model=self.embedding_model)[0]
        self._embedding_cache[key] = vector
        return vector

    def _ensure_openai_ready(self) -> None:
        if (self.llm_provider_name or "").lower() == "openai" and importlib.util.find_spec("openai") is None:
            raise ModuleNotFoundError(
                "MemoryBank requires the 'openai' Python package. Install it with: python -m pip install 'openai>=1.52.0,<=1.90.0'"
            )
        if (self.llm_provider_name or "mock") == "mock":
            raise ValueError("MemoryBank 真实适配器不支持 mock provider，请使用 openai 并设置 OPENAI_API_KEY。")
        if (self.llm_provider_name or "").lower() != "openai":
            raise ValueError("当前 MemoryBank 真实适配器仅支持 openai provider。")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("MemoryBank 真实适配器需要 OPENAI_API_KEY。")

    def _client_instance(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._debug_state["embedding_client"] = sanitized_client_kwargs(embedding_client_kwargs())
            self._client = OpenAI(**embedding_client_kwargs())
        return self._client

    def validate_setup(self) -> dict[str, Any]:
        provider = (self.llm_provider_name or "mock").lower()
        if provider == "openai" and importlib.util.find_spec("openai") is None:
            return {
                "system": self.system_name,
                "ready": False,
                "provider": provider,
                "message": "Missing Python package: openai. Install it with `python -m pip install 'openai>=1.52.0,<=1.90.0'`.",
            }
        if provider == "mock":
            return {"system": self.system_name, "ready": False, "provider": provider, "message": "不支持 mock provider。"}
        if provider != "openai":
            return {"system": self.system_name, "ready": False, "provider": provider, "message": "当前仅支持 openai provider。"}
        if not os.getenv("OPENAI_API_KEY"):
            return {"system": self.system_name, "ready": False, "provider": provider, "message": "缺少 OPENAI_API_KEY。"}
        return {"system": self.system_name, "ready": True, "provider": provider, "message": ""}


def api_key_source(primary_key: str, *, fallback_key: str | None = None) -> str:
    if os.getenv(primary_key):
        return primary_key
    if fallback_key and os.getenv(fallback_key):
        return fallback_key
    return "missing"


def sanitized_client_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(kwargs)
    api_key = sanitized.get("api_key")
    if api_key:
        text = str(api_key)
        sanitized["api_key"] = f"{text[:3]}...{text[-3:]}" if len(text) > 8 else "***"
    return sanitized


def _date_key_from_timestamp(timestamp: str) -> str | None:
    text = str(timestamp or "").strip()
    if not text:
        return None
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_match:
        return iso_match.group(1)
    for pattern in (
        r"\b\d{1,2}\s+[A-Za-z]+\s*,?\s*\d{4}\b",
        r"\b[A-Za-z]+\s+\d{1,2}\s*,?\s*\d{4}\b",
    ):
        match = re.search(pattern, text)
        if not match:
            continue
        raw = match.group(0).replace(",", "")
        for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(raw, fmt).date().isoformat()
            except ValueError:
                continue
    return None
