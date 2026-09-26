from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from collections.abc import Iterator
from typing import Any

from map_platform.datasets.schema import UnifiedSample
from map_platform.llm.gateway import LLMGateway, normalize_provider
from map_platform.utils.serialization import to_jsonable


@dataclass
class MemoryEntry:
    entry_id: str
    content: str
    raw: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None
    source_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class OrganizationState:
    entries: list[MemoryEntry]
    levels: dict[str, Any] | None = None
    weights: dict[str, Any] | None = None
    priorities: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class RetrievedMemory:
    entry_id: str
    content: str
    score: float | None
    rank: int
    source_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class RetrievalResult:
    query: str
    retrieved_entries: list[RetrievedMemory]
    top_k: int
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class PromptRecord:
    system_prompt: str | None
    user_prompt: str
    memory_context: str
    full_prompt: str
    injected_entry_ids: list[str] = field(default_factory=list)
    token_count: int | None = None
    injection_positions: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class AnswerRecord:
    answer: str
    raw_response: dict[str, Any] = field(default_factory=dict)
    latency: float | None = None
    token_usage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


class BaseMemorySystemAdapter(ABC):
    system_name = "base"

    def __init__(
        self,
        *,
        generation_model: str = "",
        embedding_model: str = "BAAI/bge-m3",
        llm_provider: str | None = None,
    ) -> None:
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        normalized_provider, provider_base_url = normalize_provider(llm_provider)
        self.llm_provider_name = None if normalized_provider == "mock" and not llm_provider else normalized_provider
        self.llm_provider_base_url = provider_base_url
        self.llm = LLMGateway(provider=llm_provider)
        self._memory_entries: list[MemoryEntry] = []
        self._org_state = OrganizationState(entries=[])
        self._retrieval_result = RetrievalResult(query="", retrieved_entries=[], top_k=0)
        self._prompt_record = PromptRecord(system_prompt=None, user_prompt="", memory_context="", full_prompt="")
        self._answer_record = AnswerRecord(answer="")
        self._debug_state: dict[str, Any] = {}
        self._sample_id = ""

    def reset(self, sample_id: str) -> None:
        self._sample_id = sample_id
        self._memory_entries = []
        self._org_state = OrganizationState(entries=[])
        self._retrieval_result = RetrievalResult(query="", retrieved_entries=[], top_k=0)
        self._prompt_record = PromptRecord(system_prompt=None, user_prompt="", memory_context="", full_prompt="")
        self._answer_record = AnswerRecord(answer="")
        self._debug_state = {}

    @abstractmethod
    def build_memory(self, history: list[dict[str, Any]], sample: UnifiedSample) -> None:
        raise NotImplementedError

    def get_memory_entries(self) -> list[MemoryEntry]:
        return list(self._memory_entries)

    def get_organization_state(self) -> OrganizationState:
        return self._org_state

    @abstractmethod
    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        raise NotImplementedError

    def build_prompt(self, query: str, retrieved: RetrievalResult, sample: UnifiedSample) -> PromptRecord:
        memory_context = self._format_retrieved_memory_block(retrieved.retrieved_entries, sample=sample)
        full_prompt = (
            "You are an intelligent memory assistant answering questions based on historical conversation memories.\n\n"
            f"{self._answering_instructions()}\n\n"
            f"# Historical Conversation Memories\n{memory_context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        self._prompt_record = PromptRecord(
            system_prompt=None,
            user_prompt=f"Question: {query}",
            memory_context=memory_context,
            full_prompt=full_prompt,
            injected_entry_ids=[item.entry_id for item in retrieved.retrieved_entries],
            token_count=self._estimate_token_count(full_prompt),
            injection_positions={item.entry_id: "middle_context" for item in retrieved.retrieved_entries},
            raw={"adapter": self.system_name},
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

    def dump_debug_state(self) -> dict[str, Any]:
        payload = {
            "sample_id": self._sample_id,
            "system": self.system_name,
            "memory_entries": [entry.to_dict() for entry in self._memory_entries],
            "organization_state": self._org_state.to_dict(),
            "retrieval": self._retrieval_result.to_dict(),
            "prompt_record": self._prompt_record.to_dict(),
            "answer_record": self._answer_record.to_dict(),
            **self._debug_state,
        }
        return to_jsonable(payload)

    def _stage_generation_model(self, *, stage: str) -> str:
        if stage == "build_memory":
            return os.getenv("MEMORY_BUILD_MODEL") or os.getenv("MEMORY_EXTRACTION_MODEL") or self.generation_model
        return self.generation_model

    def _complete_text_for_stage(self, *, prompt: str, stage: str, temperature: float = 0.0):
        model = self._stage_generation_model(stage=stage)
        with _stage_openai_env(stage):
            return self.llm.complete_text(prompt=prompt, model=model, temperature=temperature)

    def validate_setup(self) -> dict[str, Any]:
        return {
            "system": self.system_name,
            "ready": True,
            "provider": self.llm_provider_name or self.llm.provider,
            "message": "",
        }

    def close(self) -> None:
        return None

    def _normalize_role(self, speaker: Any) -> str:
        lowered = str(speaker or "").strip().lower()
        if lowered in {"assistant", "agent", "system", "bot", "ai"}:
            return "assistant" if lowered != "system" else "system"
        return "user"

    def _history_as_messages(self, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in history:
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            messages.append({"role": self._normalize_role(turn.get("speaker")), "content": text})
        return messages

    def _history_as_sourced_messages(self, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in history:
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            speaker = str(turn.get("speaker") or "").strip()
            turn_id = str(turn.get("turn_id") or turn.get("source_id") or turn.get("id") or "").strip()
            session_id = str(turn.get("session_id") or "").strip()
            timestamp = str(turn.get("timestamp") or "").strip()
            metadata_parts = []
            if turn_id:
                metadata_parts.append(f"turn_id={turn_id}")
            if session_id:
                metadata_parts.append(f"session_id={session_id}")
            if timestamp:
                metadata_parts.append(f"session_date={timestamp}")
            if speaker:
                metadata_parts.append(f"speaker={speaker}")
            header = f"[source {'; '.join(metadata_parts)}]" if metadata_parts else "[source]"
            body = f"{speaker}: {text}" if speaker else text
            messages.append(
                {
                    "role": self._normalize_role(turn.get("speaker")),
                    "content": f"{header}\n{body}",
                }
            )
        return messages

    def _history_as_text(self, history: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, turn in enumerate(history, start=1):
            speaker = str(turn.get("speaker") or "unknown")
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"[{index}] {speaker}: {text}")
        return "\n".join(lines)

    def _answering_instructions(self) -> str:
        return (
            "Answer the question using only the factual content in the Memory Block.\n"
            "For date or time questions, resolve relative references (yesterday, last week, this month, etc.) "
            "to absolute dates using the session timestamp shown in each message. "
            "For example, if a message on [2023/05/08] says 'yesterday', the answer should be 'May 7, 2023'.\n"
            "For multi-step questions, chain evidence from multiple memory items to form your answer.\n"
            "For yes/no questions, find direct supporting or contradicting evidence before answering.\n"
            "If the question asks about events, people, or facts, provide a complete answer in 1-3 sentences with all relevant details.\n"
            "If the Memory Block does not contain enough evidence to answer the question, answer exactly: Not enough information.\n"
            "Do not include turn_id, session_id, or other provenance markers in your answer."
        )

    def _format_retrieved_memory_block(
        self,
        entries: list[RetrievedMemory],
        *,
        sample: UnifiedSample | None = None,
    ) -> str:
        history_index = self._history_index(sample)
        return "\n".join(self._format_retrieved_memory(item, history_index=history_index) for item in entries)

    def _format_retrieved_memory(
        self,
        item: RetrievedMemory,
        *,
        history_index: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        metadata = dict(item.metadata or {})
        source_ids = [str(source_id) for source_id in item.source_ids if str(source_id).strip()]
        source_metadata = self._source_metadata(source_ids, history_index or {})
        header_parts: list[str] = []
        if item.entry_id:
            header_parts.append(f"memory_id={item.entry_id}")
        if source_ids:
            header_parts.append(f"source_ids={','.join(source_ids)}")
        session_id = str(
            metadata.get("session_id")
            or metadata.get("session")
            or source_metadata.get("session_id")
            or ""
        ).strip()
        if session_id:
            header_parts.append(f"session_id={session_id}")
        session_date = str(
            metadata.get("session_date")
            or metadata.get("created_at")
            or metadata.get("timestamp")
            or metadata.get("updated_at")
            or source_metadata.get("session_date")
            or ""
        ).strip()
        if session_date:
            header_parts.append(f"session_date={session_date}")
        speaker = str(metadata.get("speaker") or metadata.get("role") or source_metadata.get("speaker") or "").strip()
        if speaker:
            header_parts.append(f"speaker={speaker}")
        header = f"[{'; '.join(header_parts)}]" if header_parts else "[memory]"
        content = self._strip_redundant_speaker_prefix(item.content, speaker=speaker)
        return f"- {header}\n  {content}"

    def _strip_redundant_speaker_prefix(self, content: Any, *, speaker: str = "") -> str:
        text = str(content or "").strip()
        if speaker:
            pattern = rf"^{re.escape(speaker)}\s*:\s*"
            text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
        return text

    def _history_index(self, sample: UnifiedSample | None) -> dict[str, dict[str, Any]]:
        if sample is None:
            return {}
        index: dict[str, dict[str, Any]] = {}
        for turn in sample.history:
            turn_id = str(turn.get("turn_id") or turn.get("source_id") or turn.get("id") or "").strip()
            session_id = str(turn.get("session_id") or "").strip()
            values = {
                "session_id": session_id,
                "session_date": turn.get("timestamp"),
                "speaker": turn.get("speaker"),
            }
            if turn_id:
                index[turn_id] = values
            if session_id and session_id not in index:
                index[session_id] = values
        return index

    def _source_metadata(
        self,
        source_ids: list[str],
        history_index: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        for source_id in source_ids:
            if source_id in history_index:
                return history_index[source_id]
        return {}

    def _estimate_token_count(self, *parts: str | None) -> int:
        text = "\n".join(part for part in parts if part)
        return len(text.split())

    def _resolve_generation_gateway(self, *, fallback_provider: str | None = None) -> LLMGateway:
        if self.llm.provider == "mock" and fallback_provider and fallback_provider != "mock":
            return LLMGateway(provider=fallback_provider)
        return self.llm


@contextmanager
def _stage_openai_env(stage: str) -> Iterator[None]:
    if stage != "build_memory":
        yield
        return
    overrides = {
        "OPENAI_BASE_URL": os.getenv("MEMORY_BUILD_OPENAI_BASE_URL") or os.getenv("MEMORY_EXTRACTION_OPENAI_BASE_URL"),
        "OPENAI_API_BASE": None,
        "OPENAI_API_BASE_URL": None,
        "OPENAI_API_KEY": os.getenv("MEMORY_BUILD_OPENAI_API_KEY") or os.getenv("MEMORY_EXTRACTION_OPENAI_API_KEY"),
        "LLM_MAX_TOKENS": (
            os.getenv("MEMORY_BUILD_LLM_MAX_TOKENS")
            or os.getenv("MEMORY_EXTRACTION_LLM_MAX_TOKENS")
            or os.getenv("MEMORY_BUILD_MAX_TOKENS")
            or os.getenv("MEMORY_EXTRACTION_MAX_TOKENS")
            or "1024"
        ),
    }
    active = {key: value for key, value in overrides.items() if value is not None}
    if not active:
        yield
        return
    previous = {key: os.environ.get(key) for key in active}
    try:
        for key, value in active.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
