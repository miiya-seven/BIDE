from __future__ import annotations

from map_platform.datasets.schema import UnifiedSample
from map_platform.memory_systems.base import (
    BaseMemorySystemAdapter,
    MemoryEntry,
    OrganizationState,
    PromptRecord,
    RetrievalResult,
    RetrievedMemory,
)


class FullContextAdapter(BaseMemorySystemAdapter):
    system_name = "full_context"

    def build_memory(self, history: list[dict], sample: UnifiedSample) -> None:
        self._memory_entries = [
            MemoryEntry(
                entry_id=str(turn.get("turn_id") or f"turn_{idx + 1}"),
                content=f"{turn.get('speaker', 'unknown')}: {turn.get('text', '')}",
                raw=dict(turn),
                created_at=turn.get("timestamp"),
                updated_at=turn.get("timestamp"),
                source_ids=[str(turn.get("turn_id") or f"turn_{idx + 1}")],
                metadata={
                    "role": turn.get("speaker"),
                    "speaker": turn.get("speaker"),
                    "session_id": turn.get("session_id"),
                    "session_date": turn.get("timestamp"),
                    "full_context": True,
                },
            )
            for idx, turn in enumerate(history)
        ]
        self._org_state = OrganizationState(entries=list(self._memory_entries), raw={"mode": "full_history"})

    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        retrieved = [
            RetrievedMemory(
                entry_id=entry.entry_id,
                content=entry.content,
                score=1.0,
                rank=rank,
                source_ids=list(entry.source_ids),
                metadata=dict(entry.metadata) | {"created_at": entry.created_at, "updated_at": entry.updated_at},
            )
            for rank, entry in enumerate(self._memory_entries, start=1)
        ]
        self._retrieval_result = RetrievalResult(
            query=query,
            retrieved_entries=retrieved,
            top_k=len(retrieved),
            raw={"mode": "full_history"},
        )
        return self._retrieval_result

    def build_prompt(self, query: str, retrieved: RetrievalResult, sample: UnifiedSample) -> PromptRecord:
        memory_context = self._format_retrieved_sessions_compact(retrieved)
        full_prompt = (
            "You are an intelligent memory assistant answering questions based on historical conversation memories.\n\n"
            f"Question: {query}\n\n"
            f"{self._answering_instructions()}\n\n"
            "# Historical Conversation Memories\n"
            f"{memory_context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        self._prompt_record = PromptRecord(
            system_prompt=None,
            user_prompt=f"Question: {query}",
            memory_context=memory_context,
            full_prompt=full_prompt,
            injected_entry_ids=[entry.entry_id for entry in retrieved.retrieved_entries],
            token_count=self._estimate_token_count(full_prompt),
            injection_positions={entry.entry_id: "middle_context" for entry in retrieved.retrieved_entries},
            raw={"baseline": "full_context", "prompt_variant": "mnemis_style_with_timestamps_v3"},
        )
        return self._prompt_record

    def _format_retrieved_sessions_compact(self, retrieved: RetrievalResult) -> str:
        lines: list[str] = []
        active_session = object()
        for item in retrieved.retrieved_entries:
            raw = item.metadata or {}
            text = str(item.content or "").strip()
            if not text:
                continue
            session_id = str(raw.get("session_id") or "").strip()
            session_date = str(raw.get("session_date") or raw.get("created_at") or "").strip()
            session_key = (session_id, session_date)
            if session_key != active_session:
                header_parts = []
                if session_id:
                    header_parts.append(f"session_id={session_id}")
                if session_date:
                    header_parts.append(f"session_date={session_date}")
                lines.append(f"\n[Session {'; '.join(header_parts)}]".strip())
                active_session = session_key
            speaker = str(raw.get("speaker") or raw.get("role") or "unknown").strip()
            timestamp_str = self._format_timestamp(session_date)
            prefix = f"[{timestamp_str}] {speaker}:" if timestamp_str else f"{speaker}:"
            lines.append(f"{prefix} {self._strip_redundant_speaker_prefix(text, speaker=speaker)}")
        return "\n".join(lines).strip()

    def _format_timestamp(self, raw_ts: str) -> str:
        if not raw_ts:
            return ""
        import re
        from datetime import datetime, timezone
        ts = raw_ts.strip()
        # Already formatted like "2023/05/08 (Mon) 13:57"
        if re.match(r"\d{4}/\d{2}/\d{2}", ts):
            return ts
        # LoCoMo format: "1:56 pm on 8 May, 2023"
        try:
            dt = datetime.strptime(ts, "%I:%M %p on %d %B, %Y")
            return dt.strftime("%Y/%m/%d (%a) %H:%M")
        except (ValueError, TypeError):
            pass
        # ISO format
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.strftime("%Y/%m/%d (%a) %H:%M")
        except (ValueError, TypeError):
            pass
        # Unix timestamp (seconds or milliseconds)
        try:
            num = float(ts)
            if num > 1e12:
                num = num / 1000
            dt = datetime.fromtimestamp(num, tz=timezone.utc)
            return dt.strftime("%Y/%m/%d (%a) %H:%M")
        except (ValueError, TypeError, OSError):
            pass
        return ts
