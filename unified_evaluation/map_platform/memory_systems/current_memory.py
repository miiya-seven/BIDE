from __future__ import annotations

from typing import Any

from map_platform.datasets.schema import UnifiedSample
from map_platform.memory_systems.base import MemoryEntry, OrganizationState
from map_platform.memory_systems.simple_vector import SimpleVectorAdapter, _idf, _token_counts


class CurrentMemoryAdapter(SimpleVectorAdapter):
    system_name = "current_memory"

    def build_memory(self, history: list[dict[str, Any]], sample: UnifiedSample) -> None:
        entries: list[MemoryEntry] = []
        for index, turn in enumerate(history):
            source_id = str(turn.get("turn_id") or f"turn_{index + 1}")
            metadata = turn.get("metadata") if isinstance(turn.get("metadata"), dict) else {}
            current_texts = metadata.get("current_memory_texts") or []
            if current_texts:
                for memory_index, text in enumerate(current_texts, start=1):
                    content = str(text or "").strip()
                    if not content:
                        continue
                    entries.append(
                        MemoryEntry(
                            entry_id=f"{source_id}:memory:{memory_index}",
                            content=content,
                            raw={"source_turn": dict(turn), "source": "locomo_observation"},
                            created_at=turn.get("timestamp"),
                            updated_at=turn.get("timestamp"),
                            source_ids=[source_id],
                            metadata={
                                "session_id": turn.get("session_id"),
                                "session_date": turn.get("timestamp"),
                                "speaker": turn.get("speaker"),
                                "baseline": "current_memory",
                            },
                        )
                    )
            else:
                content = str(turn.get("text") or "").strip()
                if not content:
                    continue
                entries.append(
                    MemoryEntry(
                        entry_id=source_id,
                        content=f"{turn.get('speaker', 'unknown')}: {content}",
                        raw=dict(turn),
                        created_at=turn.get("timestamp"),
                        updated_at=turn.get("timestamp"),
                        source_ids=[source_id],
                        metadata={
                            "session_id": turn.get("session_id"),
                            "session_date": turn.get("timestamp"),
                            "speaker": turn.get("speaker"),
                            "baseline": "current_memory_fallback",
                        },
                    )
                )
        self._memory_entries = entries
        self._doc_tokens = [_token_counts(entry.content) for entry in self._memory_entries]
        self._idf = _idf(self._doc_tokens)
        self._org_state = OrganizationState(entries=list(self._memory_entries), raw={"mode": "current_memory_or_raw_fallback"})
