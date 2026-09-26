from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from map_platform.datasets.schema import UnifiedSample
from map_platform.memory_systems.base import (
    BaseMemorySystemAdapter,
    MemoryEntry,
    OrganizationState,
    RetrievalResult,
    RetrievedMemory,
)


class SimpleVectorAdapter(BaseMemorySystemAdapter):
    system_name = "simple_vector"

    def build_memory(self, history: list[dict[str, Any]], sample: UnifiedSample) -> None:
        self._memory_entries = []
        for index, turn in enumerate(history):
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            source_id = str(turn.get("turn_id") or f"turn_{index + 1}")
            content = f"{turn.get('speaker', 'unknown')}: {text}"
            self._memory_entries.append(
                MemoryEntry(
                    entry_id=source_id,
                    content=content,
                    raw=dict(turn),
                    created_at=turn.get("timestamp"),
                    updated_at=turn.get("timestamp"),
                    source_ids=[source_id],
                    metadata={
                        "session_id": turn.get("session_id"),
                        "session_date": turn.get("timestamp"),
                        "speaker": turn.get("speaker"),
                        "baseline": "simple_vector",
                    },
                )
            )
        self._doc_tokens = [_token_counts(entry.content) for entry in self._memory_entries]
        self._idf = _idf(self._doc_tokens)
        self._org_state = OrganizationState(entries=list(self._memory_entries), raw={"mode": "flat_tfidf_vector"})

    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        query_counts = _token_counts(query)
        scored: list[tuple[float, MemoryEntry]] = []
        for entry, doc_counts in zip(self._memory_entries, self._doc_tokens):
            scored.append((_tfidf_cosine(query_counts, doc_counts, self._idf), entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        retrieved = [
            RetrievedMemory(
                entry_id=entry.entry_id,
                content=entry.content,
                score=score,
                rank=rank,
                source_ids=list(entry.source_ids),
                metadata=dict(entry.metadata) | {"created_at": entry.created_at, "updated_at": entry.updated_at},
            )
            for rank, (score, entry) in enumerate(scored[: max(top_k, 0)], start=1)
        ]
        self._retrieval_result = RetrievalResult(query=query, retrieved_entries=retrieved, top_k=top_k, raw={"mode": "tfidf_cosine"})
        return self._retrieval_result


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9']+", str(text or "").lower()) if len(token) > 1]


def _token_counts(text: str) -> Counter[str]:
    return Counter(_tokens(text))


def _idf(doc_counts: list[Counter[str]]) -> dict[str, float]:
    doc_freq: Counter[str] = Counter()
    for counts in doc_counts:
        doc_freq.update(counts.keys())
    total = max(len(doc_counts), 1)
    return {token: math.log((1 + total) / (1 + freq)) + 1 for token, freq in doc_freq.items()}


def _tfidf_cosine(query: Counter[str], doc: Counter[str], idf: dict[str, float]) -> float:
    if not query or not doc:
        return 0.0
    tokens = set(query) | set(doc)
    dot = sum(query[token] * doc[token] * idf.get(token, 1.0) ** 2 for token in tokens)
    query_norm = math.sqrt(sum((query[token] * idf.get(token, 1.0)) ** 2 for token in query))
    doc_norm = math.sqrt(sum((doc[token] * idf.get(token, 1.0)) ** 2 for token in doc))
    if not query_norm or not doc_norm:
        return 0.0
    return dot / (query_norm * doc_norm)
