from __future__ import annotations

from map_platform.datasets.schema import UnifiedSample
from map_platform.memory_systems.base import BaseMemorySystemAdapter, RetrievalResult


class NoMemoryAdapter(BaseMemorySystemAdapter):
    system_name = "no_memory"

    def build_memory(self, history: list[dict], sample: UnifiedSample) -> None:
        self._memory_entries = []
        self._org_state.entries = []
        self._org_state.raw = {"mode": "question_only"}

    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        self._retrieval_result = RetrievalResult(query=query, retrieved_entries=[], top_k=top_k, raw={"mode": "none"})
        return self._retrieval_result

    def build_prompt(self, query: str, retrieved: RetrievalResult, sample: UnifiedSample):
        self._prompt_record = super().build_prompt(query=query, retrieved=retrieved, sample=sample)
        self._prompt_record.raw["baseline"] = "question_only"
        return self._prompt_record
