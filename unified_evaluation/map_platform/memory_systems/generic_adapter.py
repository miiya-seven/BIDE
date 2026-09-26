"""Protocol adapter for external memory systems.

The adapter accepts either a JSON HTTP service or a local JSON/JSONL artifact.
It normalizes both into the benchmark's MemoryEntry/RetrievalResult contract.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from .base import BaseMemorySystemAdapter, MemoryEntry, RetrievalResult, RetrievedMemory


class GenericExternalAdapter(BaseMemorySystemAdapter):
    system_name = "external"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.endpoint = os.getenv("EXTERNAL_MEMORY_URL", "").rstrip("/")
        self.artifact = os.getenv("EXTERNAL_MEMORY_ARTIFACT", "")

    def validate_setup(self) -> dict[str, Any]:
        ready = bool(self.endpoint or self.artifact)
        return {"system": self.system_name, "ready": ready,
                "message": "Set EXTERNAL_MEMORY_URL or EXTERNAL_MEMORY_ARTIFACT" if not ready else ""}

    def build_memory(self, history: list[dict[str, Any]], sample: Any) -> None:
        payload = {"sample_id": sample.sample_id, "history": history}
        if self.endpoint:
            data = self._post("/build", payload)
            entries = data.get("entries", data.get("memories", []))
        else:
            entries = self._read_artifact(sample.sample_id)
        self._memory_entries = [self._entry(item, i) for i, item in enumerate(entries)]

    def retrieve(self, query: str, sample: Any, top_k: int) -> RetrievalResult:
        if self.endpoint:
            data = self._post("/retrieve", {"sample_id": sample.sample_id, "query": query, "top_k": top_k})
            items = data.get("retrieved", data.get("results", []))
        else:
            items = self._memory_entries[:top_k]
        out = []
        for rank, item in enumerate(items[:top_k], 1):
            entry = self._entry(item, rank - 1)
            out.append(RetrievedMemory(entry.entry_id, entry.content, item.get("score"), rank,
                                       list(entry.source_ids), dict(entry.metadata)))
        self._retrieval_result = RetrievalResult(query, out, top_k, raw={"adapter": "generic_external"})
        return self._retrieval_result

    def _entry(self, item: Any, index: int) -> MemoryEntry:
        item = item if isinstance(item, dict) else {"content": str(item)}
        source_ids = item.get("source_ids") or item.get("source_ids_raw") or item.get("raw_ids") or []
        if isinstance(source_ids, str): source_ids = [source_ids]
        return MemoryEntry(str(item.get("entry_id") or item.get("id") or f"external_{index}"),
                           str(item.get("content") or item.get("text") or ""), dict(item),
                           item.get("created_at") or item.get("timestamp"), item.get("updated_at"),
                           [str(x) for x in source_ids], dict(item.get("metadata") or {}))

    def _read_artifact(self, sample_id: str) -> list[dict[str, Any]]:
        path = Path(self.artifact)
        if path.is_dir(): path = path / f"{sample_id}.jsonl"
        if not path.exists(): return []
        if path.suffix == ".jsonl": return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        payload = json.loads(path.read_text())
        return payload.get(sample_id, payload if isinstance(payload, list) else payload.get("entries", []))

    def _post(self, suffix: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.endpoint + suffix, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode())
