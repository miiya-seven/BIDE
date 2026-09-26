from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from map_platform.datasets.schema import Fact, SupersedeRelation, UnifiedSample


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


class BaseDatasetAdapter(ABC):
    dataset_name = "base"

    def load(self, data_path: str | Path, max_samples: int | None = None) -> list[UnifiedSample]:
        resolved = self.resolve_data_path(data_path)
        if resolved.is_dir():
            samples = self._load_from_asset_dir(resolved)
        else:
            records = self._read_records(resolved)
            samples = [self.parse_record(record) for record in records]
        return samples[:max_samples] if max_samples is not None else samples

    def resolve_data_path(self, data_path: str | Path) -> Path:
        path = Path(data_path).expanduser()
        if path.exists():
            return path.resolve()

        bootstrapped = self.bootstrap_assets_if_available(path)
        if bootstrapped is not None and bootstrapped.exists():
            return bootstrapped.resolve()

        guessed_dirs = [
            path.parent / self.dataset_name,
            path.parent,
            Path("data") / "processed" / self.dataset_name,
            Path("dataset") / "data" / "processed" / self.dataset_name,
        ]
        for candidate in guessed_dirs:
            if (candidate / "documents.jsonl").exists() and (candidate / "qa_samples.jsonl").exists():
                return candidate.resolve()
        raise FileNotFoundError(f"Could not find dataset asset at {path}")

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".jsonl":
            return _read_jsonl(path)
        payload = _read_json(path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("samples", "records", "data", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
        raise ValueError(f"Unsupported dataset file format: {path}")

    def _load_from_asset_dir(self, asset_dir: Path) -> list[UnifiedSample]:
        documents = _read_jsonl(asset_dir / "documents.jsonl")
        facts = _read_jsonl(asset_dir / "facts.jsonl") if (asset_dir / "facts.jsonl").exists() else []
        qa_samples = _read_jsonl(asset_dir / "qa_samples.jsonl")
        relations = _read_jsonl(asset_dir / "relations.jsonl") if (asset_dir / "relations.jsonl").exists() else []

        docs_by_id: dict[str, dict[str, Any]] = {}
        for document in documents:
            doc_id = str(document.get("doc_id") or document.get("parent_id") or "")
            if doc_id:
                docs_by_id[doc_id] = document

        facts_by_parent: dict[str, list[Fact]] = {}
        for row in facts:
            parent_id = str(row.get("parent_id") or row.get("doc_id") or "")
            facts_by_parent.setdefault(parent_id, []).append(self._fact_from_asset(row))

        relations_by_parent: dict[str, list[SupersedeRelation]] = {}
        for row in relations:
            parent_id = str(row.get("parent_id") or "")
            relations_by_parent.setdefault(parent_id, []).append(self._relation_from_asset(row))

        samples: list[UnifiedSample] = []
        for row in qa_samples:
            parent_id = str(row.get("parent_id") or row.get("doc_id") or "")
            document = docs_by_id.get(parent_id, {})
            sample = UnifiedSample(
                sample_id=str(row.get("sample_id") or ""),
                dataset=self.dataset_name,
                task_type=self.normalize_task_type(str(row.get("task_type") or row.get("question_type") or "")),
                question=str(row.get("query") or row.get("question") or ""),
                answer=row.get("gold_answer") or row.get("answer") or "",
                history=self.normalize_history(document.get("history") or []),
                question_type=str(row.get("question_type") or row.get("task_type") or ""),
                key_evidence_units=[str(item) for item in row.get("key_evidence_units", []) if str(item).strip()],
                key_evidence_quality=row.get("key_evidence_quality"),
                key_evidence_support=row.get("key_evidence_support"),
                memory_worthy_facts=facts_by_parent.get(parent_id, []),
                required_fact_ids=[str(item) for item in row.get("required_fact_ids", [])],
                forbidden_fact_ids=[str(item) for item in row.get("forbidden_fact_ids", [])],
                supersede_relations=relations_by_parent.get(parent_id, []),
                success_criterion={
                    "gold_answer": row.get("gold_answer") or row.get("answer") or "",
                    "acceptable_answers": row.get("acceptable_answers", []),
                },
                counterfactual=row.get("counterfactual"),
                metadata={
                    "parent_id": parent_id,
                    "raw_evidence_ref": row.get("raw_evidence_ref", {}),
                    "document_metadata": document.get("metadata", {}),
                    "sample_metadata": row.get("metadata", {}),
                },
            )
            samples.append(sample)
        return samples

    def _fact_from_asset(self, row: dict[str, Any]) -> Fact:
        return Fact(
            fact_id=str(row.get("fact_id") or ""),
            subject=str(row.get("subject") or "unknown"),
            attribute=str(row.get("attribute") or "unknown_attribute"),
            value=str(row.get("value") or ""),
            timestamp=row.get("timestamp"),
            source={
                "turn_ids": [str(item) for item in row.get("source_turn_ids", [])],
                "source_text": row.get("source_text"),
            },
            fact_type=str(row.get("fact_type") or "semantic"),
        )

    def _relation_from_asset(self, row: dict[str, Any]) -> SupersedeRelation:
        return SupersedeRelation(
            old_fact_id=str(row.get("old_fact_id") or ""),
            new_fact_id=str(row.get("new_fact_id") or ""),
            attribute=str(row.get("attribute") or "unknown_attribute"),
            reason=row.get("metadata", {}).get("reason") if isinstance(row.get("metadata"), dict) else None,
        )

    def normalize_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for idx, turn in enumerate(history):
            normalized.append(
                {
                    "turn_id": str(turn.get("turn_id") or turn.get("id") or f"turn_{idx + 1}"),
                    "session_id": str(turn.get("session_id") or ""),
                    "speaker": str(turn.get("speaker") or turn.get("role") or "unknown"),
                    "text": str(turn.get("text") or turn.get("content") or ""),
                    "timestamp": turn.get("timestamp"),
                    "turn_index": turn.get("turn_index", idx),
                    "metadata": dict(turn.get("metadata", {})) if isinstance(turn.get("metadata", {}), dict) else {},
                }
            )
        return normalized

    def fact_from_record(self, row: dict[str, Any]) -> Fact:
        return Fact(
            fact_id=str(row.get("fact_id") or ""),
            subject=str(row.get("subject") or "unknown"),
            attribute=str(row.get("attribute") or "unknown_attribute"),
            value=str(row.get("value") or ""),
            timestamp=row.get("timestamp"),
            source=dict(row.get("source", {})),
            fact_type=str(row.get("fact_type") or "semantic"),
        )

    def relation_from_record(self, row: dict[str, Any]) -> SupersedeRelation:
        return SupersedeRelation(
            old_fact_id=str(row.get("old_fact_id") or ""),
            new_fact_id=str(row.get("new_fact_id") or ""),
            attribute=str(row.get("attribute") or "unknown_attribute"),
            reason=row.get("reason"),
        )

    @abstractmethod
    def parse_record(self, record: dict[str, Any]) -> UnifiedSample:
        raise NotImplementedError

    @abstractmethod
    def normalize_task_type(self, raw_task_type: str) -> str:
        raise NotImplementedError

    def bootstrap_assets_if_available(self, missing_path: Path) -> Path | None:
        return None


def load_dataset(dataset: str, data_path: str | Path, max_samples: int | None = None) -> list[UnifiedSample]:
    normalized = dataset.strip().lower()
    if normalized == "locomo":
        from map_platform.datasets.locomo_adapter import LoCoMoDatasetAdapter

        adapter: BaseDatasetAdapter = LoCoMoDatasetAdapter()
    elif normalized == "longmemeval":
        from map_platform.datasets.longmemeval_adapter import LongMemEvalDatasetAdapter

        adapter = LongMemEvalDatasetAdapter()
    else:
        from map_platform.datasets.generic_adapter import GenericDatasetAdapter

        adapter = GenericDatasetAdapter(dataset_name=normalized)
    return adapter.load(data_path=data_path, max_samples=max_samples)
