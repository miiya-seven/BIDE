"""Adapter for datasets already converted to the suite's normalized schema.

This is deliberately small: it does not guess a benchmark's semantics.  A
new dataset must provide one JSON/JSONL record per question with ``question``,
``answer`` and ``history`` (or the equivalent aliases documented in the
benchmark-suite README).  Dataset-specific official judges/retrieval metrics
remain outside this generic adapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from map_platform.datasets.base import BaseDatasetAdapter
from map_platform.datasets.schema import UnifiedSample


class GenericDatasetAdapter(BaseDatasetAdapter):
    """Load normalized records without imposing LoCoMo/LME semantics."""

    def __init__(self, dataset_name: str = "generic") -> None:
        self.dataset_name = str(dataset_name or "generic").strip().lower()

    def parse_record(self, record: dict[str, Any]) -> UnifiedSample:
        history = record.get("history") or record.get("conversation_history") or record.get("conversation") or []
        # A session-oriented normalized record is also convenient for new
        # datasets.  Preserve session ids and timestamps while flattening it.
        if isinstance(history, dict):
            flattened: list[dict[str, Any]] = []
            for session_id, turns in history.items():
                if not isinstance(turns, list):
                    continue
                for index, turn in enumerate(turns):
                    if isinstance(turn, dict):
                        flattened.append({"session_id": session_id, "turn_index": index, **turn})
                    elif isinstance(turn, str):
                        flattened.append({"session_id": session_id, "turn_index": index, "text": turn})
            history = flattened
        elif isinstance(history, list):
            # Accept a compact list of strings as well as the canonical turn
            # dictionaries.  The base normalizer intentionally expects
            # dictionaries so that speaker/session metadata is not guessed.
            history = [
                item if isinstance(item, dict) else {"text": str(item)}
                for item in history
            ]

        sample_id = str(
            record.get("sample_id")
            or record.get("question_id")
            or record.get("id")
            or ""
        )
        question = str(record.get("question") or record.get("query") or "")
        answer = record.get("answer")
        if answer is None:
            answer = record.get("gold_answer") or record.get("reference_answer") or ""
        evidence = record.get("key_evidence_units") or record.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        # Optional normalized fields are allowed to be null.  Treat null as an
        # empty collection instead of making a custom dataset fail halfway
        # through a run.
        memory_worthy_facts = record.get("memory_worthy_facts") or []
        required_fact_ids = record.get("required_fact_ids") or []
        forbidden_fact_ids = record.get("forbidden_fact_ids") or []
        supersede_relations = record.get("supersede_relations") or []
        success_criterion = record.get("success_criterion")
        if not isinstance(success_criterion, dict):
            success_criterion = {"gold_answer": answer}
        raw_metadata = record.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata.setdefault("question_id", str(record.get("question_id") or sample_id))
        # Preserve grouping keys used by the shared runner when a normalized
        # dataset has multiple questions over one conversation/session.
        for key in ("conversation_id", "dialog_id", "session_group_id", "source_sample_id", "question_date"):
            if key in record and key not in metadata:
                metadata[key] = record[key]
        # Keep common retrieval references available to MAP-Eval without
        # pretending that a generic dataset has LongMemEval's official IDs.
        for key in ("answer_session_ids", "answer_turn_ids", "gold_memory_ids", "haystack_session_ids"):
            if key in record and key not in metadata:
                metadata[key] = record[key]
        return UnifiedSample(
            sample_id=sample_id,
            dataset=self.dataset_name,
            task_type=self.normalize_task_type(str(record.get("task_type") or record.get("question_type") or "generic")),
            question=question,
            answer=answer,
            history=self.normalize_history(history if isinstance(history, list) else []),
            question_type=str(record.get("question_type") or record.get("task_type") or "generic"),
            key_evidence_units=[str(item) for item in evidence if str(item).strip()],
            key_evidence_quality=record.get("key_evidence_quality"),
            key_evidence_support=record.get("key_evidence_support"),
            memory_worthy_facts=[self.fact_from_record(item) for item in memory_worthy_facts if isinstance(item, dict)],
            required_fact_ids=[str(item) for item in required_fact_ids],
            forbidden_fact_ids=[str(item) for item in forbidden_fact_ids],
            supersede_relations=[self.relation_from_record(item) for item in supersede_relations if isinstance(item, dict)],
            success_criterion=success_criterion,
            counterfactual=record.get("counterfactual"),
            metadata=metadata,
        )

    def normalize_task_type(self, raw_task_type: str) -> str:
        return str(raw_task_type or "generic").strip() or "generic"

    def bootstrap_assets_if_available(self, missing_path: Path) -> Path | None:
        # Generic datasets must be explicit; silently selecting another file is
        # dangerous for benchmark runs.
        return None
