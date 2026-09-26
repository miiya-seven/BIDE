from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from map_platform.datasets.base import BaseDatasetAdapter
from map_platform.datasets.schema import UnifiedSample


class LoCoMoDatasetAdapter(BaseDatasetAdapter):
    dataset_name = "locomo"

    def load(self, data_path: str | Path, max_samples: int | None = None) -> list[UnifiedSample]:
        resolved = self.resolve_data_path(data_path)
        if resolved.is_dir():
            return super().load(data_path=resolved, max_samples=max_samples)
        records = self._read_records(resolved)
        if records and isinstance(records[0], dict) and "conversation" in records[0] and "qa" in records[0]:
            samples = self._parse_raw_locomo(records)
            return samples[:max_samples] if max_samples is not None else samples
        return super().load(data_path=data_path, max_samples=max_samples)

    def parse_record(self, record: dict[str, Any]) -> UnifiedSample:
        return UnifiedSample(
            sample_id=str(record.get("sample_id") or record.get("id") or ""),
            dataset=self.dataset_name,
            task_type=self.normalize_task_type(str(record.get("task_type") or record.get("category") or "")),
            question=str(record.get("question") or record.get("query") or ""),
            answer=record.get("answer") or record.get("gold_answer") or "",
            history=self.normalize_history(record.get("history") or record.get("conversation_history") or []),
            question_type=str(record.get("question_type") or record.get("task_type") or record.get("category") or ""),
            key_evidence_units=[str(item) for item in record.get("key_evidence_units", []) if str(item).strip()],
            key_evidence_quality=record.get("key_evidence_quality"),
            key_evidence_support=record.get("key_evidence_support"),
            memory_worthy_facts=[self.fact_from_record(item) for item in record.get("memory_worthy_facts", [])],
            required_fact_ids=[str(item) for item in record.get("required_fact_ids", [])],
            forbidden_fact_ids=[str(item) for item in record.get("forbidden_fact_ids", [])],
            supersede_relations=[self.relation_from_record(item) for item in record.get("supersede_relations", [])],
            success_criterion=dict(record.get("success_criterion", {})),
            counterfactual=record.get("counterfactual"),
            metadata=dict(record.get("metadata", {})),
        )

    def normalize_task_type(self, raw_task_type: str) -> str:
        mapping = {
            "t1_fact": "T1_fact",
            "t1_fact_qa": "T1_fact",
            "t2_update": "T2_update",
            "t3_preference": "T3_preference",
            "t3_preference_constraint": "T3_preference",
            "t4_agentic": "T4_agentic",
            "t4_agentic_resume": "T4_agentic",
            "single_hop": "T1_fact",
            "temporal": "T2_update",
            "multi_hop": "T1_fact",
            "open_domain_qa": "T1_fact",
            "adversarial": "T2_update",
        }
        key = raw_task_type.strip().lower()
        return mapping.get(key, raw_task_type or "T1_fact")

    def bootstrap_assets_if_available(self, missing_path: Path) -> Path | None:
        output_dir = Path("data") / "processed" / self.dataset_name
        if (output_dir / "documents.jsonl").exists():
            return output_dir
        return None

    def _parse_raw_locomo(self, records: list[dict[str, Any]]) -> list[UnifiedSample]:
        samples: list[UnifiedSample] = []
        for record_index, record in enumerate(records):
            conversation = record.get("conversation") or {}
            if not isinstance(conversation, dict):
                continue
            current_memory_by_source = self._build_observation_index(record.get("observation") or {})
            history = self._raw_history(conversation, current_memory_by_source)
            history_by_id = {str(turn.get("turn_id") or ""): turn for turn in history}
            source_sample_id = str(record.get("sample_id") or record.get("id") or f"locomo_{record_index}")
            qa_items = record.get("qa") or []
            if not isinstance(qa_items, list):
                continue
            for qa_index, qa in enumerate(qa_items):
                if not isinstance(qa, dict):
                    continue
                gold_ids = [self._norm(item) for item in qa.get("evidence") or [] if self._norm(item)]
                evidence_texts = [
                    str(history_by_id[evidence_id].get("text") or "")
                    for evidence_id in gold_ids
                    if evidence_id in history_by_id and str(history_by_id[evidence_id].get("text") or "").strip()
                ]
                answer = self._norm(qa.get("answer"))
                samples.append(
                    UnifiedSample(
                        sample_id=f"{source_sample_id}__qa_{qa_index}",
                        dataset=self.dataset_name,
                        task_type=self.normalize_task_type(str(qa.get("category") or "")),
                        question=self._norm(qa.get("question")),
                        answer=answer,
                        history=history,
                        question_type=str(qa.get("category") or ""),
                        key_evidence_units=evidence_texts,
                        key_evidence_quality="gold" if gold_ids else None,
                        key_evidence_support="raw_locomo_qa_evidence",
                        success_criterion={"gold_answer": answer, "acceptable_answers": []},
                        metadata={
                            "source_sample_id": source_sample_id,
                            "qa_index": qa_index,
                            "gold_memory_ids": gold_ids,
                            "raw_evidence_ref": {"dialog_ids": gold_ids},
                            "raw_category": qa.get("category"),
                        },
                    )
                )
        return samples

    def _raw_history(
        self,
        conversation: dict[str, Any],
        current_memory_by_source: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for session_key in self._session_keys(conversation):
            turns = conversation.get(session_key)
            if not isinstance(turns, list):
                continue
            timestamp = self._norm(conversation.get(f"{session_key}_date_time"))
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue
                turn_id = self._norm(turn.get("dia_id")) or f"{session_key}:{turn_index + 1}"
                text = self._format_turn_text(turn)
                if not text:
                    continue
                history.append(
                    {
                        "turn_id": turn_id,
                        "session_id": session_key,
                        "speaker": self._norm(turn.get("speaker")) or "unknown",
                        "text": text,
                        "timestamp": timestamp,
                        "turn_index": turn_index,
                        "metadata": {
                            "raw_dia_id": turn_id,
                            "has_blip_caption": bool(self._norm(turn.get("blip_caption"))),
                            "current_memory_texts": current_memory_by_source.get(turn_id, []),
                        },
                    }
                )
        return history

    def _build_observation_index(self, observation: Any) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        if not isinstance(observation, dict):
            return index
        for session_observation in observation.values():
            if not isinstance(session_observation, dict):
                continue
            for facts in session_observation.values():
                if not isinstance(facts, list):
                    continue
                for fact_entry in facts:
                    if not isinstance(fact_entry, list) or not fact_entry:
                        continue
                    fact = self._norm(fact_entry[0])
                    if not fact:
                        continue
                    raw_ids = fact_entry[1] if len(fact_entry) > 1 else []
                    source_ids = raw_ids if isinstance(raw_ids, list) else [raw_ids]
                    for source_id in source_ids:
                        normalized_id = self._norm(source_id)
                        if normalized_id:
                            index.setdefault(normalized_id, []).append(fact)
        return index

    def _format_turn_text(self, turn: dict[str, Any]) -> str:
        text = self._norm(turn.get("text"))
        caption = self._norm(turn.get("blip_caption"))
        if caption:
            text = f"[Image: {caption}] {text}".strip()
        return text

    def _session_keys(self, conversation: dict[str, Any]) -> list[str]:
        def session_num(key: str) -> int:
            match = re.fullmatch(r"session_(\d+)", key)
            return int(match.group(1)) if match else 10**9

        return sorted(
            [key for key, value in conversation.items() if re.fullmatch(r"session_\d+", key) and isinstance(value, list)],
            key=session_num,
        )

    def _norm(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()
