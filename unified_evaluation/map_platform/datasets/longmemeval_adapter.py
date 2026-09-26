from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from map_platform.datasets.base import BaseDatasetAdapter
from map_platform.datasets.schema import UnifiedSample


class LongMemEvalDatasetAdapter(BaseDatasetAdapter):
    dataset_name = "longmemeval"

    def parse_record(self, record: dict[str, Any]) -> UnifiedSample:
        is_raw_lme = "haystack_session_ids" in record and "answer_session_ids" in record
        history = (
            record.get("history")
            or record.get("haystack_history")
            or self._normalize_raw_sessions(record.get("haystack_sessions") or [], record)
        )
        gold_ids = self._gold_session_ids(record)
        gold_turn_ids = self._gold_turn_ids(record)
        history_by_session: dict[str, list[dict[str, Any]]] = {}
        for turn in history:
            history_by_session.setdefault(str(turn.get("session_id") or ""), []).append(turn)
        evidence_texts = [
            "\n".join(str(turn.get("text") or "") for turn in history_by_session[session_id]).strip()
            for session_id in gold_ids
            if session_id in history_by_session
        ]
        return UnifiedSample(
            sample_id=str(record.get("sample_id") or record.get("question_id") or record.get("id") or ""),
            dataset=self.dataset_name,
            task_type=self.normalize_task_type(str(record.get("task_type") or record.get("question_type") or "")),
            question=str(record.get("question") or record.get("query") or ""),
            answer=record.get("answer") or record.get("gold_answer") or "",
            history=history,
            question_type=str(record.get("question_type") or record.get("task_type") or ""),
            key_evidence_units=(
                evidence_texts if is_raw_lme else [str(item) for item in record.get("key_evidence_units", []) if str(item).strip()]
            ),
            key_evidence_quality=("gold" if gold_ids else record.get("key_evidence_quality")),
            key_evidence_support=("raw_longmemeval_answer_session_ids" if is_raw_lme else record.get("key_evidence_support")),
            memory_worthy_facts=[self.fact_from_record(item) for item in record.get("memory_worthy_facts", [])],
            required_fact_ids=[str(item) for item in record.get("required_fact_ids", [])],
            forbidden_fact_ids=[str(item) for item in record.get("forbidden_fact_ids", [])],
            supersede_relations=[self.relation_from_record(item) for item in record.get("supersede_relations", [])],
            success_criterion=dict(record.get("success_criterion", {})),
            counterfactual=record.get("counterfactual"),
            metadata={
                **dict(record.get("metadata", {})),
                "question_id": str(record.get("question_id") or record.get("sample_id") or record.get("id") or ""),
                "gold_memory_ids": gold_ids,
                "raw_evidence_ref": {
                    "answer_session_ids": [self._norm(item) for item in record.get("answer_session_ids") or [] if self._norm(item)],
                    "answer_turn_ids": gold_turn_ids,
                    "evidence_sessions": [
                        self._norm(item) for item in record.get("evidence_sessions") or [] if self._norm(item)
                    ],
                    "gold_sessions": [self._norm(item) for item in record.get("gold_sessions") or [] if self._norm(item)],
                    "related_sessions": [
                        self._norm(item) for item in record.get("related_sessions") or [] if self._norm(item)
                    ],
                },
                "haystack_session_ids": record.get("haystack_session_ids") or [],
                "gold_turn_ids": gold_turn_ids,
            },
        )

    def normalize_task_type(self, raw_task_type: str) -> str:
        mapping = {
            "t1_fact": "T1_fact",
            "t1_fact_qa": "T1_fact",
            "single_session_user": "T1_fact",
            "single_session_assistant": "T1_fact",
            "single-session-user": "T1_fact",
            "single-session-assistant": "T1_fact",
            "t2_update": "T2_update",
            "temporal_update": "T2_update",
            "knowledge-update": "T2_update",
            "t3_preference": "T3_preference",
            "preference": "T3_preference",
            "constraint": "T3_preference",
            "t4_agentic": "T4_agentic",
            "agentic": "T4_agentic",
            "task_resume": "T4_agentic",
        }
        key = raw_task_type.strip().lower()
        return mapping.get(key, raw_task_type or "T1_fact")

    def bootstrap_assets_if_available(self, missing_path: Path) -> Path | None:
        candidates = [
            Path("dataset") / "workspace" / "longmemeval_keyspan_smoke.json",
        ]
        return next((candidate for candidate in candidates if candidate.exists()), None)

    def _normalize_raw_sessions(self, raw_sessions: list[Any], record: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        record = record or {}
        session_ids = [self._norm(item) for item in record.get("haystack_session_ids") or []]
        session_dates = [self._norm(item) for item in record.get("haystack_dates") or []]
        history: list[dict[str, Any]] = []
        for session_index, session in enumerate(raw_sessions):
            session_id = self._resolve_session_id(session, session_ids, session_index)
            timestamp = session_dates[session_index] if session_index < len(session_dates) else None
            turns = self._extract_session_turns(session)
            for message_index, turn in enumerate(turns):
                if isinstance(turn, str):
                    speaker, text, has_answer = "unknown", self._norm(turn), False
                elif isinstance(turn, dict):
                    speaker = self._norm(turn.get("role") or turn.get("speaker") or turn.get("author")) or "unknown"
                    text = self._norm(turn.get("content") or turn.get("text") or turn.get("message") or turn.get("utterance"))
                    has_answer = bool(turn.get("has_answer"))
                else:
                    continue
                if not text:
                    continue
                history.append({
                    "turn_id": f"{session_id}:turn_{message_index}",
                    "official_turn_id": f"{session_id}_{message_index + 1}",
                    "session_id": session_id,
                    "speaker": speaker,
                    "text": text,
                    "timestamp": timestamp,
                    "turn_index": message_index,
                    "metadata": {
                        "session_level": False,
                        "session_index": session_index,
                        "message_index": message_index,
                        "has_answer": has_answer,
                    },
                })
        return history

    def _resolve_session_id(self, session: Any, session_ids: list[str], session_index: int) -> str:
        if isinstance(session, dict):
            for key in ("session_id", "id", "name"):
                if self._norm(session.get(key)):
                    return self._norm(session.get(key))
        if session_index < len(session_ids) and session_ids[session_index]:
            return session_ids[session_index]
        return f"session_{session_index}"

    def _format_session_text(self, session: Any) -> str:
        if isinstance(session, str):
            return self._norm(session)
        turns = self._extract_session_turns(session)
        lines: list[str] = []
        for turn in turns:
            if isinstance(turn, str):
                if self._norm(turn):
                    lines.append(self._norm(turn))
                continue
            if not isinstance(turn, dict):
                continue
            role = self._norm(turn.get("role") or turn.get("speaker") or turn.get("author"))
            content = self._norm(turn.get("content") or turn.get("text") or turn.get("message") or turn.get("utterance"))
            if content:
                lines.append(f"{role}: {content}" if role else content)
        return "\n".join(lines)

    def _extract_session_turns(self, session: Any) -> list[Any]:
        if isinstance(session, list):
            return session
        if not isinstance(session, dict):
            return []
        for key in ("session", "turns", "messages", "history", "dialogue"):
            value = session.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return list(value.values())
        return []

    def _session_has_answer(self, session: Any) -> bool:
        return any(isinstance(turn, dict) and bool(turn.get("has_answer")) for turn in self._extract_session_turns(session))

    def _gold_session_ids(self, record: dict[str, Any]) -> list[str]:
        for key in ("answer_session_ids", "evidence_sessions", "gold_sessions", "related_sessions"):
            value = record.get(key)
            if isinstance(value, list):
                ids = [self._norm(item) for item in value if self._norm(item)]
                if ids:
                    return ids
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        for key in ("answer_session_ids", "evidence_sessions", "gold_sessions", "related_sessions"):
            value = metadata.get(key)
            if isinstance(value, list):
                ids = [self._norm(item) for item in value if self._norm(item)]
                if ids:
                    return ids
        return []

    def _gold_turn_ids(self, record: dict[str, Any]) -> list[str]:
        """Return official retrieval turn IDs (user turns only)."""
        session_ids = [self._norm(item) for item in record.get("haystack_session_ids") or []]
        output: list[str] = []
        for session_id, session in zip(session_ids, record.get("haystack_sessions") or []):
            for message_index, turn in enumerate(self._extract_session_turns(session)):
                if not isinstance(turn, dict):
                    continue
                if self._norm(turn.get("role")).lower() != "user":
                    continue
                if bool(turn.get("has_answer")):
                    output.append(f"{session_id}_{message_index + 1}")
        return output

    def _norm(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()
