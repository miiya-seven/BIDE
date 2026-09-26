from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from map_platform.utils.serialization import to_jsonable


@dataclass
class Fact:
    fact_id: str
    subject: str
    attribute: str
    value: str
    timestamp: str | None
    source: dict[str, Any]
    fact_type: str

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class SupersedeRelation:
    old_fact_id: str
    new_fact_id: str
    attribute: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class UnifiedSample:
    sample_id: str
    dataset: str
    task_type: str
    question: str
    answer: str | list[str]
    history: list[dict[str, Any]]
    question_type: str | None = None
    key_evidence_units: list[str] = field(default_factory=list)
    key_evidence_quality: str | None = None
    key_evidence_support: str | None = None
    memory_worthy_facts: list[Fact] = field(default_factory=list)
    required_fact_ids: list[str] = field(default_factory=list)
    forbidden_fact_ids: list[str] = field(default_factory=list)
    supersede_relations: list[SupersedeRelation] = field(default_factory=list)
    success_criterion: dict[str, Any] = field(default_factory=dict)
    counterfactual: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    def facts_by_id(self) -> dict[str, Fact]:
        return {fact.fact_id: fact for fact in self.memory_worthy_facts}

    def canonical_answer(self) -> str:
        if isinstance(self.answer, list):
            return str(self.answer[0]) if self.answer else ""
        return str(self.answer or "")

    def answer_candidates(self) -> list[str]:
        values: list[str] = []
        if isinstance(self.answer, list):
            values.extend(str(item) for item in self.answer if str(item).strip())
        elif str(self.answer or "").strip():
            values.append(str(self.answer))
        values.extend(str(item) for item in self.success_criterion.get("acceptable_answers", []) if str(item).strip())
        seen: set[str] = set()
        output: list[str] = []
        for item in values:
            if item not in seen:
                seen.add(item)
                output.append(item)
        return output

    def derived_key_evidence_units(self) -> list[str]:
        explicit = [str(item).strip() for item in self.key_evidence_units if str(item).strip()]
        if explicit:
            return explicit
        facts_by_id = self.facts_by_id()
        if self.required_fact_ids:
            derived = [
                facts_by_id[fact_id].value
                for fact_id in self.required_fact_ids
                if fact_id in facts_by_id and str(facts_by_id[fact_id].value).strip()
            ]
            if derived:
                return derived
        return [str(fact.value).strip() for fact in self.memory_worthy_facts if str(fact.value).strip()]

    def evaluation_split(self) -> str:
        quality = (self.key_evidence_quality or "").strip().lower()
        if quality in {"gold", "silver", "bronze", "exclude"}:
            return quality
        return "unlabeled"

    def question_group(self) -> str:
        return str(self.question_type or self.metadata.get("question_type") or self.task_type or "")

    def build_counterfactual_variant(self) -> "UnifiedSample":
        if self.counterfactual is None:
            raise ValueError("当前样本不包含 counterfactual。")
        payload = dict(self.counterfactual)
        return UnifiedSample(
            sample_id=f"{self.sample_id}__counterfactual",
            dataset=self.dataset,
            task_type=self.task_type,
            question=str(payload.get("question") or self.question),
            answer=payload.get("answer") or payload.get("expected_answer") or self.answer,
            history=list(payload.get("history") or self.history),
            question_type=str(payload.get("question_type") or self.question_group() or self.task_type),
            key_evidence_units=[str(item) for item in payload.get("key_evidence_units", self.key_evidence_units)],
            key_evidence_quality=payload.get("key_evidence_quality") or self.key_evidence_quality,
            key_evidence_support=payload.get("key_evidence_support") or self.key_evidence_support,
            memory_worthy_facts=list(self.memory_worthy_facts),
            required_fact_ids=list(self.required_fact_ids),
            forbidden_fact_ids=list(self.forbidden_fact_ids),
            supersede_relations=list(self.supersede_relations),
            success_criterion=dict(self.success_criterion),
            counterfactual=None,
            metadata={**self.metadata, "counterfactual_of": self.sample_id},
        )
