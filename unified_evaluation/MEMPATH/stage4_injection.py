from __future__ import annotations

from map_platform.datasets.schema import UnifiedSample
from map_platform.memory_systems.base import PromptRecord
from map_platform.metrics.judge import Judge


def _find_position(prompt_text: str, evidence_unit: str) -> int | None:
    lowered_prompt = prompt_text.lower()
    lowered_unit = evidence_unit.lower().strip()
    if not lowered_unit:
        return None
    index = lowered_prompt.find(lowered_unit)
    return index if index >= 0 else None


def _position_bucket(prompt_record: PromptRecord, evidence_unit: str, judge: Judge) -> str:
    if prompt_record.system_prompt and judge.evidence_in_text(evidence_unit, prompt_record.system_prompt):
        return "system_prompt"
    if prompt_record.user_prompt and judge.evidence_in_text(evidence_unit, prompt_record.user_prompt):
        return "front_user_prompt"
    if prompt_record.memory_context and judge.evidence_in_text(evidence_unit, prompt_record.memory_context):
        return "middle_context"
    if judge.evidence_in_text(evidence_unit, prompt_record.full_prompt):
        return "end_context"
    return "not_injected"


def evaluate_injection(
    sample: UnifiedSample,
    prompt_record: PromptRecord,
    judge: Judge,
    stage3_result: dict[str, object] | None = None,
) -> dict:
    keu = sample.derived_key_evidence_units()
    required_positions: dict[str, str] = {}
    position_scores: list[float] = []
    full_prompt = prompt_record.full_prompt or ""

    for unit in keu:
        required_positions[unit] = _position_bucket(prompt_record, unit, judge)
        position = _find_position(full_prompt, unit)
        if position is not None and len(full_prompt) > 0:
            position_scores.append(1.0 - (position / len(full_prompt)))

    prompt_covered = sum(1 for value in required_positions.values() if value != "not_injected")
    pcr = (prompt_covered / len(required_positions)) if required_positions else None
    rfr_key = next((key for key in (stage3_result or {}) if key.startswith("RFR@")), None)
    rfr_value = (stage3_result or {}).get(rfr_key) if rfr_key else None
    contradiction_in_prompt = judge.text_contradicts_answer(sample.question, sample.canonical_answer(), full_prompt)
    return {
        "PCR": pcr,
        "TL": (rfr_value - pcr) if isinstance(rfr_value, (int, float)) and isinstance(pcr, (int, float)) else None,
        "PS": (sum(position_scores) / len(position_scores)) if position_scores else None,
        "CMR_P": int(bool(contradiction_in_prompt)),
        "required_keu_positions": required_positions,
        "prompt_token_count": prompt_record.token_count,
    }
