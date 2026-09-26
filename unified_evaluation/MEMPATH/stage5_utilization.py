from __future__ import annotations

from map_platform.datasets.schema import UnifiedSample
from map_platform.memory_systems.base import AnswerRecord, PromptRecord
from map_platform.metrics.judge import Judge


def evaluate_utilization(sample: UnifiedSample, prompt_record: PromptRecord, answer_record: AnswerRecord, judge: Judge) -> dict:
    acc = judge.answer_correct(sample.question, sample.answer, answer_record.answer, sample.success_criterion)
    return {
        "Acc": int(acc),
        "pred_answer": answer_record.answer,
    }


def evaluate_counterfactual(original_result: dict, counterfactual_result: dict, sample: UnifiedSample, judge: Judge) -> dict:
    cf = sample.counterfactual or {}
    expected_answer = str(cf.get("expected_answer") or cf.get("answer") or "")
    original_answer = str(original_result.get("pred_answer") or "")
    counterfactual_answer = str(counterfactual_result.get("pred_answer") or "")
    changed = original_answer.strip() != counterfactual_answer.strip()
    changed_correctly = judge.answer_changed_correctly(original_answer, counterfactual_answer, expected_answer) if expected_answer else changed
    return {
        "sample_id": sample.sample_id,
        "original_answer": original_answer,
        "counterfactual_answer": counterfactual_answer,
        "expected_answer": expected_answer,
        "answer_changed": changed,
        "changed_correctly": changed_correctly,
        "CSS": int(changed_correctly),
        "memory_changed": original_result.get("memory_entries") != counterfactual_result.get("memory_entries"),
        "retrieval_changed": original_result.get("retrieved_entries") != counterfactual_result.get("retrieved_entries"),
        "prompt_changed": original_result.get("prompt_record") != counterfactual_result.get("prompt_record"),
    }
