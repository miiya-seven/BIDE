from __future__ import annotations

from map_platform.datasets.schema import UnifiedSample
from map_platform.memory_systems.base import RetrievalResult
from map_platform.metrics.judge import Judge


def evaluate_retrieval(
    sample: UnifiedSample,
    retrieval_result: RetrievalResult,
    judge: Judge,
    stage2_result: dict[str, object] | None = None,
    k: int = 5,
) -> dict:
    keu = sample.derived_key_evidence_units()
    top_entries = retrieval_result.retrieved_entries[:k]
    required_keu_ranks: dict[str, int] = {}
    for unit in keu:
        for entry in top_entries:
            if judge.evidence_in_text(unit, entry.content):
                required_keu_ranks[unit] = entry.rank
                break

    first_required_rank = min(required_keu_ranks.values()) if required_keu_ranks else None
    contradiction_entries = [
        entry for entry in top_entries if judge.text_contradicts_answer(sample.question, sample.canonical_answer(), entry.content)
    ]
    rfr_system = (len(required_keu_ranks) / len(keu)) if keu else None
    rfr_std_sigma = stage2_result.get("RFR_std_sigma") if isinstance(stage2_result, dict) else None
    return {
        "RAG": (rfr_system - rfr_std_sigma) if rfr_system is not None and isinstance(rfr_std_sigma, (int, float)) else None,
        f"RFR@{k}": rfr_system,
        "MRR": (1.0 / first_required_rank) if first_required_rank else 0.0,
        "required_keu_ranks": required_keu_ranks,
        "CMR_R": int(bool(contradiction_entries)),
        "contradiction_entry_ids": [entry.entry_id for entry in contradiction_entries],
        "RFR_std_sigma": rfr_std_sigma,
        "probe_config": dict(stage2_result.get("probe_config", {})) if isinstance(stage2_result, dict) else {},
    }
