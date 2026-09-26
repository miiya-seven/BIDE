from __future__ import annotations

from itertools import combinations

from map_platform.datasets.schema import UnifiedSample
from map_platform.memory_systems.base import MemoryEntry
from map_platform.metrics.judge import Judge


def evaluate_encoding(sample: UnifiedSample, memory_entries: list[MemoryEntry], judge: Judge) -> dict:
    keu = sample.derived_key_evidence_units()
    answer = sample.canonical_answer()
    covered_units: list[str] = []
    missing_units: list[str] = []
    matched_entry_ids: set[str] = set()

    for unit in keu:
        matched = [entry for entry in memory_entries if judge.evidence_in_text(unit, entry.content)]
        if matched:
            covered_units.append(unit)
            matched_entry_ids.update(entry.entry_id for entry in matched)
        else:
            missing_units.append(unit)

    duplicate_pairs = 0
    for left, right in combinations(memory_entries, 2):
        if judge.text_similarity(left.content, right.content) > 0.85:
            duplicate_pairs += 1
    pair_total = len(memory_entries) * (len(memory_entries) - 1) / 2

    contradictions = judge.find_contradicting_entries(sample.question, answer, memory_entries) if answer else []
    return {
        "Q-FCR": (len(covered_units) / len(keu)) if keu else None,
        "RDR": (duplicate_pairs / pair_total) if pair_total else 0.0,
        "CMR": (len(matched_entry_ids) / len(memory_entries)) if memory_entries else None,
        "CMR_M": int(bool(contradictions)),
        "covered_keu": covered_units,
        "missing_keu": missing_units,
        "covered_memory_entry_ids": sorted(matched_entry_ids),
        "contradiction_entry_ids": [entry.entry_id for entry in contradictions],
    }
