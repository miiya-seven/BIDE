from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from map_platform.datasets.schema import UnifiedSample
from map_platform.memory_systems.base import MemoryEntry, OrganizationState
from map_platform.metrics.judge import Judge

# 固定标准探针配置：主分只由 BM25 与 embedding/cosine 构成。
# 组织结构不改主分，只在近似平分时作为稳定的打破平局信号。
STANDARD_PROBE_CONFIG = {
    "name": "fixed_bm25_embedding_probe",
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "lexical_weight": 0.5,
    "semantic_weight": 0.5,
    "score_bucket_precision": 4,
    "structure_role": "tiebreak_only",
}


@dataclass
class ProbeScoredEntry:
    entry: MemoryEntry
    lexical_score: float
    semantic_score: float
    combined_score: float
    structure_priority: float
    score_bucket: float

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry.entry_id,
            "content": self.entry.content,
            "source_ids": list(self.entry.source_ids),
            "lexical_score": self.lexical_score,
            "semantic_score": self.semantic_score,
            "combined_score": self.combined_score,
            "structure_priority": self.structure_priority,
            "score_bucket": self.score_bucket,
        }


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _bm25_scores(query: str, entries: list[MemoryEntry]) -> dict[str, float]:
    if not entries:
        return {}
    query_tokens = _tokenize(query)
    if not query_tokens:
        return {entry.entry_id: 0.0 for entry in entries}
    doc_tokens = {entry.entry_id: _tokenize(entry.content) for entry in entries}
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        for token in set(tokens):
            doc_freq[token] += 1
    avg_doc_len = sum(len(tokens) for tokens in doc_tokens.values()) / len(doc_tokens)
    k1 = float(STANDARD_PROBE_CONFIG["bm25_k1"])
    b = float(STANDARD_PROBE_CONFIG["bm25_b"])
    scores: dict[str, float] = {}
    for entry in entries:
        tokens = doc_tokens[entry.entry_id]
        token_counter = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            df = doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (len(entries) - df + 0.5) / (df + 0.5))
            tf = token_counter.get(token, 0)
            denom = tf + k1 * (1 - b + b * (len(tokens) / max(avg_doc_len, 1e-6)))
            score += idf * ((tf * (k1 + 1)) / denom) if denom else 0.0
        scores[entry.entry_id] = score
    return scores


def _normalize_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    if not raw_scores:
        return {}
    values = list(raw_scores.values())
    min_value = min(values)
    max_value = max(values)
    if max_value - min_value <= 1e-8:
        return {key: 0.0 for key in raw_scores}
    return {key: (value - min_value) / (max_value - min_value) for key, value in raw_scores.items()}


def _level_priority(org_state: OrganizationState) -> dict[str, float]:
    if not org_state.levels:
        return {}
    rank_by_entry: dict[str, int] = {}
    for rank, (_, entry_ids) in enumerate(org_state.levels.items()):
        if not isinstance(entry_ids, list):
            continue
        for entry_id in entry_ids:
            rank_by_entry[str(entry_id)] = rank
    if not rank_by_entry:
        return {}
    max_rank = max(rank_by_entry.values()) or 1
    return {entry_id: 1.0 - (rank / max_rank if max_rank else 0.0) for entry_id, rank in rank_by_entry.items()}


def _visibility_scores(org_state: OrganizationState) -> dict[str, float]:
    raw: dict[str, float] = {}
    for mapping in (org_state.weights or {}, org_state.priorities or {}):
        for key, value in mapping.items():
            try:
                raw[str(key)] = max(raw.get(str(key), 0.0), float(value))
            except Exception:
                continue
    normalized = _normalize_scores(raw)
    for entry_id, value in _level_priority(org_state).items():
        normalized[entry_id] = max(normalized.get(entry_id, 0.0), value)
    return normalized


def _standard_probe_ranked_entries(
    query: str,
    org_state: OrganizationState,
    judge: Judge,
    *,
    use_structure: bool,
) -> list[ProbeScoredEntry]:
    entries = list(org_state.entries)
    lexical = _normalize_scores(_bm25_scores(query, entries))
    semantic = _normalize_scores({entry.entry_id: judge.text_similarity(query, entry.content) for entry in entries})
    structure_priority = _visibility_scores(org_state) if use_structure else {}
    lexical_weight = float(STANDARD_PROBE_CONFIG["lexical_weight"])
    semantic_weight = float(STANDARD_PROBE_CONFIG["semantic_weight"])
    bucket_precision = int(STANDARD_PROBE_CONFIG["score_bucket_precision"])

    scored: list[ProbeScoredEntry] = []
    for entry in entries:
        lexical_score = lexical.get(entry.entry_id, 0.0)
        semantic_score = semantic.get(entry.entry_id, 0.0)
        combined_score = lexical_weight * lexical_score + semantic_weight * semantic_score
        priority = structure_priority.get(entry.entry_id, 0.0) if use_structure else 0.0
        scored.append(
            ProbeScoredEntry(
                entry=entry,
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                combined_score=combined_score,
                structure_priority=priority,
                score_bucket=round(combined_score, bucket_precision),
            )
        )

    if use_structure:
        scored.sort(
            key=lambda item: (
                item.score_bucket,
                item.structure_priority,
                item.combined_score,
                item.entry.entry_id,
            ),
            reverse=True,
        )
    else:
        scored.sort(
            key=lambda item: (
                item.score_bucket,
                item.combined_score,
                item.entry.entry_id,
            ),
            reverse=True,
        )
    return scored


def _keu_recall(
    keu: list[str],
    ranked_entries: list[ProbeScoredEntry],
    judge: Judge,
    top_k: int,
) -> tuple[float | None, dict[str, int]]:
    if not keu:
        return None, {}
    top_entries = ranked_entries[:top_k]
    ranks: dict[str, int] = {}
    for unit in keu:
        for index, scored in enumerate(top_entries, start=1):
            if judge.evidence_in_text(unit, scored.entry.content):
                ranks[unit] = index
                break
    return len(ranks) / len(keu), ranks


def evaluate_organization(sample: UnifiedSample, org_state: OrganizationState, judge: Judge, top_k: int = 5) -> dict:
    if not org_state.entries:
        return {
            "OQG": None,
            "CP_ratio": None,
            "stage_applicability": "not_applicable",
            "probe_config": dict(STANDARD_PROBE_CONFIG),
        }

    keu = sample.derived_key_evidence_units()
    ranked_sigma = _standard_probe_ranked_entries(sample.question, org_state, judge, use_structure=True)
    ranked_flat = _standard_probe_ranked_entries(sample.question, org_state, judge, use_structure=False)
    rfr_sigma, ranks_sigma = _keu_recall(keu, ranked_sigma, judge, top_k)
    rfr_flat, ranks_flat = _keu_recall(keu, ranked_flat, judge, top_k)
    visibility = _visibility_scores(org_state)

    contradiction_entries = judge.find_contradicting_entries(sample.question, sample.canonical_answer(), org_state.entries)
    correct_entries = [entry for entry in org_state.entries if any(judge.evidence_in_text(unit, entry.content) for unit in keu)]
    cp_ratio = None
    if contradiction_entries and correct_entries:
        best_correct = max((visibility.get(entry.entry_id, 0.0) for entry in correct_entries), default=0.0)
        best_contradiction = max((visibility.get(entry.entry_id, 0.0) for entry in contradiction_entries), default=0.0)
        cp_ratio = int(best_correct > best_contradiction)

    return {
        "OQG": (rfr_sigma - rfr_flat) if rfr_sigma is not None and rfr_flat is not None else None,
        "RFR_std_sigma": rfr_sigma,
        "RFR_std_flat": rfr_flat,
        "required_keu_ranks_std_sigma": ranks_sigma,
        "required_keu_ranks_std_flat": ranks_flat,
        "CP_ratio": cp_ratio,
        "contradiction_entry_ids": [entry.entry_id for entry in contradiction_entries],
        "probe_config": dict(STANDARD_PROBE_CONFIG),
        "std_probe_top_entries_sigma": [item.to_dict() for item in ranked_sigma[:top_k]],
        "std_probe_top_entries_flat": [item.to_dict() for item in ranked_flat[:top_k]],
        "stage_applicability": "non_trivial" if (org_state.levels or org_state.weights or org_state.priorities) else "flat_index",
    }
