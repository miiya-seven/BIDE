from __future__ import annotations

from collections import defaultdict
from typing import Any


def _mean(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _rfr_value(row: dict[str, Any]) -> float | None:
    return next((value for key, value in row.get("stage3", {}).items() if key.startswith("RFR@")), None)


def _question_group(row: dict[str, Any]) -> str:
    return str(row.get("question_type") or row.get("task_type") or "")


def build_per_system_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset", ""), row.get("system", ""))].append(row)

    output: list[dict[str, Any]] = []
    for (dataset, system), items in grouped.items():
        output.append(
            {
                "dataset": dataset,
                "system": system,
                "num_samples": len(items),
                "Acc_E2E": _mean([item.get("stage5", {}).get("Acc") for item in items]),
                "Q-FCR": _mean([item.get("stage1", {}).get("Q-FCR") for item in items]),
                "RDR": _mean([item.get("stage1", {}).get("RDR") for item in items]),
                "CMR": _mean([item.get("stage1", {}).get("CMR") for item in items]),
                "CMR_M": _mean([item.get("stage1", {}).get("CMR_M") for item in items]),
                "OQG": _mean([item.get("stage2", {}).get("OQG") for item in items]),
                "RFR_std_sigma": _mean([item.get("stage2", {}).get("RFR_std_sigma") for item in items]),
                "RFR_std_flat": _mean([item.get("stage2", {}).get("RFR_std_flat") for item in items]),
                "CP_ratio": _mean([item.get("stage2", {}).get("CP_ratio") for item in items]),
                "RAG": _mean([item.get("stage3", {}).get("RAG") for item in items]),
                "RFR@k": _mean([_rfr_value(item) for item in items]),
                "CMR_R": _mean([item.get("stage3", {}).get("CMR_R") for item in items]),
                "TL": _mean([item.get("stage4", {}).get("TL") for item in items]),
                "PS": _mean([item.get("stage4", {}).get("PS") for item in items]),
                "CMR_P": _mean([item.get("stage4", {}).get("CMR_P") for item in items]),
            }
        )
    return output


def build_per_task_type_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset", ""), row.get("system", ""), row.get("task_type", ""), _question_group(row))].append(row)

    output: list[dict[str, Any]] = []
    for (dataset, system, task_type, question_type), items in grouped.items():
        output.append(
            {
                "dataset": dataset,
                "system": system,
                "task_type": task_type,
                "question_type": question_type,
                "num_samples": len(items),
                "Acc_E2E": _mean([item.get("stage5", {}).get("Acc") for item in items]),
                "Q-FCR": _mean([item.get("stage1", {}).get("Q-FCR") for item in items]),
                "OQG": _mean([item.get("stage2", {}).get("OQG") for item in items]),
                "RFR_std_sigma": _mean([item.get("stage2", {}).get("RFR_std_sigma") for item in items]),
                "RFR_std_flat": _mean([item.get("stage2", {}).get("RFR_std_flat") for item in items]),
                "RAG": _mean([item.get("stage3", {}).get("RAG") for item in items]),
                "TL": _mean([item.get("stage4", {}).get("TL") for item in items]),
                "CMR_M": _mean([item.get("stage1", {}).get("CMR_M") for item in items]),
                "CMR_P": _mean([item.get("stage4", {}).get("CMR_P") for item in items]),
            }
        )
    return output


def build_contradiction_flow_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset", ""), row.get("system", ""), row.get("task_type", ""), _question_group(row))].append(row)

    output: list[dict[str, Any]] = []
    for (dataset, system, task_type, question_type), items in grouped.items():
        with_cmr_m = [item for item in items if item.get("stage1", {}).get("CMR_M") == 1]
        with_cmr_r = [item for item in items if item.get("stage3", {}).get("CMR_R") == 1]
        with_cmr_p = [item for item in items if item.get("stage4", {}).get("CMR_P") == 1]
        m_to_r = [item for item in with_cmr_m if item.get("stage3", {}).get("CMR_R") == 1]
        r_to_p = [item for item in with_cmr_r if item.get("stage4", {}).get("CMR_P") == 1]
        m_to_p = [item for item in with_cmr_m if item.get("stage4", {}).get("CMR_P") == 1]
        output.append(
            {
                "dataset": dataset,
                "system": system,
                "task_type": task_type,
                "question_type": question_type,
                "num_samples": len(items),
                "CMR_M_rate": _mean([item.get("stage1", {}).get("CMR_M") for item in items]),
                "CMR_R_rate": _mean([item.get("stage3", {}).get("CMR_R") for item in items]),
                "CMR_P_rate": _mean([item.get("stage4", {}).get("CMR_P") for item in items]),
                "num_with_CMR_M": len(with_cmr_m),
                "num_with_CMR_R": len(with_cmr_r),
                "num_with_CMR_P": len(with_cmr_p),
                "M_to_R_propagation": (len(m_to_r) / len(with_cmr_m)) if with_cmr_m else None,
                "R_to_P_propagation": (len(r_to_p) / len(with_cmr_r)) if with_cmr_r else None,
                "M_to_P_propagation": (len(m_to_p) / len(with_cmr_m)) if with_cmr_m else None,
            }
        )
    return output


def _acc(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(float(row.get("stage5", {}).get("Acc", 0)) for row in rows) / len(rows)


def _is_t_star(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    stage1 = row.get("stage1", {})
    stage3 = row.get("stage3", {})
    stage4 = row.get("stage4", {})
    q_fcr = stage1.get("Q-FCR")
    rfr = _rfr_value(row)
    pcr = stage4.get("PCR")
    cmr_p = stage4.get("CMR_P")
    return (
        isinstance(q_fcr, (int, float))
        and q_fcr > thresholds.get("q_fcr_threshold", 0.65)
        and isinstance(rfr, (int, float))
        and rfr >= thresholds.get("rfr_threshold", 0.70)
        and isinstance(pcr, (int, float))
        and pcr > thresholds.get("pcr_threshold", 0.70)
        and cmr_p == 0
    )


def build_stage_conditioned_summary(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    full_context_index = {
        (row.get("dataset"), row.get("task_type"), _question_group(row), row.get("sample_id")): row
        for row in rows
        if row.get("system") == "full_context"
    }
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset", ""), row.get("system", ""), row.get("task_type", ""), _question_group(row))].append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, system, task_type, question_type), items in grouped.items():
        t_star = [row for row in items if _is_t_star(row, thresholds)]
        full_rows = [
            full_context_index[(dataset, task_type, question_type, row.get("sample_id"))]
            for row in t_star
            if (dataset, task_type, question_type, row.get("sample_id")) in full_context_index
        ]
        acc_system_t_star = _acc(t_star)
        acc_full_t_star = _acc(full_rows)
        summaries.append(
            {
                "dataset": dataset,
                "system": system,
                "task_type": task_type,
                "question_type": question_type,
                "num_samples": len(items),
                "T_star_size": len(t_star),
                "Acc_all": _acc(items),
                "Acc_T_star": acc_system_t_star,
                "FullContext_T_star": acc_full_t_star,
                "UG": (acc_full_t_star - acc_system_t_star) if acc_full_t_star is not None and acc_system_t_star is not None else None,
            }
        )
    return summaries
