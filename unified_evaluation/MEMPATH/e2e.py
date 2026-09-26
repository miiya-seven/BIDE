from __future__ import annotations


def build_e2e_result(stage1: dict, stage2: dict, stage3: dict, stage4: dict, stage5: dict) -> dict:
    rfr_key = next((key for key in stage3 if key.startswith("RFR@")), None)
    return {
        "Acc_E2E": stage5.get("Acc"),
        "Q-FCR": stage1.get("Q-FCR"),
        "OQG": stage2.get("OQG"),
        "RAG": stage3.get("RAG"),
        "RFR": stage3.get(rfr_key) if rfr_key else None,
        "TL": stage4.get("TL"),
        "CMR_M": stage1.get("CMR_M"),
        "CMR_R": stage3.get("CMR_R"),
        "CMR_P": stage4.get("CMR_P"),
    }
