#!/usr/bin/env python3
"""Fuse historical query parses into one 1540-question retrieval contract."""
import json, os
from pathlib import Path

H = Path(__file__).resolve().parent
B = H.parent
RUN = Path(os.environ.get('BEST276_RUN_ROOT', H/'../../../../runs/default'))
V41 = Path(os.environ.get('QUERY_REBUILT', RUN/'query/V41_QUERIES_REBUILT.jsonl'))
DEV = Path(os.environ.get('QUERY_DEV_FRAMES', RUN/'query/QUERY_FRAMES_DEV.jsonl'))
HELD_V2 = Path(os.environ.get('QUERY_HELD_FRAMES', RUN/'query/QUERY_FRAMES_HELD.jsonl'))
HELD_KEYS = Path(os.environ.get('QUERY_HELD_KEYS', RUN/'query/QUERY_KEYS_HELD.jsonl'))
HELD_CANON = Path(os.environ.get('QUERY_HELD_CANON', RUN/'query/QUERY_FRAMES_CANON.jsonl'))


def rows(path): return [json.loads(x) for x in path.open(encoding="utf8") if x.strip()]
def index(path): return {x["sample_id"]: x for x in rows(path)}
def values(obj):
    out = []
    if isinstance(obj, dict):
        for value in obj.values(): out.extend(values(value))
    elif isinstance(obj, list):
        for value in obj: out.extend(values(value))
    elif obj not in (None, "", "UNKNOWN", "NONE", "N_A", "ANY"):
        out.append(str(obj))
    return out
def unique(items):
    result = []
    for item in items:
        if item and item not in result: result.append(item)
    return result


base, dev, held_keys, held_canon = index(V41), index(DEV), index(HELD_KEYS), index(HELD_CANON)
held_v2 = index(HELD_V2) if HELD_V2.exists() else {}
contracts, retrieval = [], []
for sid, row in base.items():
    frame = row.get("query_v41") or {}; rich = dev.get(sid) or held_v2.get(sid); key = held_keys.get(sid); canonical = held_canon.get(sid)
    if rich:
        bindings = rich.get("bindings") or {}; relation = rich.get("relation") or {}; scope = rich.get("scope") or {}; protection = rich.get("list_protection") or {}
        subject = bindings.get("subject"); evaluated = bindings.get("evaluated_subject"); observer = bindings.get("observer")
        participants = bindings.get("participants") or []; answer_role = rich.get("answer_role")
        direction = relation.get("direction"); relation_family = relation.get("family"); time_owner = scope.get("time_owner")
        temporal = scope.get("temporal"); location = scope.get("location"); distinct_by = protection.get("distinct_by")
        required_sides = protection.get("required_sides") or []; preserve_occurrences = protection.get("preserve_occurrences", False)
        allowed_inference = rich.get("allowed_inference") or []
    else:
        key = key or {}; canonical = canonical or {}; targets = canonical.get("targets") or []
        subject = next((x.get("surface") for x in targets if x.get("role") in {"SUBJECT", "AGENT", "OWNER"}), None)
        evaluated = next((x.get("surface") for x in targets if x.get("surface") != subject), None)
        observer = None; participants = key.get("target_entities") or [x.get("surface") for x in targets]
        answer_role = key.get("answer_type") or canonical.get("answer_type")
        direction = "FORWARD"; relation_family = key.get("relation_family") or " ".join(canonical.get("predicate_terms") or [])
        time_owner = subject; temporal = key.get("temporal_constraint") or " ".join(canonical.get("temporal_surfaces") or [])
        location = key.get("location_constraint") or " ".join(canonical.get("location_surfaces") or [])
        distinct_by = "ANSWER_VALUE" if key.get("evidence_shape") in {"CROSS_EPISODE_SET", "TWO_SIDED_COMPARISON"} else "NONE"
        required_sides = key.get("required_roles") or []; preserve_occurrences = key.get("evidence_shape") in {"CROSS_EPISODE_SET", "TEMPORAL_SEQUENCE"}
        allowed_inference = key.get("allowed_inference") or []
    contract = {
        "sample_id": sid, "conversation_id": row["conversation_id"], "question": row["question"],
        "query_kind": frame.get("query_kind"), "answer_role": answer_role or row.get("answer_type"),
        "roles": {"subject": subject, "evaluated_subject": evaluated, "observer": observer,
                  "participants": unique(participants), "owner_roles": frame.get("owner_roles") or []},
        "relation": {"family": relation_family, "direction": direction, "queries": frame.get("relation_queries") or [],
                     "events": frame.get("event_queries") or []},
        "scope": {"temporal": temporal, "time_owner": time_owner, "location": location,
                  "polarity": frame.get("polarity"), "modality": frame.get("modality")},
        "answer_protection": {"distinct_by": distinct_by, "preserve_occurrences": preserve_occurrences,
                              "required_sides": required_sides},
        "inference": allowed_inference, "graph_operations": frame.get("graph_operations") or [],
        "lexical_expansions": frame.get("lexical_expansions") or [], "gold_visible": False,
        "sources": ["V41", "QUERY_FRAMES_V2" if rich else "HELD_QUERY_KEYS+CANONICAL_V1"],
    }
    direct_view = unique([row["question"], subject, evaluated, relation_family] + participants + (frame.get("event_queries") or []))
    structured_view = unique(direct_view + (frame.get("relation_queries") or []) + [answer_role, direction, temporal, time_owner, location, distinct_by] + required_sides + (frame.get("lexical_expansions") or []))
    contracts.append(contract)
    retrieval.append({"sample_id": sid, "conversation_id": row["conversation_id"], "question": " | ".join(direct_view),
                      "query_text": " | ".join(structured_view), "target_entities": unique([subject, evaluated] + participants),
                      "query_v41": frame, "fused_contract": contract, "gold_visible": False})
for path, data in ((H / "V42_FUSED_QUERY_CONTRACTS.jsonl", contracts), (H / "V42_FUSED_QUERIES.jsonl", retrieval)):
    with path.open("w", encoding="utf8") as handle:
        for row in data: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"status": "COMPLETE", "questions": len(contracts), "dev_v2": len(dev),
                  "held_v2": len(held_v2), "held_fallback": len(held_keys) - len(held_v2),
                  "gold_used": False}, indent=2))
