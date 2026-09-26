#!/usr/bin/env python3
"""From-scratch Query -> L1/L2/L3 -> Raw Candidate128 retrieval (gold/legacy-candidate free)."""
import json, os
from collections import Counter, defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent
OLD = Path(os.environ.get('BEST276_LEGACY_INPUT', H/'../../../../runs/default/retrieval'))

def rows(path):
    return [json.loads(line) for line in path.open(encoding="utf8") if line.strip()]

def mapping(path):
    return {row["sample_id"]: row["rankings"] for row in rows(path)}

def collapse(*rankings):
    positions = defaultdict(list)
    for ranking in rankings:
        for rank, rid in enumerate(ranking[:128], 1):
            positions[rid].append(rank)
    return sorted(positions, key=lambda rid: (-len(positions[rid]), sum(1/(20+r) for r in positions[rid]), min(positions[rid]), rid))

queries = {x["sample_id"]: x for x in rows(OLD / "V41_QUERIES_REBUILT.jsonl")}
raws = {x["raw_id"]: x for x in rows(H / "V42_RAW_VIEWS.jsonl")}
interface = {x["raw_id"]: x for x in rows(H / "V42_RAW_EVIDENCE_INTERFACE.jsonl")}
lex, dense = mapping(H / "LEXICAL_LANES.jsonl"), mapping(H / "DENSE_LANES.jsonl")
multi, occ = mapping(H / "V42_MULTIVECTOR_LANES.jsonl"), mapping(H / "OCCURRENCE_LANES.jsonl")
clause = mapping(OLD / "CLAUSE_LANES_REBUILT_QUERY.jsonl")
struct = mapping(OLD / "V41_LEGACY_STRUCTURAL_LANES.jsonl")

adjacent = defaultdict(set)
for edge in rows(H / "V42_GRAPH_EDGES.jsonl"):
    a, b = edge.get("source_raw_id"), edge.get("target_raw_id")
    if a in raws and b in raws:
        adjacent[a].add(b); adjacent[b].add(a)
for rid, item in interface.items():
    for other in item.get("antecedent_raw_ids") or []:
        if other in raws:
            adjacent[rid].add(other); adjacent[other].add(rid)

def layer_streams(sid):
    l1 = [
        dense[sid]["raw_question"], dense[sid]["raw_structured"],
        lex[sid]["raw"], multi[sid]["question"],
        clause[sid]["clause_plain_question"], clause[sid]["clause_temporal_structured"],
    ]
    l2 = [
        dense[sid]["assertion_question"], dense[sid]["assertion_structured"],
        lex[sid]["assertion"], lex[sid]["entity"],
        multi[sid]["structured"],
    ]
    l3 = [
        lex[sid]["graph"], lex[sid]["proposition"], dense[sid]["proposition"],
        occ[sid]["occ_raw"], occ[sid]["occ_consensus"],
        struct[sid]["episode"], struct[sid]["role_fact"], struct[sid]["structural_onehop"],
    ]
    return l1, l2, l3

def complete(result, streams):
    seen = set(result)
    merged = collapse(*streams)
    for rid in merged:
        if rid not in seen:
            result.append(rid); seen.add(rid)
        if len(result) == 128: break
    if len(result) != 128 or len(seen) != 128:
        raise ValueError("not strict Candidate128")
    return result

def build(sid, config):
    l1, l2, l3 = layer_streams(sid)
    kind = (queries[sid].get("query_v41") or {}).get("query_kind", "SINGLE_FACT")
    # Query type controls discovery emphasis, while every layer remains represented.
    quotas = config["default"]
    if kind in {"SET_COLLECTION", "COMPARISON"}: quotas = config["set"]
    elif kind in {"MULTIHOP", "CAUSE_RESULT"}: quotas = config["multi"]
    elif kind == "TEMPORAL": quotas = config["temporal"]
    streams_by_layer = [l1, l2, l3]
    layer_rankings = [collapse(*x) for x in streams_by_layer]
    result, seen = [], set()

    # Preserve independently corroborated Raw first. This is actual multi-layer discovery,
    # not a legacy-candidate head or a post-hoc reranker.
    layer_pos = [{rid:i for i,rid in enumerate(ranking[:192], 1)} for ranking in layer_rankings]
    universe = set().union(*(set(x) for x in layer_pos))
    consensus = sorted(universe, key=lambda rid:(-sum(rid in x for x in layer_pos), -sum(1/(20+x[rid]) for x in layer_pos if rid in x), rid))
    for rid in consensus:
        if sum(rid in x for x in layer_pos) < config["core_support"]: break
        result.append(rid); seen.add(rid)
        if len(result) == config["core"]: break

    # Fill obligation budgets directly from each layer's own projected Raw ranking.
    for ranking, quota in zip(layer_rankings, quotas):
        added = 0
        for rid in ranking:
            if rid not in seen:
                result.append(rid); seen.add(rid); added += 1
            if added == quota or len(result) == 128: break

    # Bounded explicit closure: only graph neighbors of already discovered anchors.
    closure = []
    for anchor_rank, anchor in enumerate(result):
        for neighbor in adjacent[anchor]:
            if neighbor not in seen:
                support = sum(neighbor in p for p in layer_pos)
                role = interface[neighbor]
                necessary = bool(role.get("is_strong_response") or role.get("is_antecedent") or role.get("antecedent_raw_ids") or role.get("time_signatures"))
                if support or necessary:
                    closure.append((not necessary, -support, anchor_rank, len(adjacent[neighbor]), neighbor))
    for *_, rid in sorted(set(closure)):
        if rid not in seen:
            result.append(rid); seen.add(rid)
        if len(result) == min(128, config["core"] + sum(quotas) + config["closure"]): break
    return complete(result[:128], l1+l2+l3)

configs = {
    "L123_BALANCED": {"core":48,"core_support":2,"default":(34,24,14),"set":(28,30,14),"multi":(26,24,22),"temporal":(34,18,20),"closure":8},
    "L123_RAW_PRIMARY": {"core":48,"core_support":2,"default":(44,18,10),"set":(36,24,12),"multi":(34,18,20),"temporal":(44,12,16),"closure":8},
    "L123_STRUCTURE_PRIMARY": {"core":40,"core_support":2,"default":(30,26,22),"set":(24,32,22),"multi":(22,26,30),"temporal":(30,18,30),"closure":10},
    "L123_HIGH_CONSENSUS": {"core":64,"core_support":2,"default":(26,18,12),"set":(22,22,12),"multi":(20,18,18),"temporal":(26,12,18),"closure":8},
}

output = []
for sid in sorted(queries):
    arms = {name: build(sid, cfg) for name, cfg in configs.items()}
    output.append({"sample_id":sid,"rankings":arms,"candidate_size":128,"old_candidate_used":False,"gold_visible":False})
path = H / "V42_FULL_L123_CANDIDATE128.jsonl"
with path.open("w",encoding="utf8") as handle:
    for row in output: handle.write(json.dumps(row,ensure_ascii=False)+"\n")
receipt={"status":"COMPLETE","questions":len(output),"candidate_size":128,"arms":list(configs),"old_candidate_used":False,"gold_used":False,"mechanism":"query-routed L1/L2/L3 Raw projection plus bounded explicit closure"}
(H/"V42_FULL_L123_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
