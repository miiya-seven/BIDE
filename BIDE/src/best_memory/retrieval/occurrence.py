#!/usr/bin/env python3
"""Direct V4.1 occurrence/antecedent retrieval (no old Candidate, no reranker)."""
import json,os
from collections import defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent;INPUT=Path(os.environ.get('OCCURRENCE_INPUT_DIR',H));OUTPUT=Path(os.environ.get('OCCURRENCE_OUTPUT_DIR',H));OUTPUT.mkdir(parents=True,exist_ok=True);PREFIX=os.environ.get('OCCURRENCE_PREFIX','V41');QUERY=Path(os.environ.get('OCCURRENCE_QUERY',H/'V41_QUERIES_REBUILT.jsonl'))

def rows(name):
    path=Path(name) if isinstance(name,Path) else INPUT/name
    with path.open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]

queries = rows(QUERY)
lex = {x["sample_id"]: x["rankings"] for x in rows("LEXICAL_LANES.jsonl")}
dense = {x["sample_id"]: x["rankings"] for x in rows("DENSE_LANES.jsonl")}
edges = rows(f"{PREFIX}_GRAPH_EDGES.jsonl")
props = rows(f"{PREFIX}_PROPOSITIONS.jsonl")
if os.environ.get('BEST276_DISABLE_L3')=='1':
    edges=[]
    props=[]  # No proposition-source link traversal; direct lanes stay intact.
raws = rows(f"{PREFIX}_RAW_VIEWS.jsonl")

graph = defaultdict(lambda: defaultdict(list))
for e in edges:
    a, b, typ = e["source_raw_id"], e["target_raw_id"], e["edge_type"]
    if b not in [x[0] for x in graph[a][typ]]:
        graph[a][typ].append((b, 1.0))
    if a not in [x[0] for x in graph[b][typ]]:
        graph[b][typ].append((a, 1.0))
for p in props:
    a = p["raw_id"]
    for b in p.get("source_raw_ids", []):
        if b != a and b not in [x[0] for x in graph[a]["PROPOSITION_SOURCE"]]:
            graph[a]["PROPOSITION_SOURCE"].append((b, 1.0))
            graph[b]["PROPOSITION_SOURCE"].append((a, 1.0))

by_conv = defaultdict(list)
for x in raws:
    by_conv[x["conversation_id"]].append(x["raw_id"])

seed_specs = {
    "occ_raw": [("d", "raw_question", 1.0), ("l", "raw", .8)],
    "occ_assertion": [("d", "assertion_question", 1.0), ("d", "assertion_structured", .8), ("l", "assertion", .7)],
    "occ_structured": [("d", "raw_structured", 1.0), ("d", "assertion_structured", 1.0), ("l", "proposition", .6)],
    "occ_consensus": [("d", "raw_question", 1.0), ("d", "raw_structured", .8), ("d", "assertion_question", .8),
                      ("d", "assertion_structured", .7), ("l", "raw", .7), ("l", "assertion", .7), ("l", "proposition", .35)],
}
edge_weight = {"ANTECEDENT": .95, "OCCURRENCE": .90, "CONFIRMATION": .92,
               "DENIAL": .92, "PROPOSITION_SOURCE": .95}

def direct_scores(sid, spec):
    score = defaultdict(float)
    for source, lane, weight in spec:
        ranking = dense[sid].get(lane, []) if source == "d" else lex[sid].get(lane, [])
        for rank, rid in enumerate(ranking[:128], 1):
            score[rid] += weight / (20.0 + rank)
    return score

def expand(base, paired=False):
    score = defaultdict(float, base)
    # Only strong seeds launch graph traversal; this is direct graph recall, not a rerank stage.
    seeds = sorted(base, key=lambda r: (-base[r], r))[:48]
    for rid in seeds:
        s = base[rid]
        for typ, neighbours in graph.get(rid, {}).items():
            w = edge_weight.get(typ, .75)
            for nb, _ in neighbours:
                score[nb] += s * w
                if paired:
                    # Preserve both occurrence/source members as an evidence group.
                    score[rid] += s * .08
                    for typ2, n2s in graph.get(nb, {}).items():
                        if typ2 in ("ANTECEDENT", "OCCURRENCE", "PROPOSITION_SOURCE"):
                            for nb2, _ in n2s:
                                if nb2 != rid:
                                    score[nb2] += s * w * .30
    return score

out = []
for q in queries:
    sid, cid = q["sample_id"], q["conversation_id"]
    rankings = {}
    for arm, spec in seed_specs.items():
        base = direct_scores(sid, spec)
        scores = expand(base, paired=(arm == "occ_consensus"))
        ranked = sorted(scores, key=lambda r: (-scores[r], r))
        # Guarantee a strict 128 contract without importing any prior Candidate.
        if len(ranked) < 128:
            ranked += [r for r in by_conv[cid] if r not in scores]
        rankings[arm] = ranked[:128]
    out.append({"sample_id": sid, "rankings": rankings, "candidate_size": 128, "gold_visible": False})

with (OUTPUT / "OCCURRENCE_LANES.jsonl").open("w", encoding="utf8") as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
(OUTPUT / "OCCURRENCE_RECEIPT.json").write_text(json.dumps({
    "questions": len(out), "graph_edges": len(edges), "propositions": len(props),
    "arms": list(seed_specs), "candidate_size": 128, "old_candidate_used": False,
    "reranker_used": False, "gold_used": False,
    "query_source": "V41_QUERIES_REBUILT.jsonl",
    "query_schema": "v41-query-rebuilt-v1"
}, indent=2) + "\n")
print((OUTPUT / "OCCURRENCE_RECEIPT.json").read_text())
