#!/usr/bin/env python3
"""Rebuild legacy episode, role-fact, and structural one-hop retrieval on V4.1."""
import json
import math
import re
import os
from collections import Counter, defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent
INPUT = Path(os.environ.get("STRUCTURAL_INPUT_DIR", H))
OUTPUT = Path(os.environ.get("STRUCTURAL_OUTPUT_DIR", H)); OUTPUT.mkdir(parents=True, exist_ok=True)
STOP = set("a an the and or of to in on at for from with by is are was were be been being what when where who why how which that this those his her their its".split())


def rows(path):
    return [json.loads(line) for line in path.open(encoding="utf8") if line.strip()]


def tokens(value):
    return [x for x in re.findall(r"[a-z0-9']+", str(value).lower()) if len(x) > 2 and x not in STOP]


def bm25(documents, query):
    docs = [tokens(text) for text in documents]
    df = Counter(term for doc in docs for term in set(doc))
    average = sum(map(len, docs)) / max(1, len(docs))
    result = []
    for index, doc in enumerate(docs):
        tf = Counter(doc)
        score = 0.0
        for term in tokens(query):
            if tf[term]:
                inverse = math.log(1 + (len(docs) - df[term] + .5) / (df[term] + .5))
                score += inverse * tf[term] * 2.2 / (tf[term] + 1.2 * (.25 + .75 * len(doc) / max(1, average)))
        result.append((index, score))
    return sorted(result, key=lambda item: (-item[1], item[0]))


raws = rows(INPUT / "V41_RAW_VIEWS.jsonl")
facts = rows(INPUT / "V41_ASSERTIONS.jsonl")
queries = rows(INPUT / "V41_QUERIES_REBUILT.jsonl")
edges = rows(INPUT / "V41_GRAPH_EDGES.jsonl")
postings = rows(INPUT / "V41_ENTITY_POSTINGS.jsonl")
disable_l3 = os.environ.get('BEST276_DISABLE_L3')=='1'
if disable_l3:
    edges=[]
    postings=[]
raw_by_id = {row["raw_id"]: row for row in raws}
raws_by_conversation = defaultdict(list)
facts_by_conversation = defaultdict(list)
for row in raws:
    raws_by_conversation[row["conversation_id"]].append(row)
for row in facts:
    facts_by_conversation[row["conversation_id"]].append(row)

adjacent = defaultdict(set)
for edge in edges:
    adjacent[edge["source_raw_id"]].add(edge["target_raw_id"])
    adjacent[edge["target_raw_id"]].add(edge["source_raw_id"])
entity_members = defaultdict(set)
for posting in postings:
    entity_members[(posting["raw_id"].split("::")[0], str(posting["entity"]).lower())].add(posting["raw_id"])

output = []
for query in queries:
    cid = query["conversation_id"]
    query_text = query["query_text"]
    conversation_raws = raws_by_conversation[cid]
    raw_scores = {conversation_raws[i]["raw_id"]: score for i, score in bm25([x["retrieval_text"] for x in conversation_raws], query_text)}

    episodes = defaultdict(list)
    for raw in conversation_raws:
        day = raw["raw_id"].rsplit("::", 1)[1].split(":", 1)[0]
        episodes[day].append(raw)
    episode_ids = sorted(episodes, key=lambda value: int(value[1:]))
    episode_texts = [" ".join(raw["retrieval_text"] for raw in episodes[eid]) for eid in episode_ids]
    episode_order = bm25(episode_texts, query_text)
    episode_rank = []
    for episode_index, episode_score in episode_order:
        members = sorted(episodes[episode_ids[episode_index]], key=lambda raw: (-raw_scores[raw["raw_id"]], raw["raw_id"]))
        episode_rank.extend((raw["raw_id"], episode_score, raw_scores[raw["raw_id"]]) for raw in members)
    episode_raws = [rid for rid, _, _ in episode_rank]

    conversation_facts = facts_by_conversation[cid]
    fact_texts = [" | ".join(str(fact.get(key) or "") for key in ("subject", "relation", "value", "event", "time", "polarity", "modality", "speech_act", "retrieval_text")) for fact in conversation_facts]
    fact_order = bm25(fact_texts, query_text)
    role_score = defaultdict(float)
    for position, (fact_index, score) in enumerate(fact_order, 1):
        rid = conversation_facts[fact_index]["raw_id"]
        role_score[rid] = max(role_score[rid], score + 1 / (20 + position))
    role_rank = sorted((raw["raw_id"] for raw in conversation_raws), key=lambda rid: (-role_score[rid], -raw_scores[rid], rid))

    hop_score = defaultdict(float)
    for position, rid in enumerate(role_rank[:80], 1):
        hop_score[rid] += 1 / (20 + position)
        for neighbor in adjacent[rid]:
            hop_score[neighbor] += .8 / (20 + position)
    targets = [str(value).lower() for value in query.get("target_entities", [])]
    for (posting_cid, entity), members in entity_members.items():
        if posting_cid == cid and any(target == entity or target in entity or entity in target for target in targets):
            for rid in members:
                hop_score[rid] += .025
    onehop_rank = sorted((raw["raw_id"] for raw in conversation_raws), key=lambda rid: (-hop_score[rid], -role_score[rid], -raw_scores[rid], rid))
    if disable_l3:
        # Direct Raw/L2 retrieval remains; no session-group or graph access.
        episode_raws=sorted(raw_scores,key=lambda rid:(-raw_scores[rid],rid))
        onehop_rank=list(role_rank)
    output.append({
        "sample_id": query["sample_id"], "conversation_id": cid,
        "rankings": {"episode": episode_raws[:128], "role_fact": role_rank[:128], "structural_onehop": onehop_rank[:128]},
        "gold_visible": False,
    })

with (OUTPUT / "V41_LEGACY_STRUCTURAL_LANES.jsonl").open("w", encoding="utf8") as handle:
    for row in output:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"questions": len(output), "lanes": ["episode", "role_fact", "structural_onehop"], "gold_used": False}))
