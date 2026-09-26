#!/usr/bin/env python3
"""Build direct Raw rankings from dense, clause-level views of V4.1 Raw text."""
import json
import re
import time
import urllib.request
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
INPUT = Path(os.environ.get("CLAUSE_INPUT_DIR", H))
OUTPUT = Path(os.environ.get("CLAUSE_OUTPUT_DIR", H)); OUTPUT.mkdir(parents=True, exist_ok=True)
URL = os.environ.get("EMBEDDING_URL", "")
MODEL = "BAAI/bge-m3"


def rows(path):
    return [json.loads(line) for line in path.open(encoding="utf8") if line.strip()]


def split_clauses(text):
    # Keep every fragment attributable to this Raw; no neighbouring Raw is copied.
    sentences = re.split(r"(?<=[.!?])\s+|[;]+\s*|\s+[—–]\s+", text.strip())
    out = []
    for sentence in sentences:
        sentence = sentence.strip(" \t\r\n-–—")
        if not sentence:
            continue
        # Split explicit multi-proposition turns, but retain both sides independently.
        parts = re.split(r",\s+(?=(?:but|however|although|though|yet)\b)", sentence,
                         flags=re.IGNORECASE)
        out.extend(x.strip() for x in parts if x.strip())
    return out or [text.strip() or "empty"]


def embed(texts):
    vectors = []
    for begin in range(0, len(texts), 32):
        payload = json.dumps({"model": MODEL, "input": texts[begin:begin + 32]}).encode()
        request = urllib.request.Request(
            URL, data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("EMBEDDING_API_KEY", "")})
        with urllib.request.urlopen(request, timeout=300) as response:
            data = sorted(json.loads(response.read())["data"], key=lambda x: x["index"])
        vectors.extend(x["embedding"] for x in data)
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)


raws = rows(INPUT / "V41_RAW_VIEWS.jsonl")
queries = rows(INPUT / "V41_QUERIES_REBUILT.jsonl")
started = time.time()


if (os.environ.get("BEST276_DATASET") == "longmemeval" and
        os.environ.get("BEST276_ALLOW_DEGRADED_LANES", "0") == "1"):
    # Clause expansion over 23k long sessions can create hundreds of thousands
    # of embedding inputs.  Keep this lane available for the CPU/no-local-GPU
    # adaptation with an explicit lexical backend; dense raw retrieval remains
    # the primary semantic lane.
    token_re = re.compile(r"[A-Za-z0-9_]+")
    def toks(text): return set(token_re.findall(str(text).lower()))
    by_conversation = defaultdict(list)
    for raw in raws: by_conversation[raw["conversation_id"]].append(raw)
    outputs = []
    lane_names = [f"clause_{dname}_{qname}" for qname in ("question", "structured")
                  for dname in ("plain", "speaker", "temporal")]
    for query in queries:
        qtok = toks(query.get("question", "") + " " + query.get("query_text", ""))
        scored = []
        for raw in by_conversation[query["conversation_id"]]:
            score = float(len(qtok & toks(raw.get("raw_text", ""))))
            scored.append((raw["raw_id"], score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        ranked = [rid for rid, _ in scored[:128]]
        outputs.append({"sample_id": query["sample_id"], "rankings": {name: ranked for name in lane_names}, "gold_visible": False})
    with (OUTPUT / "CLAUSE_LANES_REBUILT_QUERY.jsonl").open("w", encoding="utf8") as handle:
        for output in outputs: handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    receipt = {"questions": len(queries), "raws": len(raws), "clauses": 0,
               "model": MODEL, "backend": "degraded_lexical", "score": "lexical_overlap_fallback",
               "degraded_reason": "longmemeval_session_scale", "elapsed_seconds": time.time() - started,
               "gold_used": False}
    (OUTPUT / "CLAUSE_REBUILT_QUERY_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0)

clauses = []
for raw in raws:
    for ordinal, clause in enumerate(split_clauses(raw["raw_text"])):
        base = {"raw_id": raw["raw_id"], "conversation_id": raw["conversation_id"],
                "ordinal": ordinal, "clause": clause}
        base["plain"] = clause
        base["speaker"] = f'{raw.get("speaker", "")} {clause}'.strip()
        base["temporal"] = f'{raw.get("speaker", "")} {raw.get("timestamp", "")} {clause}'.strip()
        clauses.append(base)

doc_views = {
    "plain": embed([x["plain"] for x in clauses]),
    "speaker": embed([x["speaker"] for x in clauses]),
    "temporal": embed([x["temporal"] for x in clauses]),
}
query_views = {
    "question": embed([x["question"] for x in queries]),
    "structured": embed([x["query_text"] for x in queries]),
}

clause_by_conversation = defaultdict(list)
for index, clause in enumerate(clauses):
    clause_by_conversation[clause["conversation_id"]].append(index)

outputs = []
for qi, query in enumerate(queries):
    indices = clause_by_conversation[query["conversation_id"]]
    rankings = {}
    for qname, qmatrix in query_views.items():
        for dname, dmatrix in doc_views.items():
            raw_scores = defaultdict(lambda: -2.0)
            scores = dmatrix[indices] @ qmatrix[qi]
            for index, score in zip(indices, scores):
                raw_id = clauses[index]["raw_id"]
                raw_scores[raw_id] = max(raw_scores[raw_id], float(score))
            lane = f"clause_{dname}_{qname}"
            rankings[lane] = sorted(raw_scores, key=lambda rid: (-raw_scores[rid], rid))[:128]
    outputs.append({"sample_id": query["sample_id"], "rankings": rankings, "gold_visible": False})

with (OUTPUT / "CLAUSE_LANES_REBUILT_QUERY.jsonl").open("w", encoding="utf8") as handle:
    for output in outputs:
        handle.write(json.dumps(output, ensure_ascii=False) + "\n")

receipt = {
    "questions": len(queries), "raws": len(raws), "clauses": len(clauses),
    "document_views": list(doc_views), "query_views": list(query_views),
    "model": MODEL, "elapsed_seconds": time.time() - started, "gold_used": False,
}
(OUTPUT / "CLAUSE_REBUILT_QUERY_RECEIPT.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf8")
print(json.dumps(receipt, indent=2))
