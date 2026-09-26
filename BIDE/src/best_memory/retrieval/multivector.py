#!/usr/bin/env python3
"""Faithful BGE-M3 ColBERT MaxSim lanes on V4.1 Raw views and rebuilt queries."""
import hashlib
import json
import time
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

H = Path(__file__).resolve().parent
INPUT=Path(os.environ.get("MULTIVECTOR_INPUT_DIR",H));OUTPUT=Path(os.environ.get("MULTIVECTOR_OUTPUT_DIR",H));OUTPUT.mkdir(parents=True,exist_ok=True);PREFIX=os.environ.get("MULTIVECTOR_PREFIX","V41");QUERY=Path(os.environ.get("MULTIVECTOR_QUERY",H/"V41_QUERIES_REBUILT.jsonl"))
MODEL = Path(os.environ.get("BGE_M3_MODEL", ""))


def rows(path):
    return [json.loads(line) for line in path.open(encoding="utf8") if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


raws = rows(INPUT / f"{PREFIX}_RAW_VIEWS.jsonl")
queries = rows(QUERY)


def _degraded_lanes(reason):
    """Emit deterministic lexical stand-ins when the local CUDA lane is unavailable.

    The official BGE-M3 MaxSim implementation remains the default whenever a
    usable CUDA device is present.  LongMemEval adaptation runs on this host
    also need to finish with the remote dense/reranker services when the local
    driver is unavailable, so this branch is explicit in its receipt and never
    masquerades as a faithful ColBERT score.
    """
    token_re = re.compile(r"[A-Za-z0-9_]+")

    def tokens(text):
        return set(token_re.findall(str(text).lower()))

    def rank(query_text, conversation_id):
        q = tokens(query_text)
        scored = []
        for raw in raws:
            if raw.get("conversation_id") != conversation_id:
                continue
            d = tokens(raw.get("retrieval_text", ""))
            # BM25-like overlap keeps the fallback query-dependent and
            # deterministic without pretending to expose token MaxSim.
            score = sum(1.0 for term in q if term in d)
            scored.append((raw["raw_id"], score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [rid for rid, _ in scored[:128]]

    started_fallback = time.time()
    view_texts = {
        "question": [row.get("question", "") for row in queries],
        "structured": [row.get("query_text", "") for row in queries],
    }
    if os.environ.get("MULTIVECTOR_FUSED_VIEWS") == "1":
        # The fallback preserves the same view names; fused contracts are
        # represented by the structured query text when no token encoder is
        # available.
        for name in ("role_relation", "event", "temporal", "answer_slot"):
            view_texts[name] = [row.get("query_text", "") for row in queries]
    outputs = {name: [] for name in view_texts}
    by_conversation = defaultdict(list)
    for index, raw in enumerate(raws):
        by_conversation[raw["conversation_id"]].append(index)
    for name, texts in view_texts.items():
        output_path = OUTPUT / f"{PREFIX}_MULTIVECTOR_{name.upper()}_TOP128.jsonl"
        for query, text in zip(queries, texts):
            outputs[name].append({
                "sample_id": query["sample_id"],
                "conversation_id": query["conversation_id"],
                "ranking": [{"raw_id": rid, "score": 0.0} for rid in rank(text, query["conversation_id"])],
                "gold_visible": False,
            })
        output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs[name]))
    combined = []
    for index, query in enumerate(queries):
        combined.append({
            "sample_id": query["sample_id"],
            "conversation_id": query["conversation_id"],
            "rankings": {name: [item["raw_id"] for item in output[index]["ranking"]]
                         for name, output in outputs.items()},
            "gold_visible": False,
        })
    combined_path = OUTPUT / f"{PREFIX}_MULTIVECTOR_LANES.jsonl"
    combined_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined))
    receipt = {
        "schema": "v41-multivector-rebuilt-query-v1",
        "questions": len(queries), "raws": len(raws), "top_k": 128,
        "model": "BAAI/bge-m3", "head": "colbert_linear.pt",
        "score": "lexical_overlap_fallback", "backend": "degraded_lexical",
        "degraded_reason": reason, "gold_visible": False,
        "elapsed_seconds": time.time() - started_fallback,
        "inputs_sha256": {"queries": sha256(QUERY), "raw_views": sha256(INPUT / f"{PREFIX}_RAW_VIEWS.jsonl")},
        "outputs_sha256": {name: sha256(OUTPUT / f"{PREFIX}_MULTIVECTOR_{name.upper()}_TOP128.jsonl") for name in outputs},
    }
    (OUTPUT / f"{PREFIX}_MULTIVECTOR_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


allow_degraded = os.environ.get("BEST276_ALLOW_DEGRADED_LANES", "0") == "1"
if allow_degraded and not torch.cuda.is_available():
    _degraded_lanes("cuda_unavailable")
    raise SystemExit(0)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float16
    ).cuda().eval()
    head = torch.nn.Linear(model.config.hidden_size, 1024, bias=True).cuda().half()
    state = torch.load(MODEL / "colbert_linear.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(state if "weight" in state else {"weight": state})
except Exception as exc:
    if allow_degraded:
        _degraded_lanes(f"local_model_unavailable:{type(exc).__name__}")
        raise SystemExit(0)
    raise


@torch.inference_mode()
def encode(texts, batch, max_length):
    encoded = []
    for begin in range(0, len(texts), batch):
        inputs = tokenizer(
            texts[begin:begin + batch], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to("cuda")
        vectors = torch.nn.functional.normalize(
            head(model(**inputs).last_hidden_state[:, 1:]), p=2, dim=-1
        )
        masks = inputs["attention_mask"][:, 1:].bool()
        for vector, mask in zip(vectors, masks):
            encoded.append(vector[mask].cpu())
    return encoded


started = time.time()
doc_vectors = encode([row["retrieval_text"] for row in raws], 16, 256)
query_views = {
    "question": encode([row["question"] for row in queries], 16, 128),
    "structured": encode([row["query_text"] for row in queries], 16, 160),
}
if os.environ.get("MULTIVECTOR_FUSED_VIEWS") == "1":
    def fused_view(row, kind):
        contract = row.get("fused_contract") or {}
        roles = contract.get("roles") or {}; relation = contract.get("relation") or {}; scope = contract.get("scope") or {}
        subject = roles.get("subject") or ""; evaluated = roles.get("evaluated_subject") or ""
        if kind == "role_relation":
            parts = [subject, evaluated, relation.get("family"), relation.get("direction")]
        elif kind == "event":
            parts = [subject, evaluated] + (relation.get("events") or [])
        elif kind == "temporal":
            parts = [scope.get("time_owner") or subject, scope.get("temporal")] + (relation.get("events") or [])
        else:
            parts = [subject, evaluated, contract.get("answer_role"), relation.get("family")]
        return " | ".join(str(x) for x in parts if x)
    for name in ("role_relation", "event", "temporal", "answer_slot"):
        query_views[name] = encode([fused_view(row, name) for row in queries], 16, 128)
by_conversation = defaultdict(list)
for index, raw in enumerate(raws):
    by_conversation[raw["conversation_id"]].append(index)

outputs = {name: [] for name in query_views}
for name, query_vectors in query_views.items():
    output_path = OUTPUT / f"{PREFIX}_MULTIVECTOR_{name.upper()}_TOP128.jsonl"
    if output_path.exists() and len(rows(output_path)) == len(queries):
        outputs[name] = rows(output_path)
        continue
    for query_index, query in enumerate(queries):
        query_vector = query_vectors[query_index].cuda()
        scores = []
        conversation_docs = by_conversation[query["conversation_id"]]
        for begin in range(0, len(conversation_docs), 64):
            indices = conversation_docs[begin:begin + 64]
            lengths = [len(doc_vectors[index]) for index in indices]
            max_tokens = max(lengths)
            document_batch = torch.zeros(
                len(indices), max_tokens, query_vector.shape[-1],
                dtype=query_vector.dtype, device="cuda"
            )
            document_mask = torch.zeros(
                len(indices), max_tokens, dtype=torch.bool, device="cuda"
            )
            for row_index, (doc_index, length) in enumerate(zip(indices, lengths)):
                document_batch[row_index, :length] = doc_vectors[doc_index].cuda()
                document_mask[row_index, :length] = True
            similarities = torch.einsum("qh,bdh->bqd", query_vector, document_batch)
            similarities.masked_fill_(~document_mask[:, None, :], float("-inf"))
            batch_scores = similarities.max(dim=2).values.mean(dim=1).float().cpu()
            scores.extend(
                (raws[doc_index]["raw_id"], float(score))
                for doc_index, score in zip(indices, batch_scores)
            )
        scores.sort(key=lambda item: (-item[1], item[0]))
        outputs[name].append({
            "sample_id": query["sample_id"],
            "conversation_id": query["conversation_id"],
            "ranking": [
                {"raw_id": raw_id, "score": score} for raw_id, score in scores[:128]
            ],
            "gold_visible": False,
        })
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs[name])
    )

for name, output in outputs.items():
    path = OUTPUT / f"{PREFIX}_MULTIVECTOR_{name.upper()}_TOP128.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))

combined = []
for index, query in enumerate(queries):
    combined.append({
        "sample_id": query["sample_id"],
        "conversation_id": query["conversation_id"],
        "rankings": {
            name: [item["raw_id"] for item in output[index]["ranking"]]
            for name, output in outputs.items()
        },
        "gold_visible": False,
    })
(OUTPUT / f"{PREFIX}_MULTIVECTOR_LANES.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined)
)

receipt = {
    "schema": "v41-multivector-rebuilt-query-v1",
    "questions": len(queries),
    "raws": len(raws),
    "top_k": 128,
    "model": "BAAI/bge-m3",
    "head": "colbert_linear.pt",
    "score": "mean_query_token_maxsim",
    "document_view": "V41_RAW_VIEWS.retrieval_text",
    "question_view": "question",
    "structured_view": "query_text",
    "elapsed_seconds": time.time() - started,
    "gold_visible": False,
    "inputs_sha256": {
        "queries": sha256(QUERY),
        "raw_views": sha256(INPUT / f"{PREFIX}_RAW_VIEWS.jsonl"),
    },
    "outputs_sha256": {
        name: sha256(OUTPUT / f"{PREFIX}_MULTIVECTOR_{name.upper()}_TOP128.jsonl")
        for name in outputs
    },
}
(OUTPUT / f"{PREFIX}_MULTIVECTOR_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
