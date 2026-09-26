#!/usr/bin/env python3
"""Merge session-level memory results into conversation-owned memory.

This stage is deliberately deterministic: model extraction happens per
session, while the benchmark memory owner remains the conversation.  It keeps
all historical assertions, links their source session/turn provenance, and
marks exact subject/relation collisions from earlier sessions as superseded
when a later value replaces them.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path


def rows(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.open(encoding="utf8") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--memory-input", help="session result JSONL; defaults to standard outputs")
    ap.add_argument("--output", help="conversation memory JSONL")
    args = ap.parse_args()
    root = Path(args.run_root)
    requests = rows(root / "memory/MEMORY_REQUESTS.jsonl")
    inputs = [Path(args.memory_input)] if args.memory_input else [
        root / "memory/MEMORY_OUTPUT/L1_L2_MEMORY.jsonl",
        root / "memory/MEMORY_OUTPUT/L1_L2_MEMORY.range-000000-n000001.jsonl",
    ]
    results = {}
    for path in inputs:
        for row in rows(path):
            rid = row.get("center_raw_id") or row.get("raw_id")
            if rid:
                results[str(rid)] = row
    req_by_raw = {str(x.get("center", {}).get("raw_id")): x.get("center", {}) for x in requests}
    conversations = defaultdict(list)
    for rid, result in results.items():
        src = req_by_raw.get(rid, {})
        cid = rid.split("::", 1)[0]
        conversations[cid].append((src, result))

    output = []
    for cid, items in sorted(conversations.items()):
        items.sort(key=lambda pair: (str(pair[0].get("timestamp", "")), str(pair[0].get("source_session_id", "")), str(pair[0].get("raw_id", ""))))
        assertions = []
        for src, result in items:
            rid = str(src.get("raw_id", result.get("center_raw_id", "")))
            sid = str(src.get("source_session_id") or "UNKNOWN")
            for index, assertion in enumerate(result.get("l2_direct", []) or [], 1):
                item = dict(assertion)
                item["conversation_id"] = cid
                item["source_session_id"] = sid
                prov = dict(item.get("provenance") or {})
                prov["raw_ids"] = sorted(set([str(x) for x in (prov.get("raw_ids") or [])] + [rid]))
                item["provenance"] = prov
                item["conversation_assertion_id"] = f"{cid}::a{len(assertions)+1}"
                item["status"] = "CURRENT"
                assertions.append(item)
        # Conservative supersession: only exact subject/relation matches with
        # a changed answer value are marked; unrelated facts are untouched.
        latest = {}
        for item in assertions:
            key = (str(item.get("subject", "")).casefold(), str(item.get("surface_relation", item.get("relation_family", "")).casefold()))
            value = json.dumps(item.get("answer_value"), ensure_ascii=False, sort_keys=True)
            if key in latest and latest[key][0] != value:
                previous = latest[key][1]
                previous["status"] = "SUPERSEDED"
                previous["superseded_by"] = item["conversation_assertion_id"]
            latest[key] = (value, item)
        output.append({"conversation_id": cid, "assertions": assertions, "session_count": len(items), "source_raw_count": len(items), "schema_version": "conversation-memory-v1", "gold_visible": False})

    out = Path(args.output) if args.output else root / "memory/CONVERSATION_MEMORY.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in output), encoding="utf8")
    receipt = {"status": "COMPLETE", "conversations": len(output), "source_results": len(results), "output": str(out), "gold_visible": False}
    (out.parent / "CONVERSATION_MEMORY_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
