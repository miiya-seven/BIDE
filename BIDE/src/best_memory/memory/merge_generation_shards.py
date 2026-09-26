#!/usr/bin/env python3
"""Validate and merge non-overlapping memory-generation shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--shard-count", required=True, type=int)
    args = parser.parse_args()
    root = Path(args.run_root)
    output = root / "memory/MEMORY_OUTPUT"

    memories = {}
    statuses = {}
    for index in range(args.shard_count):
        suffix = f".shard-{index:03d}-of-{args.shard_count:03d}"
        receipt_path = output / f"RECEIPT{suffix}.json"
        if not receipt_path.exists():
            raise SystemExit(f"missing shard receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        if receipt.get("status") != "COMPLETE":
            raise SystemExit(f"shard is not complete: {receipt_path}")
        for item in rows(output / f"L1_L2_MEMORY{suffix}.jsonl"):
            rid = item["center_raw_id"]
            if rid in memories:
                raise SystemExit(f"duplicate memory raw_id across shards: {rid}")
            memories[rid] = item
        for item in rows(root / "memory" / f"V2_STATUS_MANIFEST{suffix}.jsonl"):
            rid = item["raw_id"]
            if rid in statuses:
                raise SystemExit(f"duplicate status raw_id across shards: {rid}")
            statuses[rid] = item

    expected = [item["center"]["raw_id"] for item in rows(root / "memory/MEMORY_REQUESTS.jsonl")]
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise SystemExit("MEMORY_REQUESTS contains duplicate question-local raw_ids")
    if set(memories) != expected_set or set(statuses) != expected_set:
        missing_memory = len(expected_set - set(memories))
        missing_status = len(expected_set - set(statuses))
        extra_memory = len(set(memories) - expected_set)
        extra_status = len(set(statuses) - expected_set)
        raise SystemExit(
            f"incomplete shard merge: missing_memory={missing_memory} missing_status={missing_status} "
            f"extra_memory={extra_memory} extra_status={extra_status}"
        )

    memory_path = output / "L1_L2_MEMORY.jsonl"
    status_path = root / "memory/V2_STATUS_MANIFEST.jsonl"
    with memory_path.open("w", encoding="utf8") as memory_handle, status_path.open("w", encoding="utf8") as status_handle:
        for rid in expected:
            memory_handle.write(json.dumps(memories[rid], ensure_ascii=False, sort_keys=True) + "\n")
            status_handle.write(json.dumps(statuses[rid], ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "status": "COMPLETE",
        "shards": args.shard_count,
        "rows": len(expected),
        "self_contained": sum(x["status"].startswith("SELF_CONTAINED") for x in statuses.values()),
        "context_required": sum(x["status"] == "CONTEXT_REPARSE_REQUIRED" for x in statuses.values()),
    }
    (output / "SHARD_MERGE_RECEIPT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
