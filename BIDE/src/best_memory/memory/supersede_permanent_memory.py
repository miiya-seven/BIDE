#!/usr/bin/env python3
"""Revoke one permanent cache entry while retaining its audit history."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    meta_path = root / f"memory/MEMORY_OUTPUT/wire_meta/{args.index:04d}.json"
    manifest_path = root / "memory/PERMANENT_CONTENT_MANIFEST.jsonl"
    if not meta_path.exists() or not manifest_path.exists():
        raise SystemExit("permanent metadata or manifest missing")

    meta = json.loads(meta_path.read_text(encoding="utf8"))
    if meta.get("content_status") != "CONTENT_VALIDATED_PERMANENT":
        raise SystemExit(f"index {args.index} is not permanent")
    old_sha = meta.get("content_sha256")
    changed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta.update({
        "content_status": "SUPERSEDED",
        "superseded_reason": args.reason,
        "superseded_at": changed_at,
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf8")

    rows = []
    found = False
    for line in manifest_path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["index"]) == args.index:
            row.update({"status": "SUPERSEDED", "reason": args.reason, "superseded_at": changed_at})
            found = True
        rows.append(row)
    if not found:
        raise SystemExit(f"index {args.index} missing from permanent manifest")
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf8")

    history = root / "memory/PERMANENT_CONTENT_HISTORY.jsonl"
    with history.open("a", encoding="utf8") as handle:
        handle.write(json.dumps({
            "index": args.index,
            "old_content_sha256": old_sha,
            "status": "SUPERSEDED",
            "reason": args.reason,
            "updated_at": changed_at,
        }, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": "SUPERSEDED", "index": args.index, "old_content_sha256": old_sha}))


if __name__ == "__main__":
    main()
