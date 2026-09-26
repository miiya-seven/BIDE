#!/usr/bin/env python3
"""Promote explicitly audited memory wires to the permanent content cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from best_memory.memory.run_generation import compile_memory, parse, row_iter


ALLOWED_REPAIRS = {
    "generated_local_assertion_id",
    "normalized_assertion_ids",
    "normalized_answer_type",
    "normalized_mention_provenance",
    "normalized_mentions",
    "normalized_roles",
    "normalized_relation_family",
    "normalized_scope",
    "normalized_units_list",
    "reconstructed_unit_from_assertion_provenance",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--indices", help="Comma-separated, explicitly reviewed absolute indices.")
    parser.add_argument("--all-wires", action="store_true")
    parser.add_argument("--exclude-indices", default="")
    parser.add_argument("--allow-any-validated-repair", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root)
    if bool(args.indices) == bool(args.all_wires):
        raise SystemExit("select exactly one of --indices or --all-wires")
    if args.all_wires:
        requested = {int(path.stem) for path in (root / "memory/MEMORY_OUTPUT/wire").glob("*.json") if path.stem.isdigit()}
    else:
        requested = {int(value) for value in args.indices.split(",") if value.strip()}
    requested -= {int(value) for value in args.exclude_indices.split(",") if value.strip()}
    sources = {i: row for i, row in enumerate(row_iter(root / "memory/MEMORY_REQUESTS.jsonl")) if i in requested}
    if set(sources) != requested:
        raise SystemExit(f"request indices missing: {sorted(requested - set(sources))}")

    promoted = []
    manifest_path = root / "memory/PERMANENT_CONTENT_MANIFEST.jsonl"
    existing = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf8").splitlines():
            if line.strip():
                item = json.loads(line)
                existing[int(item["index"])] = item

    for index in sorted(requested):
        wire = root / f"memory/MEMORY_OUTPUT/wire/{index:04d}.json"
        meta_path = root / f"memory/MEMORY_OUTPUT/wire_meta/{index:04d}.json"
        if not wire.exists():
            raise SystemExit(f"wire missing for index {index}")
        value, _ = parse(wire.read_bytes())
        compiled = compile_memory(value, sources[index])
        repairs = set(compiled.get("_best276_schema_repairs", []))
        disallowed = repairs - ALLOWED_REPAIRS
        if disallowed and not args.allow_any_validated_repair:
            raise SystemExit(f"index {index} has non-permanent repairs: {sorted(disallowed)}")
        meta = (json.loads(meta_path.read_text(encoding="utf8")) if meta_path.exists() else {
            "requested_model": "legacy-unknown",
            "memory_strategy": "legacy-content-import",
            "prompt_sha256": "legacy-unknown",
            "ontology_sha256": "legacy-unknown",
        })
        content_sha = digest(wire)
        meta.update({
            "content_status": "CONTENT_VALIDATED_PERMANENT",
            "content_sha256": content_sha,
            "validation_contract": "best276-l1-l2-v1",
        })
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf8")
        row = {
            "index": index,
            "raw_id": compiled["center_raw_id"],
            "content_sha256": content_sha,
            "status": "CONTENT_VALIDATED_PERMANENT",
            "repairs": sorted(repairs),
        }
        existing[index] = row
        promoted.append(row)

    manifest_path.write_text(
        "".join(json.dumps(existing[index], ensure_ascii=False, sort_keys=True) + "\n" for index in sorted(existing)),
        encoding="utf8",
    )
    print(json.dumps({"status": "COMPLETE", "promoted": promoted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
