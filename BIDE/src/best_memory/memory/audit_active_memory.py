#!/usr/bin/env python3
"""Audit active Memory wires before permanent promotion or prompt migration."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from best_memory.memory.run_generation import compile_memory, parse, row_iter


HIGH_RISK_REPAIRS = {
    "dropped_incomplete_assertion",
    "dropped_missing_answer_value",
    "dropped_malformed_assertion",
    "dropped_malformed_unit",
    "downgraded_unlinked_asserted_unit",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.run_root)
    wire_dir = root / "memory/MEMORY_OUTPUT/wire"
    requested = {int(path.stem) for path in wire_dir.glob("*.json") if path.stem.isdigit()}
    sources = {
        index: row
        for index, row in enumerate(row_iter(root / "memory/MEMORY_REQUESTS.jsonl"))
        if index in requested
    }

    acceptable: list[int] = []
    problematic: list[dict] = []
    repair_counts: Counter[str] = Counter()
    optimization_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()

    for index in sorted(requested):
        path = wire_dir / f"{index:04d}.json"
        try:
            value, usage = parse(path.read_bytes())
            compiled = compile_memory(value, sources[index])
            repairs = set(compiled.get("_best276_schema_repairs", []))
            repair_counts.update(repairs)
            optimization_counts.update(compiled.get("_best276_optimizations", []))
            assertions = compiled.get("l2_direct", [])
            non_string_subjects = sum(
                1 for assertion in assertions
                if not isinstance(assertion.get("subject"), str)
            )
            high_risk = sorted(repairs & HIGH_RISK_REPAIRS)
            if high_risk or non_string_subjects:
                problematic.append({
                    "index": index,
                    "high_risk_repairs": high_risk,
                    "non_string_subjects": non_string_subjects,
                })
                continue
            acceptable.append(index)
            totals["assertions"] += len(assertions)
            totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            totals["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        except Exception as exc:
            problematic.append({
                "index": index,
                "compile_error": f"{type(exc).__name__}:{exc}",
            })

    report = {
        "status": "COMPLETE",
        "active_wires": len(requested),
        "acceptable_count": len(acceptable),
        "problematic_count": len(problematic),
        "acceptable_indices": acceptable,
        "problematic": problematic,
        "repair_counts": dict(sorted(repair_counts.items())),
        "optimization_counts": dict(sorted(optimization_counts.items())),
        "totals": dict(totals),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({key: report[key] for key in (
        "status", "active_wires", "acceptable_count", "problematic_count",
        "repair_counts", "optimization_counts", "totals",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
