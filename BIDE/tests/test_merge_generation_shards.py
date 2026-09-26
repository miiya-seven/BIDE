import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "src/best_memory/memory/merge_generation_shards.py"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")


def build_run(tmp_path, *, duplicate=False, missing=False):
    memory = tmp_path / "memory"
    output = memory / "MEMORY_OUTPUT"
    output.mkdir(parents=True)
    ids = ["raw-0", "raw-1"]
    write_jsonl(memory / "MEMORY_REQUESTS.jsonl", [{"center": {"raw_id": rid}} for rid in ids])
    shard_ids = [["raw-0"], ([] if missing else ["raw-0" if duplicate else "raw-1"])]
    for index, rows in enumerate(shard_ids):
        suffix = f".shard-{index:03d}-of-002"
        (output / f"RECEIPT{suffix}.json").write_text('{"status":"COMPLETE"}\n', encoding="utf8")
        write_jsonl(output / f"L1_L2_MEMORY{suffix}.jsonl", [{"center_raw_id": rid} for rid in rows])
        write_jsonl(memory / f"V2_STATUS_MANIFEST{suffix}.jsonl", [
            {"raw_id": rid, "status": "SELF_CONTAINED_LLM"} for rid in rows
        ])
    return tmp_path


def run_merge(root):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--run-root", str(root), "--shard-count", "2"],
        text=True, capture_output=True,
    )


def test_complete_two_shard_merge_preserves_request_order(tmp_path):
    root = build_run(tmp_path)
    result = run_merge(root)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (root / "memory/MEMORY_OUTPUT/L1_L2_MEMORY.jsonl").read_text().splitlines()]
    assert [row["center_raw_id"] for row in rows] == ["raw-0", "raw-1"]


@pytest.mark.parametrize("case, message", [("duplicate", "duplicate memory raw_id"), ("missing", "incomplete shard merge")])
def test_invalid_shards_are_rejected(tmp_path, case, message):
    root = build_run(tmp_path, duplicate=case == "duplicate", missing=case == "missing")
    result = run_merge(root)
    assert result.returncode != 0
    assert message in result.stderr
