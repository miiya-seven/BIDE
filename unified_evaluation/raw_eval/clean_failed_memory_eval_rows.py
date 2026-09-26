from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap_project

bootstrap_project()

from map_platform.utils.serialization import to_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove failed or empty memory-eval rows so resume reruns them."
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--systems",
        default=None,
        help="Comma-separated systems. Defaults to every directory under output_root/systems.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    systems_root = output_root / "systems"

    requested = {
        item.strip() for item in str(args.systems or "").split(",") if item.strip()
    }
    if systems_root.exists():
        system_dirs = [
            path
            for path in sorted(systems_root.iterdir())
            if path.is_dir() and (not requested or path.name in requested)
        ]
    elif (output_root / "per_sample_results.jsonl").exists():
        direct_system = next(iter(requested), output_root.name)
        if len(requested) > 1:
            raise ValueError("A direct shard output can only contain one requested system.")
        system_dirs = [output_root]
    else:
        raise FileNotFoundError(
            f"Missing systems directory and direct result file under: {output_root}"
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report: list[dict[str, Any]] = []
    for system_dir in system_dirs:
        report.append(clean_system(system_dir, stamp))

    report_path = output_root / f"cleanup_failed_rows_{stamp}.json"
    write_json(report_path, report)
    for item in report:
        print(
            f"system={item['system']} total={item['total_rows']} "
            f"kept={item['kept_rows']} removed={item['removed_rows']}"
        )
    print(f"report={report_path}")


def clean_system(system_dir: Path, stamp: str) -> dict[str, Any]:
    result_path = system_dir / "per_sample_results.jsonl"
    prediction_path = system_dir / "predictions.jsonl"
    rows = load_jsonl(result_path)
    predictions = load_jsonl(prediction_path)

    latest_rows = latest_by_key(rows, ("sample_id", "system"))
    removed = [row for row in latest_rows if should_rerun(row)]
    kept = [row for row in latest_rows if not should_rerun(row)]
    removed_keys = {
        (str(row.get("sample_id") or ""), str(row.get("system") or system_dir.name))
        for row in removed
    }

    latest_predictions = latest_by_key(predictions, ("sample_id", "system"))
    failed_prediction_keys = {
        (
            str(row.get("sample_id") or ""),
            str(row.get("system") or system_dir.name),
        )
        for row in latest_predictions
        if should_rerun_prediction(row)
    }
    removed_keys.update(failed_prediction_keys)
    kept_predictions = [
        row
        for row in latest_predictions
        if (
            str(row.get("sample_id") or ""),
            str(row.get("system") or system_dir.name),
        )
        not in removed_keys
    ]

    if removed or failed_prediction_keys:
        backup_dir = system_dir / "cleanup_backups" / stamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_if_exists(result_path, backup_dir / result_path.name)
        backup_if_exists(prediction_path, backup_dir / prediction_path.name)
        write_jsonl(backup_dir / "removed_failed_rows.jsonl", removed)
        write_jsonl(result_path, kept)
        write_jsonl(prediction_path, kept_predictions)

    return {
        "system": system_dir.name,
        "total_rows": len(latest_rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "removed_prediction_rows": len(failed_prediction_keys),
        "removed_error_rows": sum(bool(row.get("error")) for row in removed),
        "removed_empty_rows": sum(
            not str(row.get("pred_answer") or "").strip() for row in removed
        ),
        "removed_incomplete_build_rows": sum(
            has_incomplete_memory_build(row) for row in removed
        ),
    }


def should_rerun(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("error"))
        or not str(row.get("pred_answer") or "").strip()
        or has_incomplete_memory_build(row)
    )


def should_rerun_prediction(row: dict[str, Any]) -> bool:
    answer = str(row.get("hypothesis") or "").strip()
    return not answer or answer == "[SYSTEM_ERROR]"


def has_incomplete_memory_build(row: dict[str, Any]) -> bool:
    adapter_debug = (
        row.get("system_diagnostics", {}).get("adapter_debug", {})
        if isinstance(row.get("system_diagnostics"), dict)
        else {}
    )
    return bool(
        isinstance(adapter_debug, dict)
        and (adapter_debug.get("add_errors") or adapter_debug.get("flush_errors"))
    )


def latest_by_key(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    latest: dict[tuple[str, ...], dict[str, Any]] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        key = tuple(str(row.get(name) or "") for name in keys)
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_json(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def backup_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copy2(source, destination)


if __name__ == "__main__":
    main()
