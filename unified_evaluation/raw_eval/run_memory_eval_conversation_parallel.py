from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import bootstrap_project

bootstrap_project()

from map_platform.datasets import load_dataset
from map_platform.scripts.run_memory_eval import (
    conversation_key,
    latest_prediction_rows,
    latest_rows_by_sample_system,
    load_jsonl,
    parse_conversation_ids,
    write_jsonl,
    write_stage2_reports,
    write_summary,
    write_trace_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one memory-eval process per conversation, then merge the shard outputs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--systems",
        default="mnemis",
        help="Comma-separated systems passed to each conversation shard. Defaults to mnemis.",
    )
    parser.add_argument(
        "--mnemis_group_id_prefix",
        default=None,
        help="Override MNEMIS_GROUP_ID_PREFIX in a generated run config.",
    )
    parser.add_argument("--conversation_ids", default=None)
    parser.add_argument(
        "--sample_ids_file",
        default=None,
        help="Optional newline-delimited exact sample ids; forwarded per conversation to run_memory_eval.py.",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=2,
        help="Maximum concurrent conversations. Mnemis defaults to 2 to avoid overloading Neo4j and the LLM API.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll_seconds", type=float, default=2.0)
    parser.add_argument(
        "--publish_seconds",
        type=float,
        default=10.0,
        help="Publish current shard rows into systems/<system> while the run is active.",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="Append every conversation worker to one log file instead of per-conversation logs.",
    )
    parser.add_argument(
        "--env_override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one config env value for this run. May be repeated.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    systems = parse_systems(args.systems)
    if not systems:
        raise ValueError("No systems selected.")
    output_root = Path(args.output_dir or config["output_dir"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_handle = acquire_run_lock(output_root)
    run_config_path = prepare_run_config(
        config=config,
        source_path=args.config,
        output_root=output_root,
        mnemis_group_id_prefix=args.mnemis_group_id_prefix,
        env_overrides=parse_env_overrides(args.env_override),
    )
    run_name = safe_name("-".join(systems))
    shards_root = output_root / "conversation_shards" / run_name
    logs_root = output_root / "logs" / "conversation_parallel" / run_name
    shards_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(config["dataset"], config["data_path"], max_samples=config.get("max_samples"))
    requested = parse_requested_conversations(args.conversation_ids)
    requested_sample_ids = (
        {
            line.strip() for line in Path(args.sample_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if args.sample_ids_file else set()
    )
    samples_by_conversation: dict[str, list[Any]] = {}
    for sample in samples:
        conversation_id = conversation_key(sample)
        if requested and conversation_id not in requested:
            continue
        if requested_sample_ids and str(sample.sample_id) not in requested_sample_ids:
            continue
        samples_by_conversation.setdefault(conversation_id, []).append(sample)
    conversations = sorted(samples_by_conversation)
    if not conversations:
        raise ValueError("No conversations remain after applying --conversation_ids.")
    max_parallel = max(1, min(args.max_parallel, len(conversations)))
    pending = list(conversations)
    running: dict[str, subprocess.Popen[Any]] = {}
    log_handles: dict[str, Any] = {}
    statuses: dict[str, dict[str, Any]] = {}
    project_root = Path(__file__).resolve().parents[1]
    last_publish = 0.0

    print(f"output_dir={output_root}", flush=True)
    print(f"systems={','.join(systems)}", flush=True)
    print(f"conversations={','.join(conversations)}", flush=True)
    print(f"max_parallel={max_parallel}", flush=True)

    try:
        while pending or running:
            while pending and len(running) < max_parallel:
                conversation_id = pending.pop(0)
                shard_dir = shards_root / conversation_id
                log_path = (
                    Path(args.log_file).expanduser().resolve()
                    if args.log_file
                    else logs_root / f"{conversation_id}.log"
                )
                shard_dir.mkdir(parents=True, exist_ok=True)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("a", encoding="utf-8", buffering=1)
                log_handle.write(
                    f"\n[conversation-run-start] conversation={conversation_id} "
                    f"systems={','.join(systems)} started_at={time.time()}\n"
                )
                command = [
                    args.python,
                    "-B",
                    str(project_root / "scripts" / "run_memory_eval.py"),
                    "--config",
                    str(run_config_path),
                    "--systems",
                    ",".join(systems),
                    "--conversation_ids",
                    conversation_id,
                    "--output_dir",
                    str(shard_dir),
                    "--print_qa_results",
                    "false",
                ]
                if requested_sample_ids:
                    command.extend([
                        "--sample_ids",
                        ",".join(str(sample.sample_id) for sample in samples_by_conversation[conversation_id]),
                    ])
                started = time.time()
                process = subprocess.Popen(
                    command,
                    cwd=project_root,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[conversation_id] = process
                log_handles[conversation_id] = log_handle
                statuses[conversation_id] = {
                    "conversation_id": conversation_id,
                    "status": "running",
                    "pid": process.pid,
                    "started_at": started,
                    "output_dir": str(shard_dir),
                    "log_path": str(log_path),
                    "command": command,
                }
                write_status(output_root, statuses)
                print(f"started conversation={conversation_id} pid={process.pid}", flush=True)

            finished: list[str] = []
            for conversation_id, process in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                finished.append(conversation_id)
                log_handles[conversation_id].close()
                validation = validate_shard(
                    Path(statuses[conversation_id]["output_dir"]),
                    expected_sample_ids={
                        str(sample.sample_id)
                        for sample in samples_by_conversation[conversation_id]
                    },
                    systems=systems,
                )
                completed = return_code == 0 and validation["valid"]
                statuses[conversation_id].update(
                    {
                        "status": "completed" if completed else "failed",
                        "return_code": return_code,
                        **validation,
                        "finished_at": time.time(),
                        "duration_seconds": time.time() - statuses[conversation_id]["started_at"],
                    }
                )
                print(
                    f"finished conversation={conversation_id} return_code={return_code} "
                    f"rows={validation['row_count']} errors={validation['row_error_count']} "
                    f"missing={validation['missing_count']}",
                    flush=True,
                )

            for conversation_id in finished:
                running.pop(conversation_id, None)
                log_handles.pop(conversation_id, None)
            write_status(output_root, statuses)
            if time.time() - last_publish >= max(1.0, args.publish_seconds):
                publish_progress_outputs(
                    output_root,
                    shards_root,
                    conversations,
                    systems,
                )
                last_publish = time.time()
            if pending or running:
                time.sleep(max(0.25, args.poll_seconds))

        failed = [item for item in statuses.values() if item.get("status") != "completed"]
        if failed:
            raise RuntimeError(f"{len(failed)} conversation shards failed; inspect {logs_root}")

        merge_outputs(output_root, shards_root, conversations, systems)
        rebuild_combined_outputs(output_root)
        print(f"combined outputs written to {output_root}", flush=True)
    except BaseException:
        terminate_running(running, log_handles)
        raise
    finally:
        lock_handle.close()


def load_config(path: str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def prepare_run_config(
    *,
    config: dict[str, Any],
    source_path: str,
    output_root: Path,
    mnemis_group_id_prefix: str | None,
    env_overrides: dict[str, str],
) -> Path:
    if not mnemis_group_id_prefix and not env_overrides:
        return Path(source_path).expanduser().resolve()

    run_config = dict(config)
    run_config["env"] = dict(config.get("env") or {})
    if mnemis_group_id_prefix:
        run_config["env"]["MNEMIS_GROUP_ID_PREFIX"] = mnemis_group_id_prefix
    run_config["env"].update(env_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "effective_config.yaml"
    path.write_text(
        yaml.safe_dump(run_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Invalid --env_override value: {item!r}; expected KEY=VALUE")
        overrides[key] = value.strip()
    return overrides


def parse_systems(value: Any) -> list[str]:
    if isinstance(value, str):
        return list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if isinstance(value, list):
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    return []


def parse_requested_conversations(value: str | None) -> set[str]:
    return parse_conversation_ids(value)


def validate_shard(
    shard_dir: Path,
    *,
    expected_sample_ids: set[str],
    systems: list[str],
) -> dict[str, Any]:
    rows = latest_rows_by_sample_system(load_jsonl(shard_dir / "per_sample_results.jsonl"))
    expected = {(sample_id, system) for sample_id in expected_sample_ids for system in systems}
    actual = {
        (str(row.get("sample_id") or ""), str(row.get("system") or ""))
        for row in rows
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    row_error_count = sum(1 for row in rows if row.get("error"))
    return {
        "valid": not missing and not unexpected and row_error_count == 0,
        "row_count": len(rows),
        "expected_row_count": len(expected),
        "row_error_count": row_error_count,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_examples": [list(item) for item in missing[:10]],
        "unexpected_examples": [list(item) for item in unexpected[:10]],
    }


def merge_outputs(
    output_root: Path,
    shards_root: Path,
    conversations: list[str],
    systems: list[str],
) -> None:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    preflight_rows: list[dict[str, Any]] = []
    for conversation_id in conversations:
        shard_dir = shards_root / conversation_id
        rows.extend(load_jsonl(shard_dir / "per_sample_results.jsonl"))
        predictions.extend(load_jsonl(shard_dir / "predictions.jsonl"))
        preflight_rows.extend(load_csv(shard_dir / "system_preflight.csv"))

    rows = latest_rows_by_sample_system(rows)
    predictions = latest_prediction_rows(predictions)
    for system in systems:
        system_dir = output_root / "systems" / safe_name(system)
        system_dir.mkdir(parents=True, exist_ok=True)
        new_rows = [row for row in rows if str(row.get("system") or "") == system]
        new_predictions = [
            row for row in predictions if str(row.get("system") or "") == system
        ]
        new_row_keys = {
            (str(row.get("sample_id") or ""), str(row.get("system") or ""))
            for row in new_rows
        }
        new_prediction_keys = {
            (
                str(row.get("question_id") or row.get("sample_id") or ""),
                str(row.get("system") or ""),
            )
            for row in new_predictions
        }
        existing_rows = load_jsonl(system_dir / "per_sample_results.jsonl")
        existing_predictions = load_jsonl(system_dir / "predictions.jsonl")
        system_rows = latest_rows_by_sample_system(
            [
                row
                for row in existing_rows
                if (
                    str(row.get("sample_id") or ""),
                    str(row.get("system") or ""),
                )
                not in new_row_keys
            ]
            + new_rows
        )
        system_predictions = latest_prediction_rows(
            [
                row
                for row in existing_predictions
                if (
                    str(row.get("question_id") or row.get("sample_id") or ""),
                    str(row.get("system") or ""),
                )
                not in new_prediction_keys
            ]
            + new_predictions
        )
        system_preflight = [
            row for row in preflight_rows if str(row.get("system") or "") == system
        ]
        system_preflight = (
            load_csv(system_dir / "system_preflight.csv") + system_preflight
        )
        write_jsonl(system_dir / "per_sample_results.jsonl", system_rows)
        write_jsonl(system_dir / "predictions.jsonl", system_predictions)
        write_csv(system_dir / "system_preflight.csv", latest_preflight_rows(system_preflight))
        write_summary(system_dir / "summary.csv", system_rows, ["system"])
        write_summary(system_dir / "summary_by_type.csv", system_rows, ["system", "question_type"])
        ensure_trace_directories(system_dir, system_rows)
        write_trace_files(system_dir, system_rows)
        write_stage2_reports(system_dir, system_rows)


def publish_progress_outputs(
    output_root: Path,
    shards_root: Path,
    conversations: list[str],
    systems: list[str],
) -> None:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    preflight_rows: list[dict[str, Any]] = []
    for conversation_id in conversations:
        shard_dir = shards_root / conversation_id
        rows.extend(load_jsonl_snapshot(shard_dir / "per_sample_results.jsonl"))
        predictions.extend(load_jsonl_snapshot(shard_dir / "predictions.jsonl"))
        preflight_rows.extend(load_csv(shard_dir / "system_preflight.csv"))

    rows = latest_rows_by_sample_system(rows)
    predictions = latest_prediction_rows(predictions)
    for system in systems:
        system_dir = output_root / "systems" / safe_name(system)
        system_dir.mkdir(parents=True, exist_ok=True)
        system_rows = [
            row for row in rows
            if str(row.get("system") or "") == system and not row.get("error")
        ]
        system_predictions = [
            row for row in predictions
            if str(row.get("system") or "") == system
            and str(row.get("hypothesis") or "").strip()
        ]
        row_keys = {
            (str(row.get("sample_id") or ""), str(row.get("system") or ""))
            for row in system_rows
        }
        prediction_keys = {
            (
                str(row.get("question_id") or row.get("sample_id") or ""),
                str(row.get("system") or ""),
            )
            for row in system_predictions
        }
        existing_rows = load_jsonl_snapshot(system_dir / "per_sample_results.jsonl")
        existing_predictions = load_jsonl_snapshot(system_dir / "predictions.jsonl")
        system_rows = latest_rows_by_sample_system(
            [
                row
                for row in existing_rows
                if (
                    str(row.get("sample_id") or ""),
                    str(row.get("system") or ""),
                )
                not in row_keys
            ]
            + system_rows
        )
        system_predictions = latest_prediction_rows(
            [
                row
                for row in existing_predictions
                if (
                    str(row.get("question_id") or row.get("sample_id") or ""),
                    str(row.get("system") or ""),
                )
                not in prediction_keys
            ]
            + system_predictions
        )
        system_preflight = [
            row for row in preflight_rows if str(row.get("system") or "") == system
        ]
        system_preflight = (
            load_csv(system_dir / "system_preflight.csv") + system_preflight
        )
        write_jsonl(system_dir / "per_sample_results.jsonl", system_rows)
        write_jsonl(system_dir / "predictions.jsonl", system_predictions)
        write_csv(
            system_dir / "system_preflight.csv",
            latest_preflight_rows(system_preflight),
        )


def load_jsonl_snapshot(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                # A worker may be appending the final line while the snapshot is read.
                continue
    return rows


def rebuild_combined_outputs(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    preflight_rows: list[dict[str, Any]] = []
    systems_root = output_root / "systems"
    for system_dir in sorted(path for path in systems_root.iterdir() if path.is_dir()):
        rows.extend(load_jsonl(system_dir / "per_sample_results.jsonl"))
        predictions.extend(load_jsonl(system_dir / "predictions.jsonl"))
        preflight_rows.extend(load_csv(system_dir / "system_preflight.csv"))

    rows = latest_rows_by_sample_system(rows)
    predictions = latest_prediction_rows(predictions)
    write_jsonl(output_root / "per_sample_results.jsonl", rows)
    write_jsonl(output_root / "predictions.jsonl", predictions)
    write_csv(output_root / "system_preflight.csv", latest_preflight_rows(preflight_rows))
    write_summary(output_root / "summary.csv", rows, ["system"])
    write_summary(output_root / "summary_by_type.csv", rows, ["system", "question_type"])
    ensure_trace_directories(output_root, rows)
    write_trace_files(output_root, rows)
    write_stage2_reports(output_root, rows)


def ensure_trace_directories(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    for dataset in {str(row.get("dataset") or "") for row in rows}:
        if dataset:
            (output_dir / "traces" / dataset).mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latest_preflight_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        system = str(row.get("system") or "")
        if system:
            latest[system] = row
    return [latest[system] for system in sorted(latest)]


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned.strip("._") or "item"


def acquire_run_lock(output_root: Path):
    path = output_root / ".conversation_parallel.lock"
    handle = path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"Another conversation-parallel run is already using {output_root}"
        ) from exc
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def terminate_running(
    running: dict[str, subprocess.Popen[Any]],
    log_handles: dict[str, Any],
) -> None:
    for process in running.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in running.values():
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for handle in log_handles.values():
        if not handle.closed:
            handle.close()


def write_status(output_root: Path, statuses: dict[str, dict[str, Any]]) -> None:
    path = output_root / "conversation_parallel_status.json"
    path.write_text(
        json.dumps(list(statuses.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
