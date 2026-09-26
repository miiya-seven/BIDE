from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import bootstrap_project

bootstrap_project()

from map_platform.utils.serialization import to_jsonable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run memory_eval systems in parallel with one process per system.")
    parser.add_argument("--config", default="configs/memory_eval.yaml")
    parser.add_argument("--systems", default=None, help="Comma-separated systems. Defaults to config systems.")
    parser.add_argument("--output_dir", "--output_root", dest="output_dir", default=None, help="Combined output directory.")
    parser.add_argument("--max_parallel", type=int, default=0, help="0 means all systems at once.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--generation_model", default=None)
    parser.add_argument("--embedding_model", default=None)
    parser.add_argument("--llm_provider", default=None)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--openai_base_url", default=None)
    parser.add_argument("--memory_build_openai_api_key", default=None)
    parser.add_argument("--memory_build_openai_base_url", default=None)
    parser.add_argument("--memory_build_model", default=None)
    parser.add_argument("--letta_base_url", default=None)
    parser.add_argument("--letta_api_key", default=None)
    parser.add_argument("--letta_model", default=None)
    parser.add_argument("--letta_embedding_model", default=None)
    parser.add_argument("--letta_model_provider", default=None)
    parser.add_argument("--skip_unavailable", default=None)
    parser.add_argument("--reuse_build_by_conversation", default=None)
    parser.add_argument("--print_qa_results", default=None)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    systems = parse_systems(args.systems if args.systems is not None else config.get("systems"))
    if not systems:
        raise ValueError("No systems configured.")

    output_root = Path(args.output_dir or config.get("output_dir") or "outputs/parallel_eval").expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    child_root = output_root / "systems"
    child_root.mkdir(parents=True, exist_ok=True)

    max_parallel = args.max_parallel if args.max_parallel and args.max_parallel > 0 else len(systems)
    pending = list(systems)
    running: dict[str, subprocess.Popen] = {}
    log_handles: dict[str, Any] = {}
    statuses: dict[str, dict[str, Any]] = {}

    print(f"output_dir={output_root}")
    print(f"systems={','.join(systems)}")
    print(f"max_parallel={max_parallel}")

    while pending or running:
        while pending and len(running) < max_parallel:
            system = pending.pop(0)
            system_dir = child_root / safe_name(system)
            system_dir.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"{safe_name(system)}.log"
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            log_handle.write(
                f"\n[parallel-run-start] system={system} started_at={time.time()}\n"
            )
            command = [
                args.python,
                "-B",
                "scripts/run_memory_eval.py",
                "--config",
                args.config,
                "--systems",
                system,
                "--output_dir",
                str(system_dir),
            ]
            if args.max_samples is not None:
                command.extend(["--max_samples", str(args.max_samples)])
            for arg_name in (
                "generation_model",
                "embedding_model",
                "llm_provider",
                "openai_api_key",
                "openai_base_url",
                "memory_build_openai_api_key",
                "memory_build_openai_base_url",
                "memory_build_model",
                "letta_base_url",
                "letta_api_key",
                "letta_model",
                "letta_embedding_model",
                "letta_model_provider",
                "skip_unavailable",
                "reuse_build_by_conversation",
                "print_qa_results",
            ):
                value = getattr(args, arg_name)
                if value not in (None, ""):
                    command.extend([f"--{arg_name}", str(value)])
            started = time.time()
            process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
            running[system] = process
            log_handles[system] = log_handle
            statuses[system] = {
                "system": system,
                "status": "running",
                "pid": process.pid,
                "started_at": started,
                "output_dir": str(system_dir),
                "log_path": str(log_path),
                "command": " ".join(command),
            }
            write_json(output_root / "parallel_status.json", list(statuses.values()))
            print(f"started system={system} pid={process.pid} log={log_path}", flush=True)

        finished: list[str] = []
        for system, process in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            finished.append(system)
            log_handles[system].close()
            statuses[system].update(
                {
                    "status": "completed" if return_code == 0 else "failed",
                    "return_code": return_code,
                    "finished_at": time.time(),
                    "duration_seconds": time.time() - float(statuses[system]["started_at"]),
                }
            )
            print(f"finished system={system} return_code={return_code}", flush=True)

        for system in finished:
            running.pop(system, None)
            log_handles.pop(system, None)

        write_json(output_root / "parallel_status.json", list(statuses.values()))
        if pending or running:
            time.sleep(5)

    combine_outputs(output_root=output_root, child_root=child_root, systems=systems)
    print(f"combined outputs written to {output_root}")


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def parse_systems(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def combine_outputs(*, output_root: Path, child_root: Path, systems: list[str]) -> None:
    per_sample_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    preflight_rows: list[dict[str, Any]] = []
    for system in systems:
        system_dir = child_root / safe_name(system)
        per_sample_rows.extend(load_jsonl(system_dir / "per_sample_results.jsonl"))
        prediction_rows.extend(load_jsonl(system_dir / "predictions.jsonl"))
        preflight_rows.extend(load_csv(system_dir / "system_preflight.csv"))

    write_jsonl(output_root / "per_sample_results.jsonl", per_sample_rows)
    write_jsonl(output_root / "predictions.jsonl", prediction_rows)
    write_csv(output_root / "system_preflight.csv", preflight_rows)
    write_summary(output_root / "summary.csv", per_sample_rows, ["system"])
    write_summary(output_root / "summary_by_type.csv", per_sample_rows, ["system", "question_type"])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(to_jsonable(row))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]], group_keys: list[str]) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in group_keys), []).append(row)
    fieldnames = group_keys + [
        "num_samples",
        "answer_accuracy",
        "avg_retrieval_recall",
        "retrieval_hit_rate",
        "avg_prompt_gold_coverage",
        "prompt_hit_rate",
        "retrieval_failure_rate",
        "injection_failure_rate",
        "utilization_failure_rate",
        "success_rate",
        "avg_latency",
        "error_rate",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
            out = {key: value for key, value in zip(group_keys, group)}
            out.update(aggregate(group_rows))
            writer.writerow(out)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_samples": len(rows),
        "answer_accuracy": mean(row.get("answer_correct") for row in rows),
        "avg_retrieval_recall": mean(row.get("retrieval_recall") for row in rows),
        "retrieval_hit_rate": mean(row.get("retrieval_hit") for row in rows),
        "avg_prompt_gold_coverage": mean(row.get("prompt_gold_coverage") for row in rows),
        "prompt_hit_rate": mean(row.get("prompt_hit") for row in rows),
        "retrieval_failure_rate": failure_rate(rows, "retrieval_failure"),
        "injection_failure_rate": failure_rate(rows, "injection_failure"),
        "utilization_failure_rate": failure_rate(rows, "utilization_failure"),
        "success_rate": failure_rate(rows, "success"),
        "avg_latency": mean(row.get("latency") for row in rows),
        "error_rate": mean(bool(row.get("error")) for row in rows),
    }


def mean(values: Any) -> float:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else 0.0


def failure_rate(rows: list[dict[str, Any]], failure_type: str) -> float:
    return sum(1 for row in rows if row.get("failure_type") == failure_type) / len(rows) if rows else 0.0


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


if __name__ == "__main__":
    main()
