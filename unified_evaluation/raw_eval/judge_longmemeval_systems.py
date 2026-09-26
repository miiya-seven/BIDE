from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


LABELS = {"correct", "partial", "wrong", "unanswerable_correct"}
DEFAULT_EXCLUDE_SYSTEMS = {"memorybank"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an LLM-as-judge pass for LongMemEval system outputs."
    )
    parser.add_argument("--input_dir", default="output-longmemeval")
    parser.add_argument("--input_file", default=None, help="Optional direct per_sample_results.jsonl input.")
    parser.add_argument("--config", default="configs/longmemeval_qwen3_32b.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--systems", default=None, help="Comma-separated systems to judge. Defaults to all systems found.")
    parser.add_argument(
        "--exclude_systems",
        default=",".join(sorted(DEFAULT_EXCLUDE_SYSTEMS)),
        help="Comma-separated systems to exclude. Defaults to memorybank.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=260)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--no_dedupe_raw",
        action="store_true",
        help="Do not rewrite judgments.jsonl after this run. Useful for parallel shard workers.",
    )
    parser.add_argument("--judge_strict_correct", action="store_true", help="Also send strict-correct samples to the judge.")
    parser.add_argument("--dry_run", action="store_true", help="Only export judge queues; do not call the model.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir or input_dir / "llm_judge_memory_build_api").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_env = load_config_env(Path(args.config))
    judge_config = resolve_judge_config(args, config_env)
    system_inputs = select_system_inputs(input_dir, args.input_file, args.systems, args.exclude_systems)
    if not system_inputs:
        raise SystemExit("No systems selected for judging.")

    all_records: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for system, system_input in system_inputs:
        rows = load_jsonl(system_input)
        if args.limit is not None:
            rows = rows[: args.limit]
        rows = shard_rows(rows, args.num_shards, args.shard_index)

        queue = [
            build_judge_item(row)
            for row in rows
            if args.judge_strict_correct or not truthy(row.get("answer_correct"))
        ]

        system_output = output_dir / "systems" / system
        system_output.mkdir(parents=True, exist_ok=True)
        write_jsonl(system_output / "judge_queue.jsonl", queue)

        if args.dry_run:
            judgments = []
        else:
            judgments = judge_system_queue(system_output, queue, judge_config, dedupe_raw=not args.no_dedupe_raw)

        strict_correct_rows = [
            auto_correct_record(row)
            for row in rows
            if truthy(row.get("answer_correct")) and not args.judge_strict_correct
        ]
        system_records = strict_correct_rows + judgments
        write_jsonl(system_output / "judgments_all.jsonl", system_records)
        write_csv(system_output / "judgments.csv", system_records)
        write_summary(system_output / "summary.csv", system_records, include_system=False)

        all_records.extend(system_records)
        run_rows.append(
            {
                "system": system,
                "input_rows": len(rows),
                "queued": len(queue),
                "judged": len(judgments),
                "auto_correct": len(strict_correct_rows),
                "dry_run": args.dry_run,
            }
        )

    write_csv(output_dir / "run_status.csv", run_rows)
    write_jsonl(output_dir / "judgments_all.jsonl", all_records)
    write_csv(output_dir / "judgments.csv", all_records)
    write_summary(output_dir / "summary.csv", all_records, include_system=True)
    write_readme(output_dir, args, judge_config, [system for system, _ in system_inputs], run_rows)
    print(f"LLM judge outputs written to {output_dir}")


def load_config_env(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        loaded = yaml.safe_load(handle) or {}
    env = loaded.get("env") if isinstance(loaded, dict) else {}
    return env if isinstance(env, dict) else {}


def resolve_judge_config(args: argparse.Namespace, env: dict[str, Any]) -> dict[str, Any]:
    api_key = (
        args.api_key
        or os.getenv("MEMORY_BUILD_OPENAI_API_KEY")
        or str(env.get("MEMORY_BUILD_OPENAI_API_KEY") or "")
    )
    base_url = (
        args.base_url
        or os.getenv("MEMORY_BUILD_OPENAI_BASE_URL")
        or str(env.get("MEMORY_BUILD_OPENAI_BASE_URL") or "")
        or os.getenv("OPENAI_BASE_URL")
        or "http://localhost:8000/v1"
    )
    model = (
        args.model
        or os.getenv("MEMORY_BUILD_MODEL")
        or str(env.get("MEMORY_BUILD_MODEL") or "")
        or os.getenv("JUDGE_MODEL")
        or "gpt-5.4-mini"
    )
    return {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "model": model,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "max_tokens": args.max_tokens,
    }


def select_system_inputs(
    input_dir: Path,
    input_file_arg: str | None,
    systems_arg: str | None,
    exclude_arg: str | None,
) -> list[tuple[str, Path]]:
    excluded = {item.strip() for item in (exclude_arg or "").split(",") if item.strip()}
    if input_file_arg:
        path = Path(input_file_arg).expanduser().resolve()
        system = infer_system_name(path)
        return [] if system in excluded else [(system, path)]

    flat_input = input_dir / "per_sample_results.jsonl"
    if flat_input.exists():
        system = infer_system_name(flat_input)
        if systems_arg:
            requested = {item.strip() for item in systems_arg.split(",") if item.strip()}
            if system not in requested:
                return []
        return [] if system in excluded else [(system, flat_input)]

    if systems_arg:
        systems = [item.strip() for item in systems_arg.split(",") if item.strip()]
    else:
        systems_root = input_dir / "systems"
        systems = sorted(
            path.name
            for path in systems_root.iterdir()
            if path.is_dir() and (path / "per_sample_results.jsonl").exists()
        )
    return [
        (system, input_dir / "systems" / system / "per_sample_results.jsonl")
        for system in systems
        if system not in excluded
    ]


def infer_system_name(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    return str(row.get("system") or path.parent.name)
    except Exception:
        pass
    return path.parent.name


def shard_rows(rows: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return rows
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit(f"shard_index must be in [0, {num_shards - 1}]")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def judge_system_queue(
    system_output: Path,
    queue: list[dict[str, Any]],
    judge_config: dict[str, Any],
    *,
    dedupe_raw: bool,
) -> list[dict[str, Any]]:
    judgments_path = system_output / "judgments.jsonl"
    judged_by_key = load_existing(judgments_path)
    pending = [item for item in queue if item_key(item) not in judged_by_key]
    if pending:
        with judgments_path.open("a", encoding="utf-8", newline="") as handle:
            for index, item in enumerate(pending, 1):
                result = judge_one(judge_config, item)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 25 == 0:
                    system = item.get("system") or "system"
                    print(f"{system}: judged {index}/{len(pending)} pending")
    deduped = list(load_existing(judgments_path).values())
    if dedupe_raw:
        write_jsonl(judgments_path, deduped)
    return deduped


def judge_one(judge_config: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    prompt = build_prompt(item)
    messages = [
        {"role": "system", "content": "You are a careful answer equivalence judge. Return JSON only."},
        {"role": "user", "content": prompt},
    ]
    started = time.perf_counter()
    text = complete_chat(judge_config, messages)
    parsed = parse_json_object(text)
    label = str(parsed.get("label") or "").strip().lower()
    if label not in LABELS:
        label = "wrong"
    return {
        **item,
        "judge_label": label,
        "judge_correct": label in {"correct", "unanswerable_correct"},
        "judge_partial": label == "partial",
        "judge_reason": str(parsed.get("reason") or "").strip(),
        "judge_raw": text,
        "judge_latency": time.perf_counter() - started,
    }


def complete_chat(judge_config: dict[str, Any], messages: list[dict[str, str]]) -> str:
    url = f"{judge_config['base_url']}/chat/completions"
    payload = {
        "model": judge_config["model"],
        "messages": messages,
        "temperature": 0,
        "max_tokens": judge_config["max_tokens"],
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {judge_config['api_key']}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(int(judge_config["max_retries"]) + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(judge_config["timeout"])) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            return str(parsed["choices"][0]["message"].get("content") or "")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= int(judge_config["max_retries"]):
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"judge request failed after retries: {last_error}")


def build_prompt(item: dict[str, Any]) -> str:
    return (
        "Judge whether the model answer should be counted as correct for the memory benchmark question.\n"
        "Use the reference answer and evidence. Be strict about factual contradictions, but accept equivalent wording.\n"
        "If the model answer contains extra information, count it correct only when the extra information does not contradict the reference/evidence.\n"
        "For date, time, duration, count, and name questions, require the exact intended value unless a paraphrase is clearly equivalent.\n"
        "If the reference answer is empty/unanswerable and the model correctly says it cannot determine, label unanswerable_correct.\n"
        "Use labels: correct, partial, wrong, unanswerable_correct.\n\n"
        f"System: {item['system']}\n"
        f"Sample ID: {item['sample_id']}\n"
        f"Question type: {item['question_type']}\n"
        f"Question: {item['question']}\n"
        f"Reference answer: {item['gold_answer']}\n"
        f"Model answer: {item['pred_answer']}\n"
        f"Strict matcher correct: {item['strict_correct']}\n"
        f"Token F1: {item['answer_f1']}\n"
        f"Gold evidence:\n{item['gold_evidence_text']}\n\n"
        'Return JSON exactly like: {"label":"correct","reason":"short reason"}'
    )


def build_judge_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": row.get("system") or "",
        "sample_id": row.get("sample_id") or "",
        "dataset": row.get("dataset") or "",
        "question_type": row.get("question_type") or "",
        "question": row.get("question") or "",
        "gold_answer": row.get("gold_answer") or "",
        "pred_answer": row.get("pred_answer") or "",
        "strict_correct": truthy(row.get("answer_correct")),
        "answer_f1": row.get("token_f1", row.get("answer_f1")),
        "gold_in_prompt": truthy(row.get("gold_in_prompt")),
        "retrieval_hit": truthy(row.get("retrieval_hit")),
        "failure_stage": row.get("failure_stage") or "",
        "failure_type": row.get("failure_type") or "",
        "gold_memory_ids": "; ".join(str(item) for item in (row.get("gold_memory_ids") or [])),
        "gold_evidence_text": evidence_texts(row),
    }


def auto_correct_record(row: dict[str, Any]) -> dict[str, Any]:
    item = build_judge_item(row)
    return {
        **item,
        "judge_label": "correct",
        "judge_correct": True,
        "judge_partial": False,
        "judge_reason": "strict matcher already marked this sample correct; not re-judged",
        "judge_raw": "",
        "judge_latency": 0.0,
    }


def evidence_texts(row: dict[str, Any]) -> str:
    units = row.get("gold_evidence_units") or []
    lines = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        source_id = unit.get("source_id") or unit.get("turn_id") or ""
        text = " ".join(str(unit.get("text") or "").split())
        lines.append(f"- {source_id}: {text}")
    return "\n".join(lines)


def item_key(row: dict[str, Any]) -> str:
    return f"{row.get('system') or ''}::{row.get('sample_id') or ''}"


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    return {item_key(row): row for row in rows}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]], *, include_system: bool) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        system = str(row.get("system") or "all") if include_system else "all"
        question_type = str(row.get("question_type") or "unknown")
        groups.setdefault((system, "all"), []).append(row)
        groups.setdefault((system, f"type_{question_type}"), []).append(row)

    summary_rows = []
    for (system, scope), group_rows in sorted(groups.items()):
        total = len(group_rows)
        correct = sum(1 for row in group_rows if truthy(row.get("judge_correct")))
        partial = sum(1 for row in group_rows if truthy(row.get("judge_partial")))
        summary_rows.append(
            {
                "system": system if include_system else "",
                "scope": scope,
                "n": total,
                "judge_correct": correct,
                "judge_accuracy": correct / total if total else "",
                "partial": partial,
                "correct_plus_partial": (correct + partial) / total if total else "",
            }
        )
    write_csv(path, summary_rows)


def write_readme(
    output_dir: Path,
    args: argparse.Namespace,
    judge_config: dict[str, Any],
    systems: list[str],
    run_rows: list[dict[str, Any]],
) -> None:
    total_queued = sum(int(row["queued"]) for row in run_rows)
    total_judged = sum(int(row["judged"]) for row in run_rows)
    lines = [
        "# LongMemEval LLM-as-judge",
        "",
        "This pass uses the MEMORY_BUILD API settings as the judge endpoint.",
        "",
        f"- input_dir: `{Path(args.input_dir)}`",
        f"- model: `{judge_config['model']}`",
        f"- base_url: `{judge_config['base_url']}`",
        f"- systems: `{', '.join(systems)}`",
        f"- queued: `{total_queued}`",
        f"- judged_by_llm: `{total_judged}`",
        f"- judge_strict_correct: `{args.judge_strict_correct}`",
        f"- dry_run: `{args.dry_run}`",
        "",
        "Labels:",
        "",
        "- `correct`: semantically correct and counted in judge accuracy.",
        "- `unanswerable_correct`: correct refusal for an unanswerable reference, counted in judge accuracy.",
        "- `partial`: partially correct, tracked separately.",
        "- `wrong`: incorrect.",
        "",
        "Files:",
        "",
        "- `summary.csv`: judge accuracy by system and question type.",
        "- `judgments.csv`: all judged rows, including auto-correct strict matches unless strict matches were re-judged.",
        "- `systems/<system>/judgments.jsonl`: raw LLM-judged rows for that system.",
        "- `systems/<system>/judge_queue.jsonl`: rows sent or pending for LLM judge.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, "", 0, 0.0):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok", "是"}
    return bool(value)


if __name__ == "__main__":
    main()
