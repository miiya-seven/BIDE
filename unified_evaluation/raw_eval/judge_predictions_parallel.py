from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from _bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from map_platform.datasets import load_dataset
from ab_memory.code.shared.runtime.model_clients import ModelClients


LABELS = {"correct", "partial", "incorrect"}
PLACEHOLDERS = {"", "dummy", "none", "null", "placeholder", "your_key", "your_openai_api_key"}


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    apply_config_env(config.get("env", {}))
    # Resolve the same URL/credential pair used by the memory Answer stage.
    # Keep the secret process-local and never write it to a receipt or log.
    model_clients = ModelClients(Path(args.config))
    if not os.getenv("OPENAI_API_KEY") and model_clients.llm_key:
        os.environ["OPENAI_API_KEY"] = model_clients.llm_key
    if not os.getenv("OPENAI_BASE_URL") and model_clients.llm_base:
        os.environ["OPENAI_BASE_URL"] = model_clients.llm_base

    output_dir = Path(args.output_dir or Path(args.predictions).parent / f"llm_judge_{safe_name(args.model or judge_model(config))}")
    output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = output_dir / "judgments.jsonl"
    summary_path = output_dir / "summary.csv"
    summary_by_type_path = output_dir / "summary_by_type.csv"

    samples = load_dataset(args.dataset or config.get("dataset", "locomo"), args.data_path or config["data_path"])
    sample_index = {sample.sample_id: sample for sample in samples}
    predictions = load_jsonl(Path(args.predictions))
    completed = load_completed(judgments_path)
    pending = [row for row in predictions if row_key(row) not in completed]

    model = args.model or judge_model(config)
    base_url = os.getenv("OPENAI_BASE_URL")
    workers = args.workers
    print(f"[judge-start] predictions={len(predictions)} pending={len(pending)} model={model} base_url={base_url} workers={workers}")

    write_lock = Lock()
    rows: list[dict[str, Any]] = []
    if judgments_path.exists():
        rows.extend(load_jsonl(judgments_path))

    def handle(pred: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(pred.get("sample_id") or "")
        sample = sample_index.get(sample_id)
        if sample is None:
            return error_row(pred, model, f"missing sample: {sample_id}")
        return judge_one(sample, pred, model=model, max_retries=args.max_retries)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(handle, pred): pred for pred in pending}
        done = 0
        for future in as_completed(future_map):
            row = future.result()
            done += 1
            rows.append(row)
            with write_lock:
                append_jsonl(judgments_path, row)
            if done % args.progress_every == 0 or done == len(pending):
                print(f"[judge-progress] done={done}/{len(pending)} last={row.get('sample_id')} label={row.get('judge_label')} error={row.get('error', '')}")

    rows = canonicalize_judgments(load_jsonl(judgments_path))
    canonical_path = output_dir / "judgments_canonical.jsonl"
    write_jsonl(canonical_path, rows)
    write_summary(rows, summary_path, key_fields=[])
    write_summary(rows, summary_by_type_path, key_fields=["question_type"])
    print(f"[judge-done] wrote {judgments_path}")
    print(f"[judge-done] wrote {summary_path}")
    print(f"[judge-done] wrote {summary_by_type_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel LLM judge for prediction JSONL files.")
    parser.add_argument("--config", default="configs/dual_layer_memory.yaml")
    parser.add_argument("--predictions", default="outputs/dual_layer_6.1/predictions.jsonl")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=int(os.getenv("JUDGE_WORKERS", "16")))
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--progress_every", type=int, default=25)
    return parser


def judge_one(sample: Any, pred: dict[str, Any], *, model: str, max_retries: int) -> dict[str, Any]:
    from openai import OpenAI

    sample_id = sample.sample_id
    system = str(pred.get("system") or "")
    predicted = str(pred.get("hypothesis") or pred.get("pred_answer") or "")
    gold = sample.canonical_answer()
    question = sample.question
    qtype = sample.question_group()
    if is_cat5_empty_answer_correct(qtype, gold, predicted):
        return {
            "version": "dual_layer_6.1",
            "system": system,
            "sample_id": sample_id,
            "question_type": qtype,
            "strict_correct": True,
            "judge_label": "correct",
            "judge_correct": True,
            "judge_partial": False,
            "error_category": "correct",
            "rationale": "Cat5 empty-answer rule: Not enough information is equivalent to an empty answer.",
            "question": question,
            "gold_answer": gold,
            "pred_answer": predicted,
            "judge_model": model,
            "raw_judge_response": "auto_cat5_empty_answer_correct",
        }
    strict_correct = strict_match(predicted, sample.answer_candidates())
    if strict_correct:
        return {
            "version": "dual_layer_6.1",
            "system": system,
            "sample_id": sample_id,
            "question_type": qtype,
            "strict_correct": True,
            "judge_label": "correct",
            "judge_correct": True,
            "judge_partial": False,
            "error_category": "correct",
            "rationale": "strict metric already marked answer correct",
            "question": question,
            "gold_answer": gold,
            "pred_answer": predicted,
            "judge_model": model,
            "raw_judge_response": "auto_strict_correct",
        }

    prompt = build_prompt(question=question, gold=sample.answer_candidates() or [gold], predicted=predicted, question_type=qtype)
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("MODEL_API_KEY")
    )
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "256")),
            )
            raw = response.choices[0].message.content or ""
            parsed = parse_judge(raw)
            return {
                "version": "dual_layer_6.1",
                "system": system,
                "sample_id": sample_id,
                "question_type": qtype,
                "strict_correct": False,
                "judge_label": parsed["label"],
                "judge_correct": parsed["label"] == "correct",
                "judge_partial": parsed["label"] == "partial",
                "error_category": parsed["error_category"],
                "rationale": parsed["rationale"],
                "question": question,
                "gold_answer": gold,
                "pred_answer": predicted,
                "judge_model": model,
                "raw_judge_response": raw,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_retries:
                break
            time.sleep(min(30, 2**attempt))
    return error_row(pred, model, last_error, sample=sample)


def build_prompt(*, question: str, gold: list[str], predicted: str, question_type: str) -> str:
    return f"""You are a strict but fair evaluation judge for a long-memory QA benchmark.

Question type: {question_type}
Question: {question}
Reference answer(s): {gold}
Predicted answer: {predicted}

Decide whether the prediction answers the question correctly.

Labels:
- correct: semantically equivalent to the reference; wording/date format may differ.
- partial: contains a supported part of the answer but misses important required information, is overbroad, or mixes correct and unsupported details.
- incorrect: wrong, contradicted, irrelevant, or says there is not enough information when the reference is answerable.

Return strict JSON only:
{{
  "label": "correct|partial|incorrect",
  "error_category": "correct|answerable_but_abstained|semantic_fact_mismatch|temporal_mismatch|entity_mismatch|list_answer_missing_or_overbroad|irrelevant_or_empty|other",
  "rationale": "one short sentence"
}}
"""


def parse_judge(raw: str) -> dict[str, str]:
    text = str(raw or "").strip()
    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = None
    if not isinstance(parsed, dict):
        lowered = text.lower()
        if "partial" in lowered:
            label = "partial"
        elif "correct" in lowered and "incorrect" not in lowered:
            label = "correct"
        else:
            label = "incorrect"
        return {"label": label, "error_category": "other" if label != "correct" else "correct", "rationale": text[:240]}
    label = str(parsed.get("label") or "").strip().lower()
    if label not in LABELS:
        label = "incorrect"
    category = str(parsed.get("error_category") or ("correct" if label == "correct" else "other")).strip()
    rationale = str(parsed.get("rationale") or "").strip()
    return {"label": label, "error_category": category, "rationale": rationale}


def strict_match(predicted: str, candidates: list[str]) -> bool:
    pred = normalize(predicted)
    if not pred:
        return False
    for candidate in candidates:
        gold = normalize(candidate)
        if gold and (pred == gold or gold in pred):
            return True
    return False


def is_cat5_empty_answer_correct(question_type: str, gold_answer: str, predicted: str) -> bool:
    if str(question_type) != "5":
        return False
    if normalize(gold_answer):
        return False
    return normalize(predicted) in {
        "not enough information",
        "not enough info",
        "insufficient information",
        "unknown",
        "no answer",
        "none",
        "",
    }


def normalize(text: str) -> str:
    lowered = str(text or "").lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def error_row(pred: dict[str, Any], model: str, error: str, *, sample: Any | None = None) -> dict[str, Any]:
    return {
        "version": "dual_layer_6.1",
        "system": str(pred.get("system") or ""),
        "sample_id": str(pred.get("sample_id") or ""),
        "question_type": sample.question_group() if sample else "",
        "strict_correct": False,
        "judge_label": "error",
        "judge_correct": False,
        "judge_partial": False,
        "error_category": "judge_error",
        "rationale": "",
        "question": sample.question if sample else "",
        "gold_answer": sample.canonical_answer() if sample else "",
        "pred_answer": str(pred.get("hypothesis") or pred.get("pred_answer") or ""),
        "judge_model": model,
        "raw_judge_response": "",
        "error": error,
    }


def write_summary(rows: list[dict[str, Any]], path: Path, *, key_fields: list[str]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        groups.setdefault(key, []).append(row)
    fieldnames = key_fields + ["n", "strict_correct", "strict_accuracy", "judge_correct", "judge_accuracy", "partial", "correct_plus_partial", "judge_errors"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, items in sorted(groups.items(), key=lambda item: item[0]):
            n = len(items)
            strict_n = sum(1 for row in items if row.get("strict_correct"))
            judge_n = sum(1 for row in items if row.get("judge_correct"))
            partial_n = sum(1 for row in items if row.get("judge_partial"))
            errors = sum(1 for row in items if row.get("judge_label") == "error")
            writer.writerow(
                {
                    **{field: value for field, value in zip(key_fields, key)},
                    "n": n,
                    "strict_correct": strict_n,
                    "strict_accuracy": strict_n / n if n else 0.0,
                    "judge_correct": judge_n,
                    "judge_accuracy": judge_n / n if n else 0.0,
                    "partial": partial_n,
                    "correct_plus_partial": (judge_n + partial_n) / n if n else 0.0,
                    "judge_errors": errors,
                }
            )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_config_env(env: dict[str, Any]) -> None:
    for key, value in env.items():
        text = str(value or "").strip()
        if text.lower() in PLACEHOLDERS:
            continue
        os.environ[str(key)] = text


def judge_model(config: dict[str, Any]) -> str:
    return str(config.get("generation_model") or os.getenv("LLM_MODEL") or "gpt-5.4-mini")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def canonicalize_judgments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one result per prediction, preferring a successful Judge result.

    The append-only journal can contain a transport-error row followed by a
    successful resume.  Such retries must not inflate the evaluation
    denominator or overwrite a valid judgment with a later error.
    """
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        current = chosen.get(key)
        row_ok = row.get("judge_label") in LABELS
        current_ok = current is not None and current.get("judge_label") in LABELS
        if current is None or row_ok or not current_ok:
            chosen[key] = row
    return [chosen[key] for key in sorted(chosen)]


def load_completed(path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    for row in load_jsonl(path):
        if row.get("judge_label") in LABELS:
            completed.add((str(row.get("sample_id") or ""), str(row.get("system") or "")))
    return completed


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("sample_id") or ""), str(row.get("system") or ""))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "judge"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
