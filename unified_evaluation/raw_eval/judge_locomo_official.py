from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import hashlib
import urllib.error
import urllib.request
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
try:
    from event_evidence_memory.clients import _aet_http_pending, _aet_http_terminal, _sha
except ModuleNotFoundError:
    _AET_LOCK = Lock()
    _AET_SEQUENCE = 0

    def _sha(value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _append_aet_event(row: dict[str, Any]) -> None:
        path = os.getenv("SCHEME_A_ET_HTTP_JOURNAL", "").strip()
        if not path:
            return
        global _AET_SEQUENCE
        with _AET_LOCK:
            _AET_SEQUENCE += 1
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {"sequence": _AET_SEQUENCE, **row}
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _aet_http_pending(*, route: str, payload: dict[str, Any], attempt: int) -> str | None:
        if not os.getenv("SCHEME_A_ET_HTTP_JOURNAL", "").strip():
            return None
        call_id = f"judge-{os.getpid()}-{time.time_ns()}"
        _append_aet_event({
            "event": "http_attempt", "call_id": call_id, "service": "official_judge",
            "stage": "judge", "route": route, "attempt": attempt, "status": "pending",
            "input_sha256": _sha(payload),
        })
        return call_id

    def _aet_http_terminal(
        call_id: str | None, *, route: str, attempt: int, status: str,
        response_sha256: str | None = None, error: str | None = None,
    ) -> None:
        if call_id is None:
            return
        row = {
            "event": "http_attempt", "call_id": call_id, "service": "official_judge",
            "stage": "judge", "route": route, "attempt": attempt, "status": status,
        }
        if response_sha256:
            row["response_sha256"] = response_sha256
        if error:
            row["error"] = error
        _append_aet_event(row)
from model_service_protocol import EXTERNAL_AUTH_BY_BASE_URL, external_credential


JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    apply_env(config.get("env") or {})
    if args.base_url:
        if args.base_url.rstrip("/") not in EXTERNAL_AUTH_BY_BASE_URL:
            raise ValueError("Judge base URL override violates the approved provider profiles")
        os.environ["OPENAI_BASE_URL"] = args.base_url.rstrip("/")
    if args.api_key:
        raise ValueError("--api_key is forbidden; use the endpoint-specific environment credential")
    if str(os.getenv("OPENAI_BASE_URL") or "").rstrip("/") not in EXTERNAL_AUTH_BY_BASE_URL:
        raise ValueError("Judge config must use an approved endpoint/auth profile")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = output_dir / "judgments.jsonl"
    failures_path = output_dir / "judge_failures.jsonl"

    rows = load_rows(args.input)
    if args.questions or args.locomo_data:
        canonical_questions = ({
            str(row.get("sample_id") or ""): row
            for row in load_rows(args.questions)
        } if args.questions else {})
        if args.locomo_data:
            locomo_questions = load_locomo_questions(Path(args.locomo_data))
            for sample_id, reference in locomo_questions.items():
                current = canonical_questions.get(sample_id) or {}
                canonical_questions[sample_id] = {**current, **reference}
        hydrated = []
        for row in rows:
            sample_id = str(row.get("sample_id") or "")
            question = canonical_questions.get(sample_id)
            if question is None:
                raise ValueError(f"Missing canonical question for {sample_id}")
            merged = dict(row)
            merged["question"] = question.get("question") or merged.get("question")
            merged["question_type"] = (
                question.get("question_type") or question.get("original_category")
            )
            merged["gold_answer"] = question.get("gold_answer")
            merged["gold_memory_ids"] = question.get("gold_memory_ids") or []
            hydrated.append(merged)
        rows = hydrated
    if args.exclude_cat5_input:
        rows = [
            row for row in rows
            if str(row.get("question_type") or row.get("original_category") or "") != "5"
        ]
    missing_gold = [
        str(row.get("sample_id") or "")
        for row in rows if row.get("gold_answer") in (None, "")
    ]
    if args.require_gold_answer and missing_gold:
        raise ValueError(
            f"Gold answer missing for {len(missing_gold)} rows; "
            f"examples={missing_gold[:10]}"
        )
    if args.sample_ids:
        wanted = {item.strip() for item in args.sample_ids.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get("sample_id") or "") in wanted]
    if args.sample_ids_file:
        wanted = {
            line.strip()
            for line in Path(args.sample_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        rows = [row for row in rows if str(row.get("sample_id") or "") in wanted]
    if args.limit:
        rows = rows[: args.limit]
    completed = load_completed(judgments_path)
    def effective_judge_system(row: dict[str, Any]) -> str:
        if not args.judge_system_field:
            return args.judge_system
        value = str(row.get(args.judge_system_field) or "").strip()
        if not value:
            raise ValueError(
                f"Missing per-row judge system field {args.judge_system_field!r} "
                f"for {row.get('sample_id')}"
            )
        return value
    pending = []
    for row in rows:
        prior = completed.get(row_key(effective_judge_system(row), row))
        current_answer = str(row.get("pred_answer") or row.get("evaluation_pred_answer") or "")
        if prior is None or normalized_answer(prior.get("generated_answer")) != normalized_answer(current_answer):
            pending.append(row)
    model = args.model or str(os.getenv("LLM_MODEL") or config.get("generation_model") or "gpt-5.4-mini")
    print(
        f"[official-locomo-judge-start] rows={len(rows)} pending={len(pending)} "
        f"judge_system={args.judge_system} model={model} workers={args.workers} "
        f"base_url={os.getenv('OPENAI_BASE_URL')}",
        flush=True,
    )

    lock = Lock()
    done = 0
    failed = 0

    def handle(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        try:
            return True, judge_one(
                row, judge_system=effective_judge_system(row), model=model, args=args
            )
        except Exception as exc:
            if args.stop_on_failure:
                raise
            return False, failure_row(
                row, judge_system=effective_judge_system(row), model=model,
                error=f"{type(exc).__name__}: {exc}"
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(handle, row): row for row in pending}
        for future in as_completed(futures):
            ok, judged = future.result()
            with lock:
                with (judgments_path if ok else failures_path).open("a", encoding="utf-8") as handle_out:
                    handle_out.write(json.dumps(judged, ensure_ascii=False, separators=(",", ":")) + "\n")
            done += int(ok)
            failed += int(not ok)
            processed = done + failed
            if processed % args.progress_every == 0 or processed == len(pending):
                print(
                    f"[official-locomo-judge-progress] processed={processed}/{len(pending)} "
                    f"done={done} failed={failed} last={judged.get('sample_id')} "
                    f"grade={judged.get('grade')}",
                    flush=True,
                )

    clean = list(load_completed(judgments_path).values())
    report_rows = (
        [row for row in clean if str(row.get("question_type")) != "5"]
        if args.exclude_cat5_metrics
        else clean
    )
    write_jsonl(judgments_path, clean)
    write_summary(report_rows, output_dir / "summary.csv", key_fields=["judge_system"])
    write_summary(report_rows, output_dir / "summary_by_type.csv", key_fields=["judge_system", "question_type"])
    write_scope_summary(
        report_rows,
        output_dir / "official_locomo_scope_summary.csv",
        include_cat5_scope=not args.exclude_cat5_metrics,
    )
    write_summary(report_rows, output_dir / "official_locomo_by_type_summary.csv", key_fields=["judge_system", "question_type"])
    print(f"[official-locomo-judge-done] judgments={judgments_path} failures_this_run={failed}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official LoCoMo-style CORRECT/WRONG LLM judge.")
    parser.add_argument("--input", required=True, help="per_sample_results.jsonl path.")
    parser.add_argument(
        "--questions", default="",
        help="Canonical question JSONL used to hydrate evaluation-only Gold metadata.",
    )
    parser.add_argument(
        "--locomo-data", default="",
        help=(
            "Original LoCoMo JSON used only inside Judge to hydrate answer, category, "
            "and evidence by sample_id; these fields never feed the memory runtime."
        ),
    )
    parser.add_argument(
        "--require-gold-answer", action="store_true",
        help="Fail before judging when any row has an empty Gold answer.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--judge_system", required=True)
    parser.add_argument(
        "--judge-system-field", default="",
        help=("Optional input field containing the per-row arm/system name. "
              "The official single-answer prompt is unchanged."),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_ids", default="", help="Comma-separated sample_id allowlist.")
    parser.add_argument("--sample_ids_file", default="", help="Newline-separated sample_id allowlist.")
    parser.add_argument(
        "--exclude_cat5_input",
        action="store_true",
        help="Do not submit Cat5 rows to the Judge; intended for the official Cat1-4 scope.",
    )
    parser.add_argument("--stop_on_failure", action="store_true")
    parser.add_argument(
        "--exclude_cat5_metrics",
        action="store_true",
        help="Keep Cat5 judgments, but exclude Cat5 rows from generated metric summaries.",
    )
    return parser


def load_locomo_questions(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LoCoMo data must be a JSON array")
    questions: dict[str, dict[str, Any]] = {}
    for conversation in payload:
        conversation_id = str(conversation.get("sample_id") or "")
        qa_rows = conversation.get("qa") or []
        if not conversation_id or not isinstance(qa_rows, list):
            continue
        for index, qa in enumerate(qa_rows):
            sample_id = f"{conversation_id}__qa_{index}"
            if sample_id in questions:
                raise ValueError(f"duplicate LoCoMo question identity: {sample_id}")
            questions[sample_id] = {
                "sample_id": sample_id,
                "question": qa.get("question"),
                "question_type": qa.get("category"),
                "gold_answer": qa.get("answer"),
                "gold_memory_ids": qa.get("evidence") or [],
                "reference_source": "judge_only_original_locomo",
            }
    return questions


def judge_one(row: dict[str, Any], *, judge_system: str, model: str, args: argparse.Namespace) -> dict[str, Any]:
    question = str(row.get("question") or "")
    golden_answer = str(row.get("gold_answer") if row.get("gold_answer") is not None else "")
    generated_answer = str(row.get("pred_answer") or row.get("evaluation_pred_answer") or "")
    raw = complete_chat(
        model=model,
        prompt=JUDGE_PROMPT.format(
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer,
        ),
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
    )
    _persist_raw_judge_response(
        judge_system=judge_system, sample_id=str(row.get("sample_id") or ""),
        question=question, golden_answer=golden_answer,
        generated_answer=generated_answer, raw_response=raw, model=model,
    )
    label = parse_label(raw)
    return {
        "judge_system": judge_system,
        "sample_id": str(row.get("sample_id") or ""),
        "system": str(row.get("system") or ""),
        "question_type": str(row.get("question_type") or row.get("original_category") or ""),
        "question": question,
        "golden_answer": golden_answer,
        "generated_answer": generated_answer,
        "local_answer_correct": bool(row.get("answer_correct")),
        "retrieval_hit": bool(row.get("retrieval_hit")),
        "prompt_hit": bool(row.get("prompt_hit")),
        "gold_in_prompt": bool(row.get("gold_in_prompt")),
        "grade": label == "CORRECT",
        "judge_label": label,
        "raw_judge_response": raw,
        "judge_model": model,
    }


def complete_chat(*, model: str, prompt: str, timeout: float, max_retries: int, max_tokens: int) -> str:
    base_url = (os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
    api_key, _ = external_credential(base_url)
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    last_error = ""
    for attempt in range(1, max_retries + 1):
        call_id = _aet_http_pending(route="judge/chat/completions", payload=payload, attempt=attempt)
        try:
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            text = str(parsed["choices"][0]["message"].get("content") or "")
            _aet_http_terminal(call_id, route="judge/chat/completions", attempt=attempt, status="ok", response_sha256=_sha(parsed))
            return text
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code}: {body}"
            _aet_http_terminal(call_id, route="judge/chat/completions", attempt=attempt, status="failed", error=last_error)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            _aet_http_terminal(call_id, route="judge/chat/completions", attempt=attempt, status="failed", error=last_error)
            if os.getenv("SCHEME_A_ET_DISABLE_CURL_FALLBACK", "").strip().lower() in {"1","true","yes","on"}:
                if attempt < max_retries:
                    time.sleep(min(120.0, 2.0 * attempt))
                continue
            # Managed runners may deny Python sockets while permitting the
            # approved curl transport used by the answerer.
            try:
                curl_call_id = _aet_http_pending(route="judge/curl/chat/completions", payload=payload, attempt=attempt)
                raw = subprocess.check_output([
                    "curl", "-sS", "--max-time", str(int(timeout)), "-X", "POST", url,
                    "-H", "Content-Type: application/json", "-H", f"Authorization: Bearer {api_key}",
                    "--data-binary", data.decode("utf-8"),
                ], text=True, timeout=timeout + 10)
                parsed = json.loads(raw)
                text = str(parsed["choices"][0]["message"].get("content") or "")
                _aet_http_terminal(curl_call_id, route="judge/curl/chat/completions", attempt=attempt, status="ok", response_sha256=_sha(parsed))
                return text
            except Exception as curl_exc:
                last_error = f"{last_error}; curl={type(curl_exc).__name__}: {curl_exc}"
                _aet_http_terminal(curl_call_id if "curl_call_id" in locals() else None, route="judge/curl/chat/completions", attempt=attempt, status="failed", error=last_error)
        if attempt < max_retries:
            time.sleep(min(120.0, 2.0 * attempt))
    raise RuntimeError(f"judge request failed after retries: {last_error}")


_RAW_JUDGE_LOCK = Lock()
def _persist_raw_judge_response(*, judge_system:str, sample_id:str, question:str, golden_answer:str, generated_answer:str, raw_response:str, model:str)->None:
    path=os.getenv("SCHEME_A_ET_JUDGE_RAW_JOURNAL","").strip()
    if not path:return
    row={"schema_version":"aet-raw-judge-response-v1","judge_system":judge_system,"sample_id":sample_id,"model":model,"question_sha256":hashlib.sha256(question.encode()).hexdigest(),"gold_sha256":hashlib.sha256(golden_answer.encode()).hexdigest(),"generated_answer_sha256":hashlib.sha256(generated_answer.encode()).hexdigest(),"raw_response":raw_response,"raw_response_sha256":hashlib.sha256(raw_response.encode()).hexdigest()}
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    with _RAW_JUDGE_LOCK,target.open("a",encoding="utf-8") as h:h.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n");h.flush();os.fsync(h.fileno())


def parse_label(raw: str) -> str:
    text = str(raw or "").strip()
    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = None
    if isinstance(parsed, dict):
        label = str(parsed.get("label") or "").strip().upper()
        return "CORRECT" if label == "CORRECT" else "WRONG"
    upper = text.upper()
    return "CORRECT" if "CORRECT" in upper and "WRONG" not in upper and "INCORRECT" not in upper else "WRONG"


def load_yaml(path: str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def apply_env(env: dict[str, Any]) -> None:
    if any(str(env.get(key) or "").strip() for key in (
        "API_KEY", "GENERATION_API_KEY", "OPENAI_API_KEY", "MEMORY_BUILD_GENERATION_API_KEY", "MEMORY_BUILD_OPENAI_API_KEY"
    )):
        raise ValueError("Judge config must not contain credentials")
    for key, value in env.items():
        text = str(value).strip()
        if text and "PLACEHOLDER" not in text:
            os.environ[str(key)] = text


def load_rows(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_key(judge_system: str, row: dict[str, Any]) -> tuple[str, str]:
    return judge_system, str(row.get("sample_id") or "")


def normalized_answer(value: Any) -> str:
    text = str(value or "").casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def load_completed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[(str(row.get("judge_system") or ""), str(row.get("sample_id") or ""))] = row
    return out


def failure_row(row: dict[str, Any], *, judge_system: str, model: str, error: str) -> dict[str, Any]:
    return {
        "judge_system": judge_system,
        "sample_id": str(row.get("sample_id") or ""),
        "system": str(row.get("system") or ""),
        "question_type": str(row.get("question_type") or row.get("original_category") or ""),
        "grade": False,
        "judge_label": "ERROR",
        "judge_model": model,
        "error": error,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (str(item.get("judge_system") or ""), str(item.get("sample_id") or ""))):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_summary(rows: list[dict[str, Any]], path: Path, *, key_fields: list[str]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        groups.setdefault(key, []).append(row)
    fields = key_fields + ["n", "correct", "accuracy"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, items in sorted(groups.items(), key=lambda item: item[0]):
            n = len(items)
            correct = sum(1 for row in items if row.get("grade") is True)
            writer.writerow({
                **{field: value for field, value in zip(key_fields, key)},
                "n": n,
                "correct": correct,
                "accuracy": correct / n if n else 0.0,
            })


def write_scope_summary(rows: list[dict[str, Any]], path: Path, *, include_cat5_scope: bool = True) -> None:
    by_system: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_system.setdefault(str(row.get("judge_system") or ""), []).append(row)
    fields = ["judge_system", "scope", "n", "correct", "accuracy"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for system, items in sorted(by_system.items()):
            scopes = {
                "all": items,
                "cat1-4_exclude_cat5": [row for row in items if str(row.get("question_type")) != "5"],
            }
            if include_cat5_scope:
                scopes["cat5_only"] = [row for row in items if str(row.get("question_type")) == "5"]
            for scope, scoped in scopes.items():
                n = len(scoped)
                correct = sum(1 for row in scoped if row.get("grade") is True)
                writer.writerow({
                    "judge_system": system,
                    "scope": scope,
                    "n": n,
                    "correct": correct,
                    "accuracy": correct / n if n else 0.0,
                })


if __name__ == "__main__":
    main()
