#!/usr/bin/env python3
"""Compute reproducible lexical metrics for authoritative LoCoMo answers."""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


BASE_SYSTEMS = (
    "full_context_legacy",
    "memgpt",
    "letta",
    "readagent",
    "current_memory",
    "simple_vector_tfidf",
    "langmem",
    "no_memory",
    "memorybank",
)

STEMMER = PorterStemmer()
def normalize_locomo(value: object) -> str:
    text = str(value or "").replace(",", "").lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def counter_f1(prediction_tokens: list[str], reference_tokens: list[str]) -> float:
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if not common:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def locomo_pair_f1(prediction: str, reference: str) -> float:
    pred = [STEMMER.stem(token) for token in normalize_locomo(prediction).split()]
    ref = [STEMMER.stem(token) for token in normalize_locomo(reference).split()]
    return counter_f1(pred, ref)


def locomo_f1(prediction: str, reference: str, category: str) -> float:
    if category == "3":
        reference = reference.split(";", 1)[0].strip()
    if category != "1":
        return locomo_pair_f1(prediction, reference)
    predictions = [part.strip() for part in prediction.split(",")]
    references = [part.strip() for part in reference.split(",")]
    if not references:
        return 0.0
    return sum(max(locomo_pair_f1(pred, ref) for pred in predictions) for ref in references) / len(references)


def mem0_tokens(value: object) -> list[str]:
    return str(value or "").lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def mem0_f1(prediction: str, reference: str) -> float:
    # This intentionally mirrors Mem0's set-based metrics/utils.py implementation.
    pred = set(mem0_tokens(prediction))
    ref = set(mem0_tokens(reference))
    if not pred or not ref:
        return 0.0
    common = len(pred & ref)
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def mem0_bleu1(prediction: str, reference: str) -> float:
    # Match Mem0's nltk.word_tokenize contract. NLTK's punkt data is required.
    pred = word_tokenize(prediction.lower())
    ref = word_tokenize(reference.lower())
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    # NLTK SmoothingFunction.method1 adds epsilon=0.1 when modified precision is zero.
    precision = (overlap if overlap else 0.1) / len(pred)
    brevity_penalty = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / len(pred))
    return brevity_penalty * precision


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_rows(data_dir: Path) -> list[dict]:
    path = data_dir / "per_sample_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing per-sample answers: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if str(row.get("question_type", row.get("category", ""))) in {"1", "2", "3", "4"}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("locomo_20260814/systems_authoritative"))
    parser.add_argument("--output", type=Path, default=Path("locomo_20260814/lexical_metrics_20260817"))
    parser.add_argument("--systems", nargs="*", default=list(BASE_SYSTEMS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summaries = []
    for system in args.systems:
        rows = load_rows(args.root / system / "data")
        scored = []
        seen = set()
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id for {system}: {sample_id}")
            seen.add(sample_id)
            category = str(row.get("question_type", row.get("category")))
            prediction = str(row.get("pred_answer", row.get("response", "")))
            reference = str(row.get("gold_answer", row.get("answer", "")))
            scored.append({
                "sample_id": sample_id,
                "category": category,
                "locomo_official_f1": locomo_f1(prediction, reference, category),
                "mem0_f1": mem0_f1(prediction, reference),
                "mem0_bleu1": mem0_bleu1(prediction, reference),
            })

        groups = {"overall": scored}
        groups.update({f"category_{cat}": [row for row in scored if row["category"] == cat] for cat in "1234"})
        aggregate = {}
        for group, items in groups.items():
            aggregate[group] = {
                "count": len(items),
                "locomo_official_f1": mean([item["locomo_official_f1"] for item in items]),
                "mem0_f1": mean([item["mem0_f1"] for item in items]),
                "mem0_bleu1": mean([item["mem0_bleu1"] for item in items]),
            }
        summaries.append({"system": system, "source": str((args.root / system / "data").resolve()), "metrics": aggregate})
        with (args.output / f"{system}.jsonl").open("w", encoding="utf-8") as handle:
            for item in scored:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    payload = {
        "schema_version": "locomo-authoritative-lexical-metrics-v1",
        "scope": "LoCoMo categories 1-4 only",
        "contracts": {
            "locomo_official_f1": "Original LoCoMo task_eval/evaluation.py; Porter stemming and category-1 multi-answer handling",
            "mem0_f1": "Mem0 evaluation/metrics/utils.py set-based token F1",
            "mem0_bleu1": "Mem0 evaluation/metrics/utils.py sentence BLEU unigram with brevity penalty",
        },
        "systems": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Authoritative lexical metrics (LoCoMo Cat1-4)", "",
        "All values are percentages. These lexical metrics are deterministic and do not replace the official LoCoMo LLM Judge.", "",
        "| System | N | LoCoMo official F1 | Mem0 F1 | Mem0 BLEU-1 |", "|---|---:|---:|---:|---:|",
    ]
    for summary in sorted(summaries, key=lambda item: item["metrics"]["overall"]["locomo_official_f1"], reverse=True):
        overall = summary["metrics"]["overall"]
        lines.append(f"| {summary['system']} | {overall['count']} | {overall['locomo_official_f1']*100:.2f} | {overall['mem0_f1']*100:.2f} | {overall['mem0_bleu1']*100:.2f} |")
    lines.extend(["", "`LoCoMo official F1` is the correct column for the original LoCoMo paper. `Mem0 F1` and `Mem0 BLEU-1` are the correct columns for the Mem0 paper contract.", ""])
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
