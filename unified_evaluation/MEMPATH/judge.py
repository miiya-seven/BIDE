from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from map_platform.datasets.schema import Fact
from map_platform.llm.gateway import LLMGateway, embed_texts
from map_platform.memory_systems.base import MemoryEntry
from map_platform.utils.hashing import sha256_json


def _normalize(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _tokenize(text: str) -> list[str]:
    return [token for token in _normalize(text).split(" ") if token]


@dataclass
class CachedJudgeResult:
    key: str
    value: bool
    metadata: dict[str, Any]


class JudgeCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judge_cache (
                    cache_key TEXT PRIMARY KEY,
                    bool_value INTEGER NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS judge_cache (
                    cache_key TEXT PRIMARY KEY,
                    bool_value INTEGER NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            self.conn.commit()

    def get(self, cache_key: str) -> CachedJudgeResult | None:
        row = self.conn.execute(
            "SELECT cache_key, bool_value, metadata FROM judge_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return CachedJudgeResult(key=row[0], value=bool(row[1]), metadata=json.loads(row[2]))

    def set(self, cache_key: str, value: bool, metadata: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO judge_cache(cache_key, bool_value, metadata)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                bool_value = excluded.bool_value,
                metadata = excluded.metadata
            """,
            (cache_key, int(value), json.dumps(metadata, ensure_ascii=False)),
        )
        self.conn.commit()


class Judge:
    def __init__(
        self,
        *,
        judge_model: str = "Qwen/Qwen3-8B",
        embedding_model: str = "BAAI/bge-m3",
        llm_provider: str | None = None,
        cache_path: str | Path = ".cache/judge_cache.sqlite",
        enable_llm_judge: bool = True,
    ) -> None:
        self.judge_model = judge_model
        self.embedding_model = embedding_model
        self.llm_provider = (llm_provider or "").lower() or None
        self.cache = JudgeCache(cache_path)
        self.llm = LLMGateway(provider=llm_provider)
        self.enable_llm_judge = enable_llm_judge
        self._embedding_cache: dict[str, list[float]] = {}

    def evidence_in_text(self, evidence_unit: str, text: str) -> bool:
        normalized_unit = _normalize(evidence_unit)
        normalized_text = _normalize(text)
        if not normalized_unit or not normalized_text:
            return False
        if normalized_unit in normalized_text:
            return True
        if SequenceMatcher(None, normalized_unit, normalized_text).ratio() >= 0.92:
            return True
        unit_tokens = set(_tokenize(evidence_unit))
        text_tokens = set(_tokenize(text))
        if unit_tokens and len(unit_tokens & text_tokens) / len(unit_tokens) >= 0.8:
            return True
        return self._llm_bool(
            task="evidence_in_text",
            payload={"evidence_unit": evidence_unit, "text": text},
            prompt=(
                "You are a strict evaluation judge.\n"
                f"Evidence unit: {evidence_unit}\n"
                f"Text: {text}\n"
                "Does the text semantically contain the evidence unit?\n"
                "Output PASS or FAIL only."
            ),
        )

    def fact_in_text(self, fact: Fact, text: str) -> bool:
        candidates = [
            fact.value,
            f"{fact.subject} {fact.attribute} {fact.value}",
            f"{fact.subject}'s {fact.attribute} is {fact.value}",
            f"{fact.attribute}: {fact.value}",
        ]
        return any(self.evidence_in_text(candidate, text) for candidate in candidates if candidate)

    def fact_in_entries(self, fact: Fact, entries: list[MemoryEntry]) -> bool:
        source_turn_ids = {str(item) for item in fact.source.get("turn_ids", [])}
        if source_turn_ids:
            targeted = [entry for entry in entries if source_turn_ids & {str(source_id) for source_id in entry.source_ids}]
            if targeted:
                return any(self.fact_in_text(fact, entry.content) for entry in targeted)
        return any(self.fact_in_text(fact, entry.content) for entry in entries)

    def answer_correct(self, question: str, gold_answer: Any, pred_answer: str, criterion: dict[str, Any]) -> bool:
        golds: list[str] = []
        if isinstance(gold_answer, list):
            golds.extend(str(item) for item in gold_answer if str(item).strip())
        elif str(gold_answer or "").strip():
            golds.append(str(gold_answer))
        golds.extend(str(item) for item in criterion.get("acceptable_answers", []) if str(item).strip())
        normalized_pred = _normalize(pred_answer)
        for gold in golds:
            normalized_gold = _normalize(gold)
            if normalized_gold and normalized_gold == normalized_pred:
                return True
            if normalized_gold and normalized_gold in normalized_pred:
                return True
            if normalized_gold and SequenceMatcher(None, normalized_gold, normalized_pred).ratio() >= 0.9:
                return True
        return self._llm_bool(
            task="answer_correct",
            payload={"question": question, "gold_answer": gold_answer, "pred_answer": pred_answer, "criterion": criterion},
            prompt=(
                "You are a strict evaluation judge.\n"
                f"Question: {question}\n"
                f"Reference answer(s): {golds}\n"
                f"Predicted answer: {pred_answer}\n"
                "Is the prediction correct?\n"
                "Output PASS or FAIL only."
            ),
        )

    def forbidden_fact_used(self, forbidden_facts: list[Fact], answer: str) -> bool:
        return any(self.fact_in_text(fact, answer) for fact in forbidden_facts)

    def answer_changed_correctly(self, original: str, counterfactual: str, expected: str) -> bool:
        if _normalize(counterfactual) == _normalize(expected):
            return True
        if _normalize(original) == _normalize(counterfactual):
            return False
        return self._llm_bool(
            task="answer_changed_correctly",
            payload={"original": original, "counterfactual": counterfactual, "expected": expected},
            prompt=(
                "You are a strict evaluation judge.\n"
                f"Original answer: {original}\n"
                f"Counterfactual answer: {counterfactual}\n"
                f"Expected answer: {expected}\n"
                "Did the answer change to the expected one?\n"
                "Output PASS or FAIL only."
            ),
        )

    def text_similarity(self, left: str, right: str) -> float:
        normalized_left = _normalize(left)
        normalized_right = _normalize(right)
        if not normalized_left or not normalized_right:
            return 0.0
        if normalized_left == normalized_right:
            return 1.0
        embedded = self._embedding_similarity(left, right)
        if embedded is not None:
            return embedded
        token_left = set(_tokenize(left))
        token_right = set(_tokenize(right))
        lexical = (len(token_left & token_right) / len(token_left | token_right)) if (token_left and token_right) else 0.0
        fuzzy = SequenceMatcher(None, normalized_left, normalized_right).ratio()
        return max(lexical, fuzzy)

    def text_contradicts_answer(self, question: str, answer: str, text: str) -> bool:
        normalized_answer = _normalize(answer)
        normalized_text = _normalize(text)
        if not normalized_answer or not normalized_text:
            return False
        if normalized_answer in normalized_text:
            return False
        similarity = self.text_similarity(answer, text)
        if similarity <= 0.3 or similarity >= 0.85:
            return False
        return self._llm_bool(
            task="text_contradicts_answer",
            payload={"question": question, "answer": answer, "text": text},
            prompt=(
                f'Question: "{question}"\n'
                f'Correct answer: "{answer}"\n'
                f'Memory entry: "{text}"\n\n'
                "Does this text make a claim that DIRECTLY CONTRADICTS the answer?\n"
                "YES: the text states something factually incompatible with the answer\n"
                "NO: the text is consistent with, neutral to, or unrelated to the answer\n\n"
                "Output YES or NO only."
            ),
            pass_tokens=("YES",),
        )

    def find_contradicting_entries(
        self,
        question: str,
        answer: str,
        entries: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        candidates: list[MemoryEntry] = []
        for entry in entries:
            similarity = self.text_similarity(answer, entry.content)
            if 0.3 < similarity < 0.85:
                candidates.append(entry)
        return [entry for entry in candidates if self.text_contradicts_answer(question, answer, entry.content)]

    def _embedding_similarity(self, left: str, right: str) -> float | None:
        left_emb = self._embed_text(left)
        right_emb = self._embed_text(right)
        if left_emb is None or right_emb is None:
            return None
        numerator = sum(a * b for a, b in zip(left_emb, right_emb))
        left_norm = math.sqrt(sum(a * a for a in left_emb))
        right_norm = math.sqrt(sum(b * b for b in right_emb))
        if left_norm == 0.0 or right_norm == 0.0:
            return None
        return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))

    def _embed_text(self, text: str) -> list[float] | None:
        if self.llm_provider != "openai":
            return None
        if not self.embedding_model or not self._embedding_api_key():
            return None
        cache_key = sha256_json({"text": text, "model": self.embedding_model})
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        try:
            vector = embed_texts([text], model=self.embedding_model)[0]
            self._embedding_cache[cache_key] = vector
            return vector
        except Exception:
            return None

    def _embedding_api_key(self) -> str | None:
        import os

        return os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or None

    def _llm_bool(
        self,
        *,
        task: str,
        payload: dict[str, Any],
        prompt: str,
        pass_tokens: tuple[str, ...] = ("PASS",),
    ) -> bool:
        if not self.enable_llm_judge or self.llm_provider in {None, "mock"}:
            return False
        cache_key = sha256_json({"task": task, "payload": payload, "model": self.judge_model})
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached.value
        response = self.llm.complete_text(prompt=prompt, model=self.judge_model, temperature=0.0)
        normalized = response.text.strip().upper()
        value = any(normalized.startswith(token) for token in pass_tokens)
        self.cache.set(cache_key, value, {"response": response.to_dict()})
        return value
