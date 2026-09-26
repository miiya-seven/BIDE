"""Question-independent routing for the optional contextual memory pass."""
from __future__ import annotations

import re
from typing import Any


_RESPONSE_PREFIX = re.compile(
    r"^\s*(?:yes|yeah|yep|no|nope|correct|exactly|right|sure|indeed|"
    r"actually|instead|me too|same here|i agree|i disagree)\b",
    re.IGNORECASE,
)
_DEICTIC = re.compile(
    r"\b(?:this|that|these|those|it|they|them|he|she|him|her|former|latter|"
    r"there|then|the same|so do i|neither do i)\b",
    re.IGNORECASE,
)


def contextual_status(memory: dict[str, Any], source: dict[str, Any]) -> str:
    """Return the second-pass routing status without consulting a question.

    The gate is deliberately conservative: any unresolved L1 unit is routed,
    as are short response-like or deictic utterances whose extracted facts do
    not establish a concrete non-pronominal subject. Long self-contained
    statements remain on the validated base L1/L2 path.
    """
    units = [
        unit
        for clause in (memory.get("l1") or {}).get("clause_units", [])
        if isinstance(clause, dict)
        for unit in clause.get("units", [])
        if isinstance(unit, dict)
    ]
    if any(unit.get("status") == "UNRESOLVED" for unit in units):
        return "CONTEXT_REPARSE_REQUIRED"

    assertions = [x for x in memory.get("l2_direct", []) if isinstance(x, dict)]
    if units and all(unit.get("status") == "NON_PROPOSITIONAL" for unit in units) and not assertions:
        return "SELF_CONTAINED_LLM"

    center = source.get("center") or {}
    text = str(center.get("text") or "").strip()
    if not text:
        return "SELF_CONTAINED_LLM"

    subjects = [str(x.get("subject") or "").strip() for x in assertions]
    concrete_subject = any(
        subject and not re.fullmatch(r"(?:i|you|he|she|it|we|they|this|that|these|those)", subject, re.I)
        for subject in subjects
    )
    compact = len(text) <= 240
    if compact and _RESPONSE_PREFIX.search(text):
        return "CONTEXT_REPARSE_REQUIRED"
    if compact and _DEICTIC.search(text) and not concrete_subject:
        return "CONTEXT_REPARSE_REQUIRED"
    return "SELF_CONTAINED_LLM"
