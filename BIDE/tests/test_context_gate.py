from best_memory.memory.context_gate import contextual_status
from best_memory.memory.run_generation import _repair_memory_shape


def source(text):
    return {"center": {"text": text}}


def memory(*, status="ASSERTED", subject="Alice"):
    return {
        "l1": {"clause_units": [{"units": [{"status": status}]}]},
        "l2_direct": [{"subject": subject}] if subject is not None else [],
    }


def test_unresolved_memory_requires_context():
    assert contextual_status(memory(status="UNRESOLVED"), source("The appointment is Tuesday.")) == "CONTEXT_REPARSE_REQUIRED"


def test_short_confirmation_requires_context():
    assert contextual_status(memory(subject=None), source("Yes, exactly.")) == "CONTEXT_REPARSE_REQUIRED"


def test_short_deictic_without_concrete_subject_requires_context():
    assert contextual_status(memory(subject="it"), source("It happened there after that.")) == "CONTEXT_REPARSE_REQUIRED"


def test_self_contained_statement_reuses_base_memory():
    text = "Alice graduated from Stanford University with a degree in economics."
    assert contextual_status(memory(subject="Alice"), source(text)) == "SELF_CONTAINED_LLM"


def test_nonpropositional_request_does_not_require_context():
    value = {
        "l1": {"clause_units": [{"units": [
            {"status": "NON_PROPOSITIONAL", "assertion_ids": []},
            {"status": "NON_PROPOSITIONAL", "assertion_ids": []},
        ]}]},
        "l2_direct": [],
    }
    text = "Tell me a story about how she became a great warrior."
    assert contextual_status(value, source(text)) == "SELF_CONTAINED_LLM"


def test_malformed_l1_is_relinked_from_valid_assertion_provenance():
    src = {"center": {"raw_id": "q::D1:1", "text": "Alice graduated.", "clauses": [{"clause_id": "c1", "text": "Alice graduated."}]}}
    value = {
        "center_raw_id": "q::D1:1",
        "l1": {"clause_units": [{"clause_id": "c1", "text": "Alice graduated."}], "mentions": []},
        "l2_direct": [{
            "assertion_id": "a1", "subject": "Alice", "relation_family": "OTHER",
            "surface_relation": "graduated", "answer_value": {"type": "OTHER", "value": "graduated"},
            "roles": {}, "scope": {}, "modality": "UNKNOWN", "polarity": "UNKNOWN",
            "retrieval_text": "Alice graduated", "provenance": {"raw_ids": ["q::D1:1"], "clause_ids": ["c1"]},
        }],
    }
    repaired = _repair_memory_shape(value, src)
    unit = repaired["l1"]["clause_units"][0]["units"][0]
    assert unit["status"] == "ASSERTED"
    assert unit["assertion_ids"] == ["a1"]
    assert unit["reason"] is None


def test_invalid_unit_status_with_assertion_link_is_asserted():
    src = {"center": {"raw_id": "q::D1:1", "clauses": [{"clause_id": "c1", "text": "Alice graduated."}]}}
    value = {
        "center_raw_id": "q::D1:1",
        "l1": {"clause_units": [{"clause_id": "c1", "units": [{
            "unit_id": "u1", "span": "Alice graduated.", "status": "DIRECT",
            "assertion_ids": ["a1"], "reason": None,
        }]}], "mentions": []},
        "l2_direct": [{
            "assertion_id": "a1", "subject": "Alice", "relation_family": "OTHER",
            "surface_relation": "graduated", "answer_value": {"type": "OTHER", "value": "graduated"},
            "roles": {}, "scope": {}, "modality": "UNKNOWN", "polarity": "UNKNOWN",
            "retrieval_text": "Alice graduated", "provenance": {"raw_ids": ["q::D1:1"], "clause_ids": ["c1"]},
        }],
    }
    repaired = _repair_memory_shape(value, src)
    assert repaired["l1"]["clause_units"][0]["units"][0]["status"] == "ASSERTED"
