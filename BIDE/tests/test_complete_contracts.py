import ast
from pathlib import Path


def _function(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf8"))
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"ALLOWED_ACTS":{"ASSERTION","CONFIRMATION","DENIAL","ANSWER","CORRECTION","QUESTION","REACTION","REQUEST","NON_PROPOSITIONAL"}, "ALLOWED_DEPS":{"NONE","COREFERENCE","ELLIPSIS","CONFIRMATION","DEICTIC","OTHER"}}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def test_contextual_v41_complete_contract_accepts_explicit_links():
    validate = _function("src/best_memory/memory/contextual.py", "validate")
    value = {
        "center_summary":"Jon confirms the fair was useful.",
        "context_dependency":"CONFIRMATION",
        "antecedent_raw_ids":["conv::D1:1"],
        "speech_acts":[{
            "act":"CONFIRMATION", "center_quote":"Yes, it was useful",
            "antecedent_raw_ids":["conv::D1:1"], "assertion":None,
            "resolved_proposition":{"subject":"the fair","relation":"was","value":"useful","event":"fair visit","time":"","modality":"actual","polarity":"positive","source_raw_ids":["conv::D1:1","conv::D1:2"]},
        }],
        "occurrence_links":[{"owner":"Jon","event":"fair visit","relation":"same event","antecedent_raw_ids":["conv::D1:1"]}],
    }
    assert validate(value, "conv::D1:2", ["conv::D1:1"]) == []


def test_contextual_v41_rejects_unanchored_relation():
    validate = _function("src/best_memory/memory/contextual.py", "validate")
    value = {"center_summary":"yes","context_dependency":"CONFIRMATION","antecedent_raw_ids":[],"speech_acts":[{"act":"CONFIRMATION","antecedent_raw_ids":[],"resolved_proposition":None}],"occurrence_links":[]}
    assert "ACT_0_MISSING_ANTECEDENT" in validate(value, "conv::D1:2", ["conv::D1:1"])


def test_contextual_v41_normalizer_drops_unanchored_occurrence():
    normalize = _function("src/best_memory/memory/contextual.py", "normalize")
    value = {"context_dependency":"QUESTION", "occurrence_links":[{"antecedent_raw_ids":[]},{"antecedent_raw_ids":["conv::D1:1"]}]}
    assert normalize(value, ["conv::D1:1"]) == {"context_dependency":"OTHER", "occurrence_links":[{"antecedent_raw_ids":["conv::D1:1"]}]}


def test_runtime_query_merge_drops_gold_fields():
    source = Path("src/best_memory/query/merge_queries.py").read_text(encoding="utf8")
    assert "'answer'" not in source
    assert "'evidence'" not in source
    assert "missing validated complete query plan" in source


def test_contextual_self_antecedent_is_not_a_neighbor_edge():
    normalize = _function("src/best_memory/memory/contextual.py", "normalize")
    value = {"context_dependency":"COREFERENCE", "antecedent_raw_ids":["center"],
             "speech_acts":[{"act":"ASSERTION","antecedent_raw_ids":["center","neighbor"],
                             "resolved_proposition":{"source_raw_ids":["center","neighbor"]}}],
             "occurrence_links":[]}
    result=normalize(value,["neighbor"],"center")
    assert result["antecedent_raw_ids"]==[]
    assert result["speech_acts"][0]["antecedent_raw_ids"]==["neighbor"]
    assert result["speech_acts"][0]["resolved_proposition"]["source_raw_ids"]==["center","neighbor"]


def test_contextual_manifest_is_not_silently_overruled(monkeypatch):
    import os
    needed=_function("src/best_memory/memory/contextual.py","context_needed")
    needed.__globals__.update(FULL_CONTEXT=False,os=os)
    monkeypatch.delenv('CONTEXTUAL_USE_LENGTH_GATE',raising=False)
    assert needed({'raw_id':'manifest_required_turn'}) is True
