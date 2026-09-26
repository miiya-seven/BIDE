import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_longmemeval_default_is_turn_and_keeps_context(tmp_path, monkeypatch):
    from best_memory.memory import prepare

    source = tmp_path / "input.json"
    source.write_text(json.dumps([{
        "question_id": "q1",
        "haystack_sessions": [[
            {"role": "user", "content": "Alice went home."},
            {"role": "assistant", "content": "She took the train."},
        ]],
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2020-01-01"],
        "question": "Where did Alice go?",
    }]), encoding="utf-8")
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "prepare.py", "--input", str(source), "--run-root", str(out),
        "--dataset", "longmemeval",
    ])
    prepare.main()

    receipt = json.loads((out / "PREPARATION_RECEIPT.json").read_text())
    assert receipt["granularity"] == "turn"
    rows = [json.loads(x) for x in (out / "memory" / "MEMORY_REQUESTS.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["center"]["source_session_id"] == "s1"
    assert rows[0]["context_only"][0]["raw_id"].endswith("D1:2")
    assert rows[1]["context_only"][0]["raw_id"].endswith("D1:1")


def test_memory_request_view_preserves_context_only_without_promoting_it():
    from best_memory.memory.run_generation import memory_request_view

    source = {
        "center": {
            "raw_id": "q1::D1:2",
            "speaker": "assistant",
            "timestamp": "2020-01-01",
            "source_session_id": "s1",
            "clauses": [{"clause_id": "c2", "text": "She took the train."}],
        },
        "context_only": [{
            "raw_id": "q1::D1:1",
            "speaker": "user",
            "text": "Alice went home.",
            "source_session_id": "s1",
            "context_only": True,
        }],
    }
    view = memory_request_view(source, "longmemeval")
    assert view["center"]["raw_id"] == "q1::D1:2"
    assert view["center"]["clauses"] == [{"clause_id": "c2", "text": "She took the train."}]
    assert view["context_only"][0]["raw_id"] == "q1::D1:1"
    assert "context_only" not in view["center"]
