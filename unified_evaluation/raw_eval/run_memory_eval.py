from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import yaml

from pathlib import Path

try:
    from _bootstrap import bootstrap_project
except ImportError:
    def bootstrap_project():
        root = Path(__file__).resolve().parents[1]
        if str(root) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(root))
        return root

bootstrap_project()

from map_platform.datasets import UnifiedSample, load_dataset
from map_platform.memory_systems.base import MemoryEntry, PromptRecord, RetrievalResult
from map_platform.memory_systems.registry import build_memory_adapter
from map_platform.utils.serialization import to_jsonable


ERROR_PRED_ANSWER = "[SYSTEM_ERROR]"


SYSTEM_PROFILES: dict[str, dict[str, Any]] = {
    "no_memory": {
        "memory_mode": "question_only",
        "evaluation_role": "lower_bound_and_contamination_probe",
        "retrieval_policy": "none",
        "injection_policy": "none",
        "primary_metrics": ["answer_accuracy"],
        "diagnostic_focus": "whether the model can answer without memory evidence",
    },
    "full_context": {
        "memory_mode": "full_history_upper_bound",
        "evaluation_role": "full_history_context_upper_bound",
        "retrieval_policy": "all_history",
        "injection_policy": "full_history_prompt",
        "primary_metrics": ["answer_accuracy", "gold_context_coverage"],
        "diagnostic_focus": "whether the generation model can use complete history when evidence is present",
    },
    "simple_vector": {
        "memory_mode": "flat_turn_vector_retrieval",
        "evaluation_role": "local_sparse_retrieval_baseline",
        "retrieval_policy": "tfidf_top_k_turns",
        "injection_policy": "retrieved_turns_prompt",
        "primary_metrics": ["retrieval_recall", "prompt_gold_coverage", "answer_accuracy"],
        "diagnostic_focus": "lexical retrieval over raw dialogue turns",
    },
    "current_memory": {
        "memory_mode": "locomo_observation_or_raw_turn_retrieval",
        "evaluation_role": "dataset_current_memory_baseline",
        "retrieval_policy": "tfidf_top_k_observations_or_turns",
        "injection_policy": "retrieved_memories_prompt",
        "primary_metrics": ["retrieval_recall", "prompt_gold_coverage", "answer_accuracy"],
        "diagnostic_focus": "whether LoCoMo observation memories preserve answer evidence",
    },
    "mem0": {
        "memory_mode": "extracted_semantic_vector_memory",
        "evaluation_role": "sdk_memory_extraction_and_retrieval_system",
        "retrieval_policy": "sdk_semantic_search",
        "injection_policy": "retrieved_memories_prompt",
        "primary_metrics": ["memory_extraction_traceability", "retrieval_recall", "answer_accuracy"],
        "diagnostic_focus": "LLM-extracted memories, vector retrieval, and source provenance",
    },
    "memorybank": {
        "memory_mode": "time_bucket_summary_plus_detail_memory",
        "evaluation_role": "hierarchical_long_term_memory_system",
        "retrieval_policy": "semantic_similarity_plus_retention_weight",
        "injection_policy": "summaries_then_detailed_memories",
        "primary_metrics": ["summary_vs_detail_usage", "retrieval_recall", "answer_accuracy"],
        "diagnostic_focus": "time grouping, summarization, retention weights, and detail recovery",
    },
    "readagent": {
        "memory_mode": "gist_memory_with_page_lookup",
        "evaluation_role": "read_then_lookup_agentic_memory_system",
        "retrieval_policy": "gist_selection_then_page_expansion",
        "injection_policy": "gist_plus_expanded_pages_prompt",
        "primary_metrics": ["page_selection_recall", "prompt_gold_coverage", "answer_accuracy"],
        "diagnostic_focus": "whether gist lookup selects pages containing gold evidence",
    },
    "langmem": {
        "memory_mode": "semantic_and_procedural_memory_store",
        "evaluation_role": "sdk_semantic_procedural_memory_system",
        "retrieval_policy": "semantic_store_search_plus_procedural_guidance",
        "injection_policy": "procedural_system_prompt_plus_retrieved_semantic_memories",
        "primary_metrics": ["semantic_retrieval_recall", "procedural_injection_trace", "answer_accuracy"],
        "diagnostic_focus": "separation of semantic facts and procedural guidance",
    },
    "letta": {
        "memory_mode": "agent_core_memory_plus_archival_memory",
        "evaluation_role": "stateful_agent_memory_system",
        "retrieval_policy": "archival_passage_search",
        "injection_policy": "core_memory_plus_archival_retrieval",
        "primary_metrics": ["archival_retrieval_recall", "core_memory_trace", "answer_accuracy"],
        "diagnostic_focus": "agent core memory, archival passages, and stateful answering",
    },
    "memgpt": {
        "memory_mode": "agent_core_memory_plus_archival_memory",
        "evaluation_role": "memgpt_compatible_agent_memory_system",
        "retrieval_policy": "archival_passage_search",
        "injection_policy": "core_memory_plus_archival_retrieval",
        "primary_metrics": ["archival_retrieval_recall", "core_memory_trace", "answer_accuracy"],
        "diagnostic_focus": "MemGPT/Letta-style core and archival memory behavior",
    },
    "external": {
        "memory_mode": "external_adapter",
        "evaluation_role": "user_supplied_memory_system",
        "retrieval_policy": "adapter_defined",
        "injection_policy": "retrieved_memories_prompt",
        "primary_metrics": ["retrieval_recall", "answer_accuracy"],
        "diagnostic_focus": "external adapter behavior",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate memory systems on raw LoCoMo / LongMemEval evidence ids.")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config. CLI args override config values.")
    parser.add_argument("--dataset", required=False, choices=["locomo", "longmemeval"])
    parser.add_argument("--data_path", required=False)
    parser.add_argument("--systems", required=False, help="Comma-separated systems.")
    parser.add_argument(
        "--conversation_ids",
        default=None,
        help="Comma-separated conversation ids or ranges, e.g. conv-47,conv-49 or conv-47-50.",
    )
    parser.add_argument(
        "--sample_ids",
        default=None,
        help="Comma-separated exact sample ids, e.g. conv-30__qa_87.",
    )
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", required=False)
    parser.add_argument("--generation_model", default=None)
    parser.add_argument("--embedding_model", default=None)
    parser.add_argument("--llm_provider", default=None)
    parser.add_argument("--openai_api_key", default=None, help="Optional; prefer env var or config env block.")
    parser.add_argument("--openai_base_url", default=None, help="Optional OpenAI-compatible API base URL.")
    parser.add_argument("--memory_build_openai_api_key", default=None, help="Optional external API key for memory extraction/build.")
    parser.add_argument("--memory_build_openai_base_url", default=None, help="Optional external OpenAI-compatible base URL for memory extraction/build.")
    parser.add_argument("--memory_build_model", default=None, help="Optional model used only for memory extraction/build.")
    parser.add_argument("--letta_base_url", default=None)
    parser.add_argument("--letta_api_key", default=None)
    parser.add_argument("--letta_model", default=None, help="Optional Letta model handle, e.g. openai/gpt-4o-mini.")
    parser.add_argument("--letta_embedding_model", default=None, help="Optional Letta embedding handle.")
    parser.add_argument("--letta_model_provider", default=None, help="Optional provider prefix for Letta model handles.")
    parser.add_argument("--skip_unavailable", default=None, help="Skip systems that fail validate_setup. Default: true.")
    parser.add_argument(
        "--reuse_build_by_conversation",
        default=None,
        help="Build memory once per conversation/system, then evaluate all QA samples for that conversation.",
    )
    parser.add_argument(
        "--print_qa_results",
        default=None,
        help="Print per-QA question/gold/prediction lines while running. Default: true.",
    )
    parser.add_argument("--resume", default=None, help="Resume from an existing output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    resolved = resolve_config(config, args)
    apply_env(resolved.get("env", {}))

    output_dir = Path(resolved.get("resume") or resolved["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MEMORY_EVAL_OUTPUT_DIR"] = str(output_dir)
    samples = load_dataset(resolved["dataset"], resolved["data_path"], max_samples=resolved.get("max_samples"))
    samples = filter_samples_by_conversation(samples, resolved.get("conversation_ids"))
    samples = filter_samples_by_sample_id(samples, resolved.get("sample_ids"))
    if not samples:
        raise ValueError("No samples remain after applying --conversation_ids/--sample_ids filters.")
    preflight_rows = run_system_preflight(
        systems=resolved["systems"],
        generation_model=resolved["generation_model"],
        embedding_model=resolved["embedding_model"],
        llm_provider=resolved.get("llm_provider"),
    )
    write_csv(output_dir / "system_preflight.csv", preflight_rows)
    systems = runnable_systems(preflight_rows, skip_unavailable=bool(resolved.get("skip_unavailable", True)))
    if not systems:
        messages = "; ".join(
            f"{row.get('system')}: {row.get('message') or 'unavailable'}"
            for row in preflight_rows
        )
        raise RuntimeError(f"No runnable systems after preflight. {messages}")

    result_path = output_dir / "per_sample_results.jsonl"
    prediction_path = output_dir / "predictions.jsonl"
    rows: list[dict[str, Any]] = load_jsonl(result_path)
    prediction_rows: list[dict[str, Any]] = load_jsonl(prediction_path)
    completed = {
        (str(row.get("sample_id")), str(row.get("system")))
        for row in rows
        if row.get("error") in (None, "", False) or row.get("skipped")
    }
    if bool(resolved.get("reuse_build_by_conversation")):
        for row in run_reuse_build_by_conversation(
            samples=samples,
            systems=systems,
            completed=completed,
            top_k=resolved["top_k"],
            generation_model=resolved["generation_model"],
            embedding_model=resolved["embedding_model"],
            llm_provider=resolved.get("llm_provider"),
            print_qa_results=bool(resolved.get("print_qa_results", True)),
        ):
            rows.append(row)
            append_jsonl(result_path, row)
            append_trace(output_dir, row["dataset"], row["system"], row)
            pred_row = prediction_row_by_sample_id(samples, row)
            prediction_rows.append(pred_row)
            append_jsonl(prediction_path, pred_row)
    else:
        for sample in samples:
            for system_name in systems:
                if (sample.sample_id, system_name) in completed:
                    continue
                row = run_one(
                    sample=sample,
                    system_name=system_name,
                    top_k=resolved["top_k"],
                    generation_model=resolved["generation_model"],
                    embedding_model=resolved["embedding_model"],
                    llm_provider=resolved.get("llm_provider"),
                )
                if bool(resolved.get("print_qa_results", True)):
                    print_qa_result(row)
                rows.append(row)
                append_jsonl(result_path, row)
                append_trace(output_dir, sample.dataset, system_name, row)
                pred_row = prediction_row(sample, row)
                prediction_rows.append(pred_row)
                append_jsonl(prediction_path, pred_row)

    rows = latest_rows_by_sample_system(rows)
    prediction_rows = latest_prediction_rows(prediction_rows)
    write_jsonl(result_path, rows)
    write_jsonl(prediction_path, prediction_rows)
    write_summary(output_dir / "summary.csv", rows, ["system"])
    write_summary(output_dir / "summary_by_type.csv", rows, ["system", "question_type"])
    write_trace_files(output_dir, rows)
    write_stage2_reports(output_dir, rows)
    if resolved["dataset"] == "longmemeval":
        write_longmemeval_predictions(output_dir, prediction_rows)


def load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def resolve_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    models = config.get("models") if isinstance(config.get("models"), dict) else {}
    env_config = config.get("env") if isinstance(config.get("env"), dict) else {}
    cli_env = {
        key: value
        for key, value in {
            "OPENAI_API_KEY": args.openai_api_key,
            "OPENAI_BASE_URL": args.openai_base_url,
            "MEMORY_BUILD_OPENAI_API_KEY": args.memory_build_openai_api_key,
            "MEMORY_BUILD_OPENAI_BASE_URL": args.memory_build_openai_base_url,
            "MEMORY_BUILD_MODEL": args.memory_build_model,
            "LETTA_BASE_URL": args.letta_base_url,
            "LETTA_API_KEY": args.letta_api_key,
            "LETTA_MODEL": args.letta_model,
            "LETTA_EMBEDDING_MODEL": args.letta_embedding_model,
            "LETTA_MODEL_PROVIDER": args.letta_model_provider,
        }.items()
        if value not in (None, "")
    }
    env = {
        key: value
        for key, value in {**env_config, **cli_env}.items()
        if value not in (None, "")
    }
    llm_provider = args.llm_provider or config.get("llm_provider")
    if not llm_provider:
        llm_provider = infer_llm_provider(
            cli_env=cli_env,
            env_config=env_config,
            existing_env=os.environ,
        )
    systems_value = args.systems if args.systems is not None else config.get("systems")
    systems = parse_systems(systems_value)
    resolved = {
        "dataset": args.dataset or config.get("dataset"),
        "data_path": args.data_path or config.get("data_path"),
        "systems": systems,
        "conversation_ids": args.conversation_ids or config.get("conversation_ids"),
        "sample_ids": args.sample_ids or config.get("sample_ids"),
        "top_k": args.top_k if args.top_k is not None else int(config.get("top_k", 5)),
        "max_samples": args.max_samples if args.max_samples is not None else config.get("max_samples"),
        "output_dir": args.output_dir or config.get("output_dir"),
        "generation_model": args.generation_model
        or os.getenv("LLM_MODEL")
        or os.getenv("MODEL_NAME")
        or os.getenv("OPENAI_MODEL")
        or models.get("generation_model")
        or config.get("generation_model")
        or "Qwen/Qwen3-8B",
        "embedding_model": args.embedding_model
        or os.getenv("EMBEDDING_MODEL")
        or os.getenv("EMBEDDING_MODEL_NAME")
        or models.get("embedding_model")
        or config.get("embedding_model")
        or "BAAI/bge-m3",
        "llm_provider": llm_provider,
        "env": env,
        "skip_unavailable": parse_bool(
            args.skip_unavailable if args.skip_unavailable is not None else config.get("skip_unavailable", True)
        ),
        "reuse_build_by_conversation": parse_bool(
            args.reuse_build_by_conversation
            if args.reuse_build_by_conversation is not None
            else config.get("reuse_build_by_conversation", False)
        ),
        "print_qa_results": parse_bool(
            args.print_qa_results if args.print_qa_results is not None else config.get("print_qa_results", True)
        ),
        "resume": args.resume or config.get("resume"),
    }
    missing = [key for key in ("dataset", "data_path", "systems", "output_dir") if not resolved.get(key)]
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}. Provide them via CLI or --config.")
    return resolved


def infer_llm_provider(
    *,
    cli_env: dict[str, Any],
    env_config: dict[str, Any],
    existing_env: Any,
) -> str | None:
    def has_openai_settings(source: dict[str, Any]) -> bool:
        return bool(
            str(source.get("OPENAI_API_KEY") or "").strip()
            or str(source.get("OPENAI_BASE_URL") or "").strip()
            or str(source.get("OPENAI_API_BASE") or "").strip()
            or str(source.get("OPENAI_API_BASE_URL") or "").strip()
        )

    if has_openai_settings(cli_env) or has_openai_settings(env_config):
        return "openai"
    if has_openai_settings(existing_env):
        return "openai"
    return None


def parse_systems(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def parse_conversation_ids(value: Any) -> set[str]:
    if value in (None, ""):
        return set()
    raw_items = value if isinstance(value, list) else str(value).split(",")
    conversation_ids: set[str] = set()
    for raw_item in raw_items:
        item = str(raw_item).strip()
        if not item:
            continue
        range_match = re.fullmatch(r"([A-Za-z_]*-?)(\d+)-(\d+)", item)
        if range_match:
            prefix, start_text, end_text = range_match.groups()
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            width = max(len(start_text), len(end_text))
            for number in range(start, end + step, step):
                conversation_ids.add(f"{prefix}{number:0{width}d}" if width > 1 else f"{prefix}{number}")
            continue
        conversation_ids.add(item)
    return conversation_ids


def filter_samples_by_conversation(samples: list[UnifiedSample], value: Any) -> list[UnifiedSample]:
    conversation_ids = parse_conversation_ids(value)
    if not conversation_ids:
        return samples
    return [sample for sample in samples if conversation_key(sample) in conversation_ids]


def parse_sample_ids(value: Any) -> set[str]:
    if value in (None, ""):
        return set()
    raw_items = value if isinstance(value, list) else str(value).split(",")
    return {str(item).strip() for item in raw_items if str(item).strip()}


def filter_samples_by_sample_id(samples: list[UnifiedSample], value: Any) -> list[UnifiedSample]:
    sample_ids = parse_sample_ids(value)
    if not sample_ids:
        return samples
    return [sample for sample in samples if sample.sample_id in sample_ids]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def run_system_preflight(
    *,
    systems: list[str],
    generation_model: str,
    embedding_model: str,
    llm_provider: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generation_error = generation_api_preflight_error(
        generation_model=generation_model,
        llm_provider=llm_provider,
    )
    for system_name in systems:
        adapter = None
        try:
            adapter = build_memory_adapter(
                system_name,
                generation_model=generation_model,
                embedding_model=embedding_model,
                llm_provider=llm_provider,
            )
            row = adapter.validate_setup()
            row.setdefault("system", system_name)
            row.setdefault("ready", True)
            row.setdefault("provider", llm_provider or "")
            row.setdefault("message", "")
            if generation_error:
                row["ready"] = False
                row["message"] = (
                    f"{row['message']}; " if row.get("message") else ""
                ) + f"Generation API unavailable: {generation_error}"
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "system": system_name,
                    "ready": False,
                    "provider": llm_provider or "",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    pass
    return rows


def generation_api_preflight_error(
    *, generation_model: str, llm_provider: str | None
) -> str | None:
    if not parse_bool(os.getenv("MEMORY_EVAL_PREFLIGHT_GENERATION", "true")):
        return None
    try:
        from map_platform.llm.gateway import LLMGateway

        response = LLMGateway(provider=llm_provider).complete_text(
            prompt="Reply with exactly: OK",
            model=generation_model,
            temperature=0.0,
        )
        if not str(response.text or "").strip():
            return "empty response"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def runnable_systems(preflight_rows: list[dict[str, Any]], *, skip_unavailable: bool) -> list[str]:
    systems: list[str] = []
    for row in preflight_rows:
        if row.get("ready") or not skip_unavailable:
            systems.append(str(row.get("system") or ""))
    return [system for system in systems if system]


def apply_env(env: dict[str, Any]) -> None:
    for key, value in env.items():
        text = str(value).strip()
        if text and not is_placeholder_env_value(key=str(key), value=text):
            os.environ[str(key)] = text
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_API_BASE_URL")
    if base_url:
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
        if is_local_base_url(base_url):
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") and not is_remote_key_context() else "dummy"
        if os.getenv("OPENAI_API_BASE") == base_url:
            os.environ.pop("OPENAI_API_BASE", None)
        if os.getenv("OPENAI_API_BASE_URL") == base_url:
            os.environ.pop("OPENAI_API_BASE_URL", None)
    build_base_url = os.getenv("MEMORY_BUILD_OPENAI_BASE_URL") or os.getenv("MEMORY_EXTRACTION_OPENAI_BASE_URL")
    build_api_key = os.getenv("MEMORY_BUILD_OPENAI_API_KEY") or os.getenv("MEMORY_EXTRACTION_OPENAI_API_KEY")
    build_model = os.getenv("MEMORY_BUILD_MODEL") or os.getenv("MEMORY_EXTRACTION_MODEL")
    if build_base_url:
        os.environ.setdefault("MEM0_LLM_BASE_URL", build_base_url)
        os.environ.setdefault("LANGMEM_LLM_BASE_URL", build_base_url)
    if build_api_key:
        os.environ.setdefault("MEM0_LLM_API_KEY", build_api_key)
        os.environ.setdefault("LANGMEM_LLM_API_KEY", build_api_key)
    if build_model:
        os.environ.setdefault("MEM0_LLM_MODEL", build_model)
        os.environ.setdefault("LANGMEM_LLM_MODEL", build_model)
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_EMBEDDING_BASE_URL")
    if embedding_base_url:
        os.environ.setdefault("EMBEDDING_BASE_URL", embedding_base_url)
        os.environ.setdefault("OPENAI_EMBEDDING_BASE_URL", embedding_base_url)
    if os.getenv("EMBEDDING_API_KEY"):
        os.environ.setdefault("OPENAI_EMBEDDING_API_KEY", os.getenv("EMBEDDING_API_KEY", ""))
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8000/v1")
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    os.environ.setdefault("LLM_MODEL", "Qwen/Qwen3-8B")
    os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
    os.environ.setdefault("EMBEDDING_API_KEY", "dummy")
    os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-m3")


def is_placeholder_env_value(*, key: str, value: str) -> bool:
    normalized = value.strip().upper()
    if normalized in {"YOUR_KEY", "YOUR_OPENAI_API_KEY", "YOUR_LETTA_API_KEY", "DUMMY"}:
        return True
    if key.upper().endswith("_KEY") and normalized in {"", "NONE", "NULL", "PLACEHOLDER"}:
        return True
    return False


def is_local_base_url(base_url: str) -> bool:
    normalized = base_url.strip().lower()
    return (
        "://localhost" in normalized
        or "://127.0.0.1" in normalized
        or "://0.0.0.0" in normalized
    )


def is_remote_key_context() -> bool:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return True
    if key.lower() in {"dummy", "none", "null", "placeholder"}:
        return False
    return True


def run_one(
    *,
    sample: UnifiedSample,
    system_name: str,
    top_k: int,
    generation_model: str,
    embedding_model: str,
    llm_provider: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    gold_memory_ids = gold_ids(sample)
    original_category = original_question_type(sample)
    memory_entries: list[MemoryEntry] = []
    retrieval = RetrievalResult(query="", retrieved_entries=[], top_k=top_k)
    prompt = PromptRecord(system_prompt=None, user_prompt="", memory_context="", full_prompt="")
    adapter = None
    try:
        adapter = build_memory_adapter(
            system_name,
            generation_model=generation_model,
            embedding_model=embedding_model,
            llm_provider=llm_provider,
        )
        adapter.reset(sample.sample_id)
        adapter.build_memory(sample.history, sample)
        memory_entries = adapter.get_memory_entries()
        retrieval = adapter.retrieve(sample.question, sample, top_k)
        prompt = adapter.build_prompt(sample.question, retrieval, sample)
        answer = adapter.generate_answer(prompt, sample)
        persist_sample_artifact(adapter, sample, include_memory=True)
        trace = build_trace(system_name, sample, retrieval, prompt, memory_entries)
        metrics = compute_metrics(
            sample=sample,
            system_name=system_name,
            gold_answer=sample.canonical_answer(),
            pred_answer=answer.answer,
            gold_memory_ids=gold_memory_ids,
            retrieved_source_ids=trace["retrieved_source_ids"],
            injected_source_ids=trace["injected_source_ids"],
        )
        return {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "system": system_name,
            "system_profile": system_profile(system_name),
            "question_type": sample.question_group(),
            "original_question_type": original_category,
            "original_category": original_category,
            "question": sample.question,
            "gold_answer": sample.answer,
            "pred_answer": answer.answer,
            "normalized_answer": normalize_for_match(answer.answer),
            "evaluation_pred_answer": metrics.get("evaluation_pred_answer"),
            "evaluation_normalized_answer": metrics.get("evaluation_normalized_answer"),
            "gold_memory_ids": gold_memory_ids,
            "gold_evidence_units": gold_evidence_units(sample, gold_memory_ids),
            **trace,
            "system_diagnostics": {
                **system_diagnostics(system_name, sample, retrieval, prompt, memory_entries),
                "adapter_debug": adapter_debug_diagnostics(adapter),
            },
            **metrics,
            "latency": time.perf_counter() - started,
            "token_usage": answer.token_usage or {},
            "answer_raw_response": answer.raw_response or {},
            "prompt_tokens": (answer.token_usage or {}).get("prompt_tokens")
            or (answer.token_usage or {}).get("prompt_tokens_est")
            or prompt.token_count,
            "completion_tokens": (answer.token_usage or {}).get("completion_tokens")
            or (answer.token_usage or {}).get("completion_tokens_est"),
            "total_tokens": (answer.token_usage or {}).get("total_tokens"),
            "adapter_backend": adapter.dump_debug_state().get("official_source", {}).get("implementation_mode", "native"),
            "retrieved_memories": summarized_retrieved_memories(system_name, retrieval),
            "error": None,
        }
    except Exception as exc:
        trace = build_trace(system_name, sample, retrieval, prompt, memory_entries)
        metrics = compute_metrics(
            sample=sample,
            system_name=system_name,
            gold_answer=sample.canonical_answer(),
            pred_answer=ERROR_PRED_ANSWER,
            gold_memory_ids=gold_memory_ids,
            retrieved_source_ids=trace["retrieved_source_ids"],
            injected_source_ids=trace["injected_source_ids"],
        )
        metrics.update(error_metrics())
        return {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "system": system_name,
            "system_profile": system_profile(system_name),
            "question_type": sample.question_group(),
            "original_question_type": original_category,
            "original_category": original_category,
            "question": sample.question,
            "gold_answer": sample.answer,
            "pred_answer": ERROR_PRED_ANSWER,
            "normalized_answer": normalize_for_match(ERROR_PRED_ANSWER),
            "gold_memory_ids": gold_memory_ids,
            "gold_evidence_units": gold_evidence_units(sample, gold_memory_ids),
            **trace,
            "system_diagnostics": {
                "stage": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "llm_request": llm_request_diagnostics(generation_model),
                "adapter_debug": adapter_debug_diagnostics(adapter) if adapter is not None else {},
            },
            **metrics,
            "latency": time.perf_counter() - started,
            "token_usage": {},
            "prompt_tokens": prompt.token_count,
            "completion_tokens": None,
            "total_tokens": None,
            "adapter_backend": "unavailable",
            "retrieved_memories": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass


def eval_build_max_attempts() -> int:
    raw = (
        os.getenv("MEMORY_EVAL_BUILD_MAX_ATTEMPTS")
        or os.getenv("MEMORY_BUILD_MAX_ATTEMPTS")
        or "3"
    )
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def eval_build_retry_delay(attempt: int) -> float:
    base = float(os.getenv("MEMORY_EVAL_BUILD_RETRY_BASE_SECONDS", "30"))
    cap = float(os.getenv("MEMORY_EVAL_BUILD_RETRY_MAX_SECONDS", "180"))
    return min(cap, base * (2 ** max(0, attempt - 1)))


def run_reuse_build_by_conversation(
    *,
    samples: list[UnifiedSample],
    systems: list[str],
    completed: set[tuple[str, str]],
    top_k: int,
    generation_model: str,
    embedding_model: str,
    llm_provider: str | None,
    print_qa_results: bool = True,
) -> Iterator[dict[str, Any]]:
    grouped_samples: dict[str, list[UnifiedSample]] = defaultdict(list)
    for sample in samples:
        grouped_samples[conversation_key(sample)].append(sample)

    for conversation_id, group_samples in grouped_samples.items():
        for system_name in systems:
            pending = [sample for sample in group_samples if (sample.sample_id, system_name) not in completed]
            if not pending:
                continue
            adapter = None
            build_started = time.perf_counter()
            build_error: str | None = None
            memory_entries: list[MemoryEntry] = []
            print(
                f"[reuse-build-start] system={system_name} conversation={conversation_id} qa={len(pending)}",
                flush=True,
            )
            build_sample = pending[0]
            for attempt in range(1, eval_build_max_attempts() + 1):
                try:
                    adapter = build_memory_adapter(
                        system_name,
                        generation_model=generation_model,
                        embedding_model=embedding_model,
                        llm_provider=llm_provider,
                    )
                    adapter.reset(conversation_id)
                    adapter.build_memory(build_sample.history, build_sample)
                    memory_entries = adapter.get_memory_entries()
                    persist_build_artifact(adapter, build_sample, conversation_id)
                    build_error = None
                    print(
                        f"[reuse-build-done] system={system_name} conversation={conversation_id} memories={len(memory_entries)} "
                        f"attempt={attempt} latency={time.perf_counter() - build_started:.2f}s",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    build_error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[reuse-build-error] system={system_name} conversation={conversation_id} attempt={attempt} "
                        f"error={build_error}",
                        flush=True,
                    )
                    if adapter is not None:
                        try:
                            adapter.close()
                        except Exception:
                            pass
                    adapter = None
                    if attempt < eval_build_max_attempts():
                        time.sleep(eval_build_retry_delay(attempt))

            if build_error:
                print(
                    f"[reuse-build-skip] system={system_name} conversation={conversation_id} "
                    "leaving samples incomplete for resume",
                    flush=True,
                )
                continue

            for sample in pending:
                print(
                    f"[reuse-answer-start] system={system_name} conversation={conversation_id} sample={sample.sample_id}",
                    flush=True,
                )
                if adapter is None or build_error:
                    row = unavailable_row_from_build_error(
                        sample=sample,
                        system_name=system_name,
                        top_k=top_k,
                        build_error=build_error or "build adapter unavailable",
                        conversation_id=conversation_id,
                        build_latency=time.perf_counter() - build_started,
                        adapter=adapter,
                    )
                    if print_qa_results:
                        print_qa_result(row)
                    yield row
                    continue
                row = run_one_after_build(
                    sample=sample,
                    system_name=system_name,
                    top_k=top_k,
                    adapter=adapter,
                    memory_entries=memory_entries,
                    conversation_id=conversation_id,
                    build_latency=time.perf_counter() - build_started,
                )
                print(
                    f"[reuse-answer-done] system={system_name} conversation={conversation_id} sample={sample.sample_id} "
                    f"correct={row.get('answer_correct')} error={row.get('error') or ''}",
                    flush=True,
                )
                if print_qa_results:
                    print_qa_result(row)
                yield row
                abort_mnemis_shard = os.getenv(
                    "MNEMIS_ABORT_CONVERSATION_ON_ANSWER_ERROR", ""
                ).strip().lower() in {"1", "true", "yes", "on"}
                if system_name == "mnemis" and row.get("error") and abort_mnemis_shard:
                    raise RuntimeError(
                        f"Mnemis answer failed for {sample.sample_id}; "
                        f"aborting conversation shard: {row['error']}"
                    )
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    pass


def run_one_after_build(
    *,
    sample: UnifiedSample,
    system_name: str,
    top_k: int,
    adapter: Any,
    memory_entries: list[MemoryEntry],
    conversation_id: str,
    build_latency: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    gold_memory_ids = gold_ids(sample)
    original_category = original_question_type(sample)
    retrieval = RetrievalResult(query="", retrieved_entries=[], top_k=top_k)
    prompt = PromptRecord(system_prompt=None, user_prompt="", memory_context="", full_prompt="")
    try:
        retrieval = adapter.retrieve(sample.question, sample, top_k)
        prompt = adapter.build_prompt(sample.question, retrieval, sample)
        answer = adapter.generate_answer(prompt, sample)
        persist_sample_artifact(adapter, sample, include_memory=False)
        trace = build_trace(system_name, sample, retrieval, prompt, memory_entries)
        metrics = compute_metrics(
            sample=sample,
            system_name=system_name,
            gold_answer=sample.canonical_answer(),
            pred_answer=answer.answer,
            gold_memory_ids=gold_memory_ids,
            retrieved_source_ids=trace["retrieved_source_ids"],
            injected_source_ids=trace["injected_source_ids"],
        )
        return {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "system": system_name,
            "system_profile": system_profile(system_name),
            "question_type": sample.question_group(),
            "original_question_type": original_category,
            "original_category": original_category,
            "question": sample.question,
            "gold_answer": sample.answer,
            "pred_answer": answer.answer,
            "normalized_answer": normalize_for_match(answer.answer),
            "evaluation_pred_answer": metrics.get("evaluation_pred_answer"),
            "evaluation_normalized_answer": metrics.get("evaluation_normalized_answer"),
            "gold_memory_ids": gold_memory_ids,
            "gold_evidence_units": gold_evidence_units(sample, gold_memory_ids),
            **trace,
            "system_diagnostics": {
                **system_diagnostics(system_name, sample, retrieval, prompt, memory_entries),
                "adapter_debug": adapter_debug_diagnostics(adapter),
                "build_reuse": {
                    "enabled": True,
                    "conversation_id": conversation_id,
                    "build_latency": build_latency,
                },
            },
            **metrics,
            "latency": time.perf_counter() - started,
            "build_latency": build_latency,
            "token_usage": answer.token_usage or {},
            "answer_raw_response": answer.raw_response or {},
            "prompt_tokens": (answer.token_usage or {}).get("prompt_tokens")
            or (answer.token_usage or {}).get("prompt_tokens_est")
            or prompt.token_count,
            "completion_tokens": (answer.token_usage or {}).get("completion_tokens")
            or (answer.token_usage or {}).get("completion_tokens_est"),
            "total_tokens": (answer.token_usage or {}).get("total_tokens"),
            "adapter_backend": adapter.dump_debug_state().get("official_source", {}).get("implementation_mode", "native"),
            "retrieved_memories": summarized_retrieved_memories(system_name, retrieval),
            "error": None,
        }
    except Exception as exc:
        return unavailable_row_from_build_error(
            sample=sample,
            system_name=system_name,
            top_k=top_k,
            build_error=f"{type(exc).__name__}: {exc}",
            conversation_id=conversation_id,
            build_latency=build_latency,
            adapter=adapter,
            memory_entries=memory_entries,
            retrieval=retrieval,
            prompt=prompt,
        )


def unavailable_row_from_build_error(
    *,
    sample: UnifiedSample,
    system_name: str,
    top_k: int,
    build_error: str,
    conversation_id: str,
    build_latency: float,
    adapter: Any | None,
    memory_entries: list[MemoryEntry] | None = None,
    retrieval: RetrievalResult | None = None,
    prompt: PromptRecord | None = None,
) -> dict[str, Any]:
    memory_entries = memory_entries or []
    retrieval = retrieval or RetrievalResult(query="", retrieved_entries=[], top_k=top_k)
    prompt = prompt or PromptRecord(system_prompt=None, user_prompt="", memory_context="", full_prompt="")
    gold_memory_ids = gold_ids(sample)
    original_category = original_question_type(sample)
    trace = build_trace(system_name, sample, retrieval, prompt, memory_entries)
    metrics = compute_metrics(
        sample=sample,
        system_name=system_name,
        gold_answer=sample.canonical_answer(),
        pred_answer=ERROR_PRED_ANSWER,
        gold_memory_ids=gold_memory_ids,
        retrieved_source_ids=trace["retrieved_source_ids"],
        injected_source_ids=trace["injected_source_ids"],
    )
    metrics.update(error_metrics())
    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "system": system_name,
        "system_profile": system_profile(system_name),
        "question_type": sample.question_group(),
        "original_question_type": original_category,
        "original_category": original_category,
        "question": sample.question,
        "gold_answer": sample.answer,
        "pred_answer": ERROR_PRED_ANSWER,
        "normalized_answer": normalize_for_match(ERROR_PRED_ANSWER),
        "gold_memory_ids": gold_memory_ids,
        "gold_evidence_units": gold_evidence_units(sample, gold_memory_ids),
        **trace,
        "system_diagnostics": {
            "stage": "unavailable",
            "error": build_error,
            "llm_request": llm_request_diagnostics(str(getattr(adapter, "generation_model", ""))),
            "adapter_debug": adapter_debug_diagnostics(adapter) if adapter is not None else {},
            "build_reuse": {
                "enabled": True,
                "conversation_id": conversation_id,
                "build_latency": build_latency,
            },
        },
        **metrics,
        "latency": build_latency,
        "build_latency": build_latency,
        "token_usage": {},
        "prompt_tokens": prompt.token_count,
        "completion_tokens": None,
        "total_tokens": None,
        "adapter_backend": "unavailable",
        "retrieved_memories": [],
        "error": build_error,
    }


def conversation_key(sample: UnifiedSample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    for key in ("conversation_id", "dialog_id", "conversation", "session_group_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    sample_id = str(sample.sample_id)
    if "__qa_" in sample_id:
        return sample_id.split("__qa_", 1)[0]
    if "__counterfactual" in sample_id:
        return sample_id.split("__counterfactual", 1)[0]
    return sample_id


def gold_ids(sample: UnifiedSample) -> list[str]:
    metadata_ids = sample.metadata.get("gold_memory_ids") if isinstance(sample.metadata, dict) else None
    if isinstance(metadata_ids, list):
        return [normalize_text(item) for item in metadata_ids if normalize_text(item)]
    raw_ref = sample.metadata.get("raw_evidence_ref", {}) if isinstance(sample.metadata, dict) else {}
    for key in (
        "dialog_ids",
        "evidence",
        "answer_session_ids",
        "evidence_sessions",
        "gold_sessions",
        "related_sessions",
        "turn_ids",
        "evidence_turn_ids",
        "supporting_turn_ids",
        "evidence_session_ids",
    ):
        value = raw_ref.get(key) if isinstance(raw_ref, dict) else None
        if isinstance(value, list):
            return [normalize_text(item) for item in value if normalize_text(item)]
    return []


def gold_evidence_units(sample: UnifiedSample, gold_memory_ids: list[str]) -> list[dict[str, Any]]:
    history_by_id: dict[str, dict[str, Any]] = {}
    for turn in sample.history:
        for key in (turn.get("turn_id"), turn.get("session_id")):
            normalized = normalize_text(key)
            if normalized and normalized not in history_by_id:
                history_by_id[normalized] = turn
    units: list[dict[str, Any]] = []
    for source_id in gold_memory_ids:
        turn = history_by_id.get(source_id)
        units.append(
            {
                "source_id": source_id,
                "text": str(turn.get("text") or "").strip() if turn else "",
                "session_id": normalize_text(turn.get("session_id")) if turn else "",
                "turn_id": normalize_text(turn.get("turn_id")) if turn else "",
                "source_mapping_available": turn is not None,
            }
        )
    if units:
        return units
    return [
        {"source_id": "", "text": str(item).strip(), "source_mapping_available": False}
        for item in sample.derived_key_evidence_units()
        if str(item).strip()
    ]


def original_question_type(sample: UnifiedSample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return normalize_text(
        metadata.get("raw_category")
        or metadata.get("category")
        or metadata.get("question_type")
        or sample.question_type
        or sample.task_type
    )


def system_profile(system_name: str) -> dict[str, Any]:
    normalized = system_name.strip().lower()
    return dict(SYSTEM_PROFILES.get(normalized, {
        "memory_mode": "retrieval_augmented_memory",
        "evaluation_role": "custom_memory_system",
        "retrieval_policy": "adapter_defined",
        "injection_policy": "adapter_defined",
        "primary_metrics": ["retrieval_recall", "prompt_gold_coverage", "answer_accuracy"],
        "diagnostic_focus": "adapter-defined memory behavior",
    }))


def build_trace(
    system_name: str,
    sample: UnifiedSample,
    retrieval: RetrievalResult,
    prompt: PromptRecord,
    memory_entries: list[MemoryEntry],
) -> dict[str, Any]:
    normalized_system = system_name.strip().lower()
    gold_memory_ids = gold_ids(sample)
    gold_set = normalized_id_set(gold_memory_ids)
    if normalized_system == "no_memory":
        retrieved_source_ids: list[str] = []
        injected_source_ids: list[str] = []
        return {
            "retrieved_source_ids": retrieved_source_ids,
            "injected_source_ids": injected_source_ids,
            "retrieved_gold_ids": sorted(gold_set & normalized_id_set(retrieved_source_ids)),
            "prompt_gold_ids": sorted(gold_set & normalized_id_set(injected_source_ids)),
            "missing_retrieved_gold_ids": sorted(gold_set - normalized_id_set(retrieved_source_ids)),
            "missing_prompt_gold_ids": sorted(gold_set - normalized_id_set(injected_source_ids)),
            "source_mapping_available": False,
            "retrieval_source_mapping_available": False,
            "prompt_source_mapping_available": False,
            "memory_mode": system_profile(system_name)["memory_mode"],
            "metric_policy": "zero_recall_baseline",
            "trace_available": False,
            "trace_method": "not_applicable",
        }
    if normalized_system == "full_context":
        gold_in_context = [source_id for source_id in gold_memory_ids if source_id_in_history(source_id, sample)]
        retrieved_source_ids = list(gold_in_context)
        injected_source_ids = list(gold_in_context)
        return {
            "retrieved_source_ids": retrieved_source_ids,
            "injected_source_ids": injected_source_ids,
            "retrieved_gold_ids": sorted(gold_set & normalized_id_set(retrieved_source_ids)),
            "prompt_gold_ids": sorted(gold_set & normalized_id_set(injected_source_ids)),
            "missing_retrieved_gold_ids": sorted(gold_set - normalized_id_set(retrieved_source_ids)),
            "missing_prompt_gold_ids": sorted(gold_set - normalized_id_set(injected_source_ids)),
            "source_mapping_available": bool(gold_in_context or not gold_memory_ids),
            "retrieval_source_mapping_available": bool(gold_in_context or not gold_memory_ids),
            "prompt_source_mapping_available": bool(gold_in_context or not gold_memory_ids),
            "memory_mode": system_profile(system_name)["memory_mode"],
            "metric_policy": "gold_coverage_without_listing_all_context",
            "trace_available": bool(gold_in_context or not gold_memory_ids),
            "trace_method": "full_context_gold_coverage",
            "full_context_num_history_units": len(sample.history),
        }
    retrieved_source_ids = flatten_source_ids_from_retrieval(retrieval, sample)
    injected_source_ids = flatten_source_ids_from_prompt(prompt, memory_entries, retrieval, sample)
    retrieved_set = normalized_id_set(retrieved_source_ids)
    injected_set = normalized_id_set(injected_source_ids)
    return {
        "retrieved_source_ids": retrieved_source_ids,
        "injected_source_ids": injected_source_ids,
        "retrieved_gold_ids": sorted(gold_set & retrieved_set),
        "prompt_gold_ids": sorted(gold_set & injected_set),
        "missing_retrieved_gold_ids": sorted(gold_set - retrieved_set),
        "missing_prompt_gold_ids": sorted(gold_set - injected_set),
        "source_mapping_available": bool(retrieved_source_ids or injected_source_ids),
        "retrieval_source_mapping_available": bool(retrieved_source_ids),
        "prompt_source_mapping_available": bool(injected_source_ids),
        "memory_mode": system_profile(system_name)["memory_mode"],
        "metric_policy": "retrieval_and_prompt_source_ids",
        "trace_available": bool(retrieved_source_ids or injected_source_ids or not gold_memory_ids),
        "trace_method": trace_method(retrieval, memory_entries),
    }


TURN_ID_RE_FOR_METRIC = re.compile(r"\bD\d+:\d+\b", re.I)


def normalized_id_set(values: list[str]) -> set[str]:
    out: set[str] = set()
    for value in values or []:
        text = str(value or "")
        turn_ids = TURN_ID_RE_FOR_METRIC.findall(text)
        parts = turn_ids or [text]
        for part in parts:
            normalized = normalize_text(part).lower()
            if normalized:
                out.add(normalized)
    return out


def source_id_in_history(source_id: str, sample: UnifiedSample) -> bool:
    normalized = normalize_text(source_id)
    if not normalized:
        return False
    for turn in sample.history:
        if normalized in {normalize_text(turn.get("turn_id")), normalize_text(turn.get("session_id"))}:
            return True
    return False


def system_diagnostics(
    system_name: str,
    sample: UnifiedSample,
    retrieval: RetrievalResult,
    prompt: PromptRecord,
    memory_entries: list[MemoryEntry],
) -> dict[str, Any]:
    normalized = system_name.strip().lower()
    base = {
        "profile": system_profile(system_name),
        "num_memory_entries": len(memory_entries),
        "num_retrieved_entries": len(retrieval.retrieved_entries),
        "num_injected_entries": len(prompt.injected_entry_ids),
        "prompt_token_count": prompt.token_count,
    }
    if normalized == "no_memory":
        return {
            **base,
            "diagnostic_stage": "question_only",
            "expected_retrieval_recall": 0.0,
            "expected_prompt_coverage": 0.0,
        }
    if normalized == "full_context":
        return {
            **base,
            "diagnostic_stage": "full_history_injection",
            "num_history_units": len(sample.history),
            "gold_ids_present_in_history": [item for item in gold_ids(sample) if source_id_in_history(item, sample)],
            "output_policy": "retrieved_memories_summarized_to_avoid_full_history_duplication",
        }
    if normalized in {"simple_vector", "current_memory", "mem0", "langmem"}:
        return {
            **base,
            "diagnostic_stage": "top_k_memory_retrieval",
            "top_k": retrieval.top_k,
            "retrieved_entry_ids": [entry.entry_id for entry in retrieval.retrieved_entries],
            "retrieved_scores": [entry.score for entry in retrieval.retrieved_entries],
            "retrieved_source_id_counts": [len(entry.source_ids) for entry in retrieval.retrieved_entries],
        }
    if normalized == "memorybank":
        buckets = bucket_counts(memory_entries)
        retrieved_buckets = bucket_counts_from_retrieval(retrieval)
        return {
            **base,
            "diagnostic_stage": "hierarchical_summary_detail_retrieval",
            "memory_bucket_counts": buckets,
            "retrieved_bucket_counts": retrieved_buckets,
            "retrieved_retention_weights": [
                entry.metadata.get("retention_weight") for entry in retrieval.retrieved_entries
            ],
            "retrieved_semantic_scores": [
                entry.metadata.get("semantic_score") for entry in retrieval.retrieved_entries
            ],
        }
    if normalized == "readagent":
        return {
            **base,
            "diagnostic_stage": "gist_lookup_and_page_expansion",
            "selected_pages": retrieval.raw.get("selected_pages") if isinstance(retrieval.raw, dict) else None,
            "num_parallel_lookup_chars": len(str(retrieval.raw.get("parallel_lookup_prompt", "")))
            if isinstance(retrieval.raw, dict)
            else None,
            "num_expanded_pages": sum(
                1 for entry in retrieval.retrieved_entries if entry.metadata.get("memory_bucket") == "page"
            ),
            "retrieved_page_ids": [entry.metadata.get("page_id") for entry in retrieval.retrieved_entries],
        }
    if normalized in {"letta", "memgpt"}:
        return {
            **base,
            "diagnostic_stage": "agent_core_and_archival_memory",
            "memory_bucket_counts": bucket_counts(memory_entries),
            "retrieved_bucket_counts": bucket_counts_from_retrieval(retrieval),
            "core_entry_ids": [
                entry.entry_id for entry in memory_entries if entry.metadata.get("memory_bucket") == "core"
            ],
            "archival_retrieved_ids": [entry.entry_id for entry in retrieval.retrieved_entries],
        }
    return {**base, "diagnostic_stage": "adapter_defined"}


def adapter_debug_diagnostics(adapter: Any) -> dict[str, Any]:
    debug = adapter.dump_debug_state()
    noisy_keys = {
        "memory_entries",
        "organization_state",
        "retrieval",
        "prompt_record",
        "answer_record",
        "build_memory_response",
    }
    compact = {key: value for key, value in debug.items() if key not in noisy_keys}
    if isinstance(compact.get("mem0_direct_llm_calls"), list):
        compact["mem0_direct_llm_calls"] = [
            {
                "model": item.get("model"),
                "status": item.get("status"),
                "latency": item.get("latency"),
                "response_format": item.get("response_format"),
                "token_param": item.get("token_param"),
                "max_tokens": item.get("max_tokens"),
                "retried_without_json_mode": item.get("retried_without_json_mode", False),
                "content_preview": str(item.get("content_preview") or "")[:160],
            }
            for item in compact["mem0_direct_llm_calls"]
            if isinstance(item, dict)
        ]
    if isinstance(compact.get("build_memory_batches"), list):
        compact["build_memory_batches"] = [
            {
                "batch": item.get("batch"),
                "total_batches": item.get("total_batches"),
                "num_messages": item.get("num_messages"),
                "start": item.get("start"),
                "status": item.get("status"),
                "response_preview": str(item.get("response_preview") or "")[:160],
                "semantic_update_preview": str(item.get("semantic_update_preview") or "")[:160],
                "procedural_update_preview": str(item.get("procedural_update_preview") or "")[:160],
            }
            for item in compact["build_memory_batches"]
            if isinstance(item, dict)
        ]
    dual_layer = compact.get("dual_layer")
    if isinstance(dual_layer, dict):
        query_traces = dual_layer.get("query_traces", [])
        last_query_trace = query_traces[-1] if isinstance(query_traces, list) and query_traces else {}
        controller_trace = {}
        packet_gate_trace = {}
        slot_v2_trace = {}
        if isinstance(last_query_trace, dict):
            controller_trace = last_query_trace.get("evidence_controller") or {}
            packet_gate_trace = last_query_trace.get("packet_gate") or {}
            slot_v2_trace = last_query_trace.get("slot_program_v2") or {}
        compact["dual_layer"] = {
            "logic_layer_keys": sorted(
                dual_layer.get("logic_layer", {}).keys()
                if isinstance(dual_layer.get("logic_layer"), dict)
                else []
            ),
            "raw_layer_keys": sorted(
                dual_layer.get("raw_layer", {}).keys()
                if isinstance(dual_layer.get("raw_layer"), dict)
                else []
            ),
            "write_trace_count": len(dual_layer.get("write_traces", []))
            if isinstance(dual_layer.get("write_traces"), list)
            else 0,
            "query_trace_count": len(dual_layer.get("query_traces", []))
            if isinstance(dual_layer.get("query_traces"), list)
            else 0,
            "last_evidence_controller": controller_trace,
            "last_slot_program_v2": slot_v2_trace,
            "last_packet_gate": packet_gate_trace,
        }
    return compact


def llm_request_diagnostics(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "memory_build_openai_base_url": os.getenv("MEMORY_BUILD_OPENAI_BASE_URL"),
        "memory_build_model": os.getenv("MEMORY_BUILD_MODEL"),
        "mem0_llm_base_url": os.getenv("MEM0_LLM_BASE_URL"),
        "langmem_llm_base_url": os.getenv("LANGMEM_LLM_BASE_URL"),
        "openai_api_key_source": env_value_source("OPENAI_API_KEY"),
        "memory_build_openai_api_key_source": env_value_source("MEMORY_BUILD_OPENAI_API_KEY"),
        "mem0_llm_api_key_source": env_value_source("MEM0_LLM_API_KEY"),
        "langmem_llm_api_key_source": env_value_source("LANGMEM_LLM_API_KEY"),
        "llm_use_extra_body": os.getenv("LLM_USE_EXTRA_BODY"),
        "llm_disable_thinking": os.getenv("LLM_DISABLE_THINKING", "1"),
    }


def env_value_source(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        return "unset"
    text = str(value)
    if not text:
        return "empty"
    if text.lower() in {"dummy", "none", "null", "placeholder"}:
        return text.lower()
    return f"set:{len(text)}chars"


def bucket_counts(entries: list[MemoryEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        bucket = str(entry.metadata.get("memory_bucket") or entry.metadata.get("baseline") or "unknown")
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def bucket_counts_from_retrieval(retrieval: RetrievalResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in retrieval.retrieved_entries:
        bucket = str(entry.metadata.get("memory_bucket") or entry.metadata.get("baseline") or "unknown")
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def flatten_source_ids_from_retrieval(retrieval: RetrievalResult, sample: UnifiedSample) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for entry in retrieval.retrieved_entries:
        for source_id in usable_source_ids(entry.source_ids, sample):
            add_unique(result, seen, source_id)
        for source_id in infer_source_ids_from_text(entry.content, sample):
            add_unique(result, seen, source_id)
        for source_id in infer_source_ids_from_batch_metadata(entry.content, entry.metadata, sample, query=retrieval.query):
            add_unique(result, seen, source_id)
    return result


def flatten_source_ids_from_prompt(
    prompt: PromptRecord,
    memory_entries: list[MemoryEntry],
    retrieval: RetrievalResult,
    sample: UnifiedSample,
) -> list[str]:
    entries_by_id = {entry.entry_id: entry for entry in memory_entries}
    retrieved_by_id = {entry.entry_id: entry for entry in retrieval.retrieved_entries}
    result: list[str] = []
    seen: set[str] = set()
    for entry_id in prompt.injected_entry_ids:
        memory_entry = entries_by_id.get(entry_id)
        retrieved_entry = retrieved_by_id.get(entry_id)
        source_ids = []
        if memory_entry is not None:
            source_ids.extend(usable_source_ids(memory_entry.source_ids, sample))
        if retrieved_entry is not None:
            source_ids.extend(usable_source_ids(retrieved_entry.source_ids, sample))
        if not source_ids:
            source_ids.append(entry_id)
        for source_id in source_ids:
            add_unique(result, seen, source_id)
        texts = []
        if memory_entry is not None:
            texts.append(memory_entry.content)
        if retrieved_entry is not None:
            texts.append(retrieved_entry.content)
        for text in texts:
            for source_id in infer_source_ids_from_text(text, sample):
                add_unique(result, seen, source_id)
            metadata = retrieved_entry.metadata if retrieved_entry is not None else memory_entry.metadata if memory_entry is not None else {}
            for source_id in infer_source_ids_from_batch_metadata(text, metadata, sample, query=retrieval.query):
                add_unique(result, seen, source_id)
    for source_id in infer_source_ids_from_text(prompt.memory_context, sample):
        add_unique(result, seen, source_id)
    return result


def usable_source_ids(source_ids: list[str], sample: UnifiedSample) -> list[str]:
    normalized = [normalize_text(item) for item in source_ids if normalize_text(item)]
    if not normalized:
        return []
    broad_limit = max(20, len(sample.history) // 4)
    if len(set(normalized)) > broad_limit:
        return []
    return normalized


def infer_source_ids_from_text(text: str, sample: UnifiedSample) -> list[str]:
    explicit_ids = explicit_source_ids_from_text(text, sample)
    normalized_text = normalize_for_source_match(text)
    output: list[str] = list(explicit_ids)
    seen: set[str] = set(explicit_ids)
    if not normalized_text:
        return output
    for turn in sample.history:
        turn_text = normalize_for_source_match(turn.get("text"))
        if not turn_text:
            continue
        if turn_text in normalized_text or normalized_text in turn_text:
            add_unique(output, seen, turn.get("turn_id"))
            add_unique(output, seen, turn.get("session_id"))
            continue
        turn_tokens = set(turn_text.split())
        text_tokens = set(normalized_text.split())
        if turn_tokens and len(turn_tokens & text_tokens) / len(turn_tokens) >= 0.85:
            add_unique(output, seen, turn.get("turn_id"))
            add_unique(output, seen, turn.get("session_id"))
    return output


def explicit_source_ids_from_text(text: str, sample: UnifiedSample) -> list[str]:
    raw_text = str(text or "")
    if not raw_text.strip():
        return []
    valid_ids = {
        normalize_text(value)
        for turn in sample.history
        for value in (turn.get("turn_id"), turn.get("session_id"))
        if normalize_text(value)
    }
    candidates = re.findall(r"\bD\d+:\d+\b|\bsession_\d+\b", raw_text, flags=re.IGNORECASE)
    for match in re.finditer(
        r"\b(?:source_turn_ids?|source_ids?|turn_ids?|session_ids?)\s*[:=]\s*([^;\]\)\n]+)",
        raw_text,
        flags=re.IGNORECASE,
    ):
        candidates.extend(re.split(r"[\s,;]+", match.group(1)))
    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = normalize_text(str(candidate).strip("[](){}'\".,;"))
        if cleaned in valid_ids:
            add_unique(output, seen, cleaned)
    return output


def infer_source_ids_from_batch_metadata(
    text: str,
    metadata: dict[str, Any],
    sample: UnifiedSample,
    *,
    query: str = "",
) -> list[str]:
    candidate_ids = source_batch_turn_ids_from_metadata(metadata)
    if not candidate_ids:
        return []
    history_by_id = {normalize_text(turn.get("turn_id")): turn for turn in sample.history if normalize_text(turn.get("turn_id"))}
    scored: list[tuple[float, int, str]] = []
    match_text = f"{query} {text}"
    for index, source_id in enumerate(candidate_ids):
        turn = history_by_id.get(source_id)
        if not turn:
            continue
        score = evidence_candidate_score(match_text, turn)
        if score >= 0.34:
            scored.append((score, index, source_id))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score = scored[0][0]
    selected = [source_id for score, _index, source_id in scored if score >= best_score - 0.08]
    return selected[:5]


def source_batch_turn_ids_from_metadata(metadata: dict[str, Any]) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    candidates: list[Any] = []
    for container in (metadata, metadata.get("metadata")):
        if isinstance(container, dict):
            value = container.get("source_batch_turn_ids") or container.get("source_turn_ids") or []
            if isinstance(value, list):
                candidates.extend(value)
            elif value:
                candidates.append(value)
    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        add_unique(output, seen, candidate)
    return output


def evidence_candidate_score(match_text: str, turn: dict[str, Any]) -> float:
    turn_text = normalize_for_source_match(turn.get("text"))
    memory_text = normalize_for_source_match(match_text)
    if not turn_text or not memory_text:
        return 0.0
    turn_tokens = set(turn_text.split())
    memory_tokens = set(memory_text.split())
    if not turn_tokens or not memory_tokens:
        return 0.0
    recall_score = len(turn_tokens & memory_tokens) / len(turn_tokens)
    precision_score = len(turn_tokens & memory_tokens) / len(memory_tokens)
    score = (0.8 * recall_score) + (0.2 * precision_score)
    timestamp = str(turn.get("timestamp") or "")
    for date_value in extract_dates(timestamp):
        if date_value in extract_dates(match_text):
            score += 0.2
            break
    return min(score, 1.0)


def normalize_for_source_match(value: Any) -> str:
    normalized = normalize_text(value).lower()
    normalized = re.sub(r"[^a-z0-9']+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def trace_method(retrieval: RetrievalResult, memory_entries: list[MemoryEntry]) -> str:
    explicit = any(entry.source_ids for entry in retrieval.retrieved_entries) or any(entry.source_ids for entry in memory_entries)
    return "explicit_or_inferred" if explicit else "content_inferred"


def summarized_retrieved_memories(system_name: str, retrieval: RetrievalResult) -> list[dict[str, Any]]:
    if system_name.strip().lower() == "full_context":
        return [
            {
                "summary": "full_context injects the complete sample history",
                "num_entries": len(retrieval.retrieved_entries),
            }
        ]
    rows: list[dict[str, Any]] = []
    for entry in retrieval.retrieved_entries:
        row = entry.to_dict()
        source_ids = row.get("source_ids")
        if isinstance(source_ids, list) and len(source_ids) > 10:
            row["source_ids"] = source_ids[:10]
            row["source_ids_truncated"] = len(source_ids) - 10
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            nested = metadata.get("metadata")
            if isinstance(nested, dict):
                batch_ids = nested.get("source_batch_turn_ids")
                if isinstance(batch_ids, list) and len(batch_ids) > 10:
                    nested["source_batch_turn_ids"] = batch_ids[:10]
                    nested["source_batch_turn_ids_truncated"] = len(batch_ids) - 10
        rows.append(row)
    return rows


def add_unique(result: list[str], seen: set[str], value: Any) -> None:
    normalized = normalize_text(value)
    if normalized and normalized not in seen:
        seen.add(normalized)
        result.append(normalized)


def compute_metrics(
    *,
    sample: UnifiedSample,
    system_name: str,
    gold_answer: str,
    pred_answer: str,
    gold_memory_ids: list[str],
    retrieved_source_ids: list[str],
    injected_source_ids: list[str],
) -> dict[str, Any]:
    answer_candidates = sample.answer_candidates() or [gold_answer]
    eval_pred_answer = canonicalize_answer_for_eval(pred_answer)
    eval_answer_candidates = [canonicalize_answer_for_eval(item) for item in answer_candidates]
    nonempty_eval_answer_candidates = [item for item in eval_answer_candidates if normalize_for_match(item)]
    normalized_candidates = [normalize_for_match(item) for item in eval_answer_candidates]
    nonempty_normalized_candidates = [item for item in normalized_candidates if item]
    normalized_pred = normalize_for_match(eval_pred_answer)

    # Cat5 / empty gold_answer: if all gold candidates are empty, the expected behavior
    # is either refusal ("not enough information") or a correction of the false premise.
    # Treat refusals and non-trivial corrections as correct.
    if not nonempty_normalized_candidates:
        _refusal_phrases = {"not enough information", "not enough info", "insufficient information",
                           "no answer", "unknown", "cannot be determined", "cannot be answered",
                           "no information", "not mentioned", "not available"}
        is_refusal = any(phrase in normalized_pred for phrase in _refusal_phrases) if normalized_pred else True
        # A substantive correction (e.g., "X did not do Y; it was Z who...") is also acceptable
        is_correction = len(normalized_pred.split()) > 5 if normalized_pred else False
        answer_correct = is_refusal or is_correction
        answer_f1 = 1.0 if answer_correct else 0.0
        bleu_1 = 1.0 if answer_correct else 0.0
        exact_accuracy = is_refusal
        normalized_accuracy = is_refusal
        date_accuracy = False
    else:
        pred_dates = extract_dates(eval_pred_answer)
        gold_dates = {date_value for candidate in nonempty_eval_answer_candidates for date_value in extract_dates(candidate)}
        answer_f1 = max((token_f1(eval_pred_answer, candidate) for candidate in nonempty_eval_answer_candidates), default=0.0)
        exact_accuracy = bool(eval_pred_answer and nonempty_eval_answer_candidates) and any(
            normalize_text(eval_pred_answer) == normalize_text(candidate) for candidate in nonempty_eval_answer_candidates
        )
        normalized_accuracy = any(candidate == normalized_pred for candidate in nonempty_normalized_candidates)
        answer_contains_gold = any(candidate in normalized_pred for candidate in nonempty_normalized_candidates)
        date_accuracy = bool(gold_dates and pred_dates and gold_dates.issubset(pred_dates))
        date_mismatch = bool(gold_dates and pred_dates and not gold_dates.issubset(pred_dates))
        if gold_dates:
            answer_correct = (normalized_accuracy or answer_contains_gold or date_accuracy) and not date_mismatch
        else:
            answer_correct = normalized_accuracy or answer_f1 >= 0.5 or answer_contains_gold
        bleu_1 = max((bleu1(eval_pred_answer, candidate) for candidate in nonempty_eval_answer_candidates), default=0.0)
    has_gold_evidence = bool(normalized_id_set(gold_memory_ids))
    retrieval_recall = recall(gold_memory_ids, retrieved_source_ids) if has_gold_evidence else None
    prompt_gold_coverage = recall(gold_memory_ids, injected_source_ids) if has_gold_evidence else None
    retrieval_hit = hit(gold_memory_ids, retrieved_source_ids) if has_gold_evidence else None
    prompt_hit = hit(gold_memory_ids, injected_source_ids) if has_gold_evidence else None
    complete_evidence_recall = complete_recall(gold_memory_ids, retrieved_source_ids) if has_gold_evidence else None
    prompt_complete_evidence_recall = complete_recall(gold_memory_ids, injected_source_ids) if has_gold_evidence else None
    injection_loss = (
        float(retrieval_recall) - float(prompt_gold_coverage)
        if retrieval_recall is not None and prompt_gold_coverage is not None
        else None
    )
    gold_in_prompt = bool(prompt_complete_evidence_recall) if has_gold_evidence else False
    accuracy_given_gold_in_prompt = bool(answer_correct) if gold_in_prompt else None
    post_evidence_answer_failure = bool(gold_in_prompt and not answer_correct)
    failure_type = classify_failure(
        sample=sample,
        system_name=system_name,
        answer_correct=answer_correct,
        has_gold_evidence=has_gold_evidence,
        retrieval_recall=retrieval_recall,
        prompt_gold_coverage=prompt_gold_coverage,
        complete_evidence_recall=complete_evidence_recall,
        prompt_complete_evidence_recall=prompt_complete_evidence_recall,
    )
    return {
        "answer_exact_accuracy": exact_accuracy,
        "normalized_answer_accuracy": normalized_accuracy,
        "date_answer_accuracy": date_accuracy,
        "answer_correct": answer_correct,
        "answer_f1": answer_f1,
        "token_f1": answer_f1,
        "bleu_1": bleu_1,
        "evaluation_pred_answer": eval_pred_answer,
        "evaluation_normalized_answer": normalized_pred,
        "has_gold_evidence": has_gold_evidence,
        "retrieval_hit": retrieval_hit,
        "retrieval_recall": retrieval_recall,
        "complete_evidence_recall": complete_evidence_recall,
        "prompt_hit": prompt_hit,
        "prompt_gold_coverage": prompt_gold_coverage,
        "prompt_complete_evidence_recall": prompt_complete_evidence_recall,
        "injection_loss": injection_loss,
        "gold_in_prompt": gold_in_prompt,
        "accuracy_given_gold_in_prompt": accuracy_given_gold_in_prompt,
        "post_evidence_answer_failure": post_evidence_answer_failure,
        "failure_stage": failure_stage(failure_type),
        "failure_type": failure_type,
    }


def classify_failure(
    *,
    sample: UnifiedSample,
    system_name: str,
    answer_correct: bool,
    has_gold_evidence: bool,
    retrieval_recall: float | None,
    prompt_gold_coverage: float | None,
    complete_evidence_recall: bool | None,
    prompt_complete_evidence_recall: bool | None,
) -> str:
    if answer_correct:
        return "success"
    if is_abstention_or_unanswerable(sample):
        return "abstention_or_hallucination_failure"
    if system_name.strip().lower() == "no_memory":
        return "unknown_failure"
    if not has_gold_evidence:
        return "unknown_failure"
    if not retrieval_recall:
        return "retrieval_failure"
    if complete_evidence_recall is False and retrieval_recall > 0:
        return "partial_evidence_failure"
    if retrieval_recall and not prompt_gold_coverage:
        return "injection_failure"
    if prompt_complete_evidence_recall is False and prompt_gold_coverage and prompt_gold_coverage > 0:
        return "partial_evidence_failure"
    if prompt_complete_evidence_recall:
        return "post_evidence_answer_failure"
    return "unknown_failure"


def failure_stage(failure_type: str) -> str:
    mapping = {
        "success": "success",
        "system_error": "system_error",
        "retrieval_failure": "retrieval",
        "partial_evidence_failure": "retrieval_or_prompt_partial",
        "injection_failure": "prompt_injection",
        "post_evidence_answer_failure": "answer_generation",
        "abstention_or_hallucination_failure": "answer_generation",
        "unknown_failure": "unknown",
    }
    return mapping.get(failure_type, "unknown")


def error_metrics() -> dict[str, Any]:
    return {
        "answer_exact_accuracy": False,
        "normalized_answer_accuracy": False,
        "date_answer_accuracy": False,
        "answer_correct": False,
        "answer_f1": 0.0,
        "token_f1": 0.0,
        "bleu_1": 0.0,
        "gold_in_prompt": False,
        "accuracy_given_gold_in_prompt": None,
        "post_evidence_answer_failure": False,
        "failure_stage": "system_error",
        "failure_type": "system_error",
    }


def is_abstention_or_unanswerable(sample: UnifiedSample) -> bool:
    label = " ".join(
        str(item or "")
        for item in (
            original_question_type(sample),
            sample.task_type,
            sample.metadata.get("raw_category") if isinstance(sample.metadata, dict) else "",
        )
    ).lower()
    return any(marker in label for marker in ("adversarial", "unanswerable", "unanswerable/unknown"))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = answer_tokens(prediction)
    gold_tokens = answer_tokens(ground_truth)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum(min(pred_counts[token], gold_counts[token]) for token in pred_counts.keys() & gold_counts.keys())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall_value = overlap / len(gold_tokens)
    return 2 * precision * recall_value / (precision + recall_value)


def bleu1(prediction: str, ground_truth: str) -> float:
    pred_tokens = answer_tokens(prediction)
    gold_tokens = answer_tokens(ground_truth)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum(min(pred_counts[token], gold_counts[token]) for token in pred_counts)
    precision = overlap / len(pred_tokens)
    brevity_penalty = 1.0 if len(pred_tokens) > len(gold_tokens) else pow(2.718281828, 1 - len(gold_tokens) / len(pred_tokens))
    return brevity_penalty * precision


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def extract_dates(text: Any) -> set[str]:
    raw = normalize_text(text)
    if not raw:
        return set()
    dates: set[str] = set()
    for year, month, day in re.findall(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", raw):
        add_valid_date(dates, int(year), int(month), int(day))
    for day, month_name, year in re.findall(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})\b",
        raw,
        flags=re.IGNORECASE,
    ):
        month = MONTHS.get(month_name.lower())
        if month:
            add_valid_date(dates, int(year), month, int(day))
    for month_name, day, year in re.findall(
        r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
        raw,
        flags=re.IGNORECASE,
    ):
        month = MONTHS.get(month_name.lower())
        if month:
            add_valid_date(dates, int(year), month, int(day))
    return dates


def add_valid_date(output: set[str], year: int, month: int, day: int) -> None:
    try:
        output.add(date(year, month, day).isoformat())
    except ValueError:
        return


def answer_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalize_for_match(text))


def normalize_for_match(text: Any) -> str:
    normalized = normalize_text(text).lower()
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9']+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def recall(gold_ids: list[str], predicted_ids: list[str]) -> float:
    gold = normalized_id_set(gold_ids)
    if not gold:
        return 0.0
    predicted = normalized_id_set(predicted_ids)
    return len(gold & predicted) / len(gold)


def complete_recall(gold_ids: list[str], predicted_ids: list[str]) -> bool:
    gold = normalized_id_set(gold_ids)
    if not gold:
        return False
    predicted = normalized_id_set(predicted_ids)
    return gold.issubset(predicted)


def hit(gold_ids: list[str], predicted_ids: list[str]) -> bool:
    gold = normalized_id_set(gold_ids)
    predicted = normalized_id_set(predicted_ids)
    return bool(gold & predicted)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _strip_reasoning_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def canonicalize_answer_for_eval(text: Any) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    cleaned = _strip_reasoning_tags(cleaned)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(
        r"^(final\s+answer|answer|response|hypothesis|prediction)\s*[:\-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = cleaned.strip(" \t\r\n'\"`")
    cleaned = re.sub(r"^\s*[\-\*\u2022]\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def latest_rows_by_sample_system(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (str(row.get("sample_id") or ""), str(row.get("system") or ""))
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order]


def latest_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (str(row.get("question_id") or row.get("sample_id") or ""), str(row.get("system") or ""))
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order]


def prediction_row(sample: UnifiedSample, row: dict[str, Any]) -> dict[str, Any]:
    hypothesis = prediction_hypothesis(row)
    if sample.dataset == "longmemeval":
        return {
            "question_id": sample.metadata.get("question_id") or sample.sample_id,
            "system": row["system"],
            "hypothesis": hypothesis,
        }
    return {"sample_id": sample.sample_id, "system": row["system"], "hypothesis": hypothesis}


def prediction_row_by_sample_id(samples: list[UnifiedSample], row: dict[str, Any]) -> dict[str, Any]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    sample = sample_by_id.get(str(row.get("sample_id")))
    if sample is not None:
        return prediction_row(sample, row)
    return {"sample_id": row.get("sample_id"), "system": row["system"], "hypothesis": prediction_hypothesis(row)}


def prediction_hypothesis(row: dict[str, Any]) -> str:
    hypothesis = str(row.get("pred_answer") or "").strip()
    if hypothesis:
        return hypothesis
    if row.get("error"):
        return ERROR_PRED_ANSWER
    return ""


def print_qa_result(row: dict[str, Any], max_chars: int = 260) -> None:
    status = "ok" if row.get("answer_correct") else "miss"
    if row.get("error"):
        status = "error"
    prefix = (
        f"[qa-result] system={row.get('system')} sample={row.get('sample_id')} "
        f"correct={row.get('answer_correct')} status={status}"
    )
    print(prefix, flush=True)
    print(f"  Q: {compact_for_console(row.get('question'), max_chars)}", flush=True)
    print(f"  G: {compact_for_console(row.get('gold_answer'), max_chars)}", flush=True)
    print(f"  P: {compact_for_console(row.get('pred_answer'), max_chars)}", flush=True)


def compact_for_console(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def artifact_root() -> Path | None:
    value = os.getenv("MEMORY_EVAL_OUTPUT_DIR", "").strip()
    return Path(value) / "artifacts" if value else None


def artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def persist_build_artifact(adapter: Any, sample: UnifiedSample, conversation_id: str) -> None:
    root = artifact_root()
    if root is None:
        return
    debug = adapter.dump_debug_state()
    organization = dict(debug.get("organization_state") or {})
    organization_entries = organization.pop("entries", [])
    payload = {
        "artifact_type": "conversation_memory_build",
        "dataset": sample.dataset,
        "system": adapter.system_name,
        "conversation_id": conversation_id,
        "memory_entries": debug.get("memory_entries") or organization_entries,
        "organization_state": organization,
        "build_trace": {
            key: value
            for key, value in debug.items()
            if key not in {"memory_entries", "organization_state", "retrieval", "prompt_record", "answer_record"}
        },
    }
    path = root / sample.dataset / adapter.system_name / "build" / f"{artifact_name(conversation_id)}.json"
    write_json(path, payload)


def persist_sample_artifact(adapter: Any, sample: UnifiedSample, *, include_memory: bool) -> None:
    root = artifact_root()
    if root is None:
        return
    debug = adapter.dump_debug_state()
    if not include_memory:
        debug.pop("memory_entries", None)
        debug.pop("organization_state", None)
    payload = {
        "artifact_type": "sample_pipeline_trace",
        "dataset": sample.dataset,
        "system": adapter.system_name,
        "sample_id": sample.sample_id,
        **debug,
    }
    path = root / sample.dataset / adapter.system_name / "samples" / f"{artifact_name(sample.sample_id)}.json"
    write_json(path, payload)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(to_jsonable(row))


def trace_path(output_dir: Path, dataset: str, system_name: str) -> Path:
    return output_dir / "traces" / dataset / f"{system_name}.jsonl"


def append_trace(output_dir: Path, dataset: str, system_name: str, row: dict[str, Any]) -> None:
    append_jsonl(trace_path(output_dir, dataset, system_name), trace_row(row))


def write_trace_files(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset") or ""), str(row.get("system") or ""))].append(trace_row(row))
    for (dataset, system_name), trace_rows in grouped.items():
        if dataset and system_name:
            write_jsonl(trace_path(output_dir, dataset, system_name), trace_rows)


def trace_row(row: dict[str, Any]) -> dict[str, Any]:
    answer_metrics = {
        "answer_exact_accuracy": row.get("answer_exact_accuracy"),
        "normalized_answer_accuracy": row.get("normalized_answer_accuracy"),
        "date_answer_accuracy": row.get("date_answer_accuracy"),
        "answer_correct": row.get("answer_correct"),
        "token_f1": row.get("token_f1", row.get("answer_f1")),
        "bleu_1": row.get("bleu_1"),
    }
    retrieval_metrics = {
        "has_gold_evidence": row.get("has_gold_evidence"),
        "retrieval_recall": row.get("retrieval_recall"),
        "retrieval_hit": row.get("retrieval_hit"),
        "complete_evidence_recall": row.get("complete_evidence_recall"),
        "source_mapping_available": row.get("retrieval_source_mapping_available"),
    }
    prompt_metrics = {
        "prompt_gold_coverage": row.get("prompt_gold_coverage"),
        "prompt_hit": row.get("prompt_hit"),
        "prompt_complete_evidence_recall": row.get("prompt_complete_evidence_recall"),
        "injection_loss": row.get("injection_loss"),
        "source_mapping_available": row.get("prompt_source_mapping_available"),
    }
    post_evidence_metrics = {
        "gold_in_prompt": row.get("gold_in_prompt"),
        "accuracy_given_gold_in_prompt": row.get("accuracy_given_gold_in_prompt"),
        "post_evidence_answer_failure": row.get("post_evidence_answer_failure"),
    }
    return {
        "sample_id": row.get("sample_id"),
        "dataset": row.get("dataset"),
        "original_category": row.get("original_category") or row.get("question_type"),
        "question_type": row.get("original_question_type") or row.get("question_type"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "gold_evidence_units": row.get("gold_evidence_units") or [],
        "system": row.get("system"),
        "retrieved_evidence_source_ids": row.get("retrieved_source_ids") or [],
        "retrieved_gold_ids": row.get("retrieved_gold_ids") or [],
        "missing_retrieved_gold_ids": row.get("missing_retrieved_gold_ids") or [],
        "prompt_evidence_source_ids": row.get("injected_source_ids") or [],
        "prompt_gold_ids": row.get("prompt_gold_ids") or [],
        "missing_prompt_gold_ids": row.get("missing_prompt_gold_ids") or [],
        "source_mapping_available": row.get("source_mapping_available"),
        "final_answer": row.get("pred_answer"),
        "normalized_answer": row.get("normalized_answer"),
        "answer_metrics": answer_metrics,
        "retrieval_metrics": retrieval_metrics,
        "prompt_injection_metrics": prompt_metrics,
        "post_evidence_answer_metrics": post_evidence_metrics,
        "latency": row.get("latency"),
        "token_usage": row.get("token_usage") or {},
        "answer_raw_response": row.get("answer_raw_response") or {},
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "total_tokens": row.get("total_tokens"),
        "failure_stage": row.get("failure_stage"),
        "failure_type": row.get("failure_type"),
        "error": row.get("error"),
    }


def write_longmemeval_predictions(output_dir: Path, prediction_rows: list[dict[str, Any]]) -> None:
    by_system: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        by_system[str(row["system"])].append(
            {"question_id": str(row["question_id"]), "hypothesis": str(row["hypothesis"])}
        )
    for system, rows in by_system.items():
        write_jsonl(output_dir / f"longmemeval_predictions_{system}.jsonl", rows)


def write_stage2_reports(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_grouped_report(output_dir / "overall_performance.csv", rows, ["system"])
    write_grouped_report(output_dir / "category_performance.csv", rows, ["system", "original_category"])
    write_grouped_report(output_dir / "stagewise_evidence_flow.csv", rows, ["system", "original_category"])
    write_failure_distribution(output_dir / "failure_distribution.csv", rows)
    write_efficiency_report(output_dir / "efficiency.csv", rows)


def write_grouped_report(path: Path, rows: list[dict[str, Any]], group_keys: list[str]) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    out_rows: list[dict[str, Any]] = []
    for group, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        out = {key: value for key, value in zip(group_keys, group)}
        out.update(aggregate(group_rows))
        out_rows.append(out)
    write_csv(path, out_rows)


def write_failure_distribution(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("system") or ""),
                str(row.get("original_category") or row.get("question_type") or ""),
                str(row.get("failure_type") or "unknown_failure"),
            )
        ].append(row)
    total_by_group: dict[tuple[str, str], int] = defaultdict(int)
    for system, category, _failure_type in grouped:
        total_by_group[(system, category)] += len(grouped[(system, category, _failure_type)])
    out_rows = []
    for (system, category, failure_type), group_rows in sorted(grouped.items()):
        denominator = total_by_group[(system, category)]
        out_rows.append(
            {
                "system": system,
                "original_category": category,
                "failure_type": failure_type,
                "failure_stage": failure_stage(failure_type),
                "count": len(group_rows),
                "rate": len(group_rows) / denominator if denominator else 0.0,
            }
        )
    write_csv(path, out_rows)


def write_efficiency_report(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("system") or ""), str(row.get("original_category") or row.get("question_type") or ""))].append(row)
    out_rows: list[dict[str, Any]] = []
    for (system, category), group_rows in sorted(grouped.items()):
        out_rows.append(
            {
                "system": system,
                "original_category": category,
                "num_samples": len(group_rows),
                "avg_latency": mean(row.get("latency") for row in group_rows),
                "avg_prompt_tokens": mean(row.get("prompt_tokens") for row in group_rows),
                "avg_completion_tokens": mean(row.get("completion_tokens") for row in group_rows),
                "avg_total_tokens": mean(row.get("total_tokens") for row in group_rows),
                "error_rate": mean(bool(row.get("error")) for row in group_rows),
            }
        )
    write_csv(path, out_rows)


def write_summary(path: Path, rows: list[dict[str, Any]], group_keys: list[str]) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    fieldnames = group_keys + [
        "num_samples",
        "num_gold_evidence_samples",
        "answer_exact_accuracy",
        "normalized_answer_accuracy",
        "date_answer_accuracy",
        "answer_accuracy",
        "avg_token_f1",
        "avg_bleu_1",
        "avg_retrieval_recall",
        "retrieval_hit_rate",
        "complete_evidence_recall_rate",
        "avg_prompt_gold_coverage",
        "prompt_hit_rate",
        "prompt_complete_evidence_recall_rate",
        "avg_injection_loss",
        "accuracy_given_gold_in_prompt",
        "post_evidence_answer_failure_rate",
        "retrieval_failure_rate",
        "injection_failure_rate",
        "utilization_failure_rate",
        "partial_evidence_failure_rate",
        "abstention_or_hallucination_failure_rate",
        "success_rate",
        "avg_latency",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_total_tokens",
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
        "num_gold_evidence_samples": sum(1 for row in rows if row.get("has_gold_evidence")),
        "answer_exact_accuracy": mean(row.get("answer_exact_accuracy") for row in rows),
        "normalized_answer_accuracy": mean(row.get("normalized_answer_accuracy") for row in rows),
        "date_answer_accuracy": mean(row.get("date_answer_accuracy") for row in rows),
        "answer_accuracy": mean(row["answer_correct"] for row in rows),
        "avg_token_f1": mean(row.get("token_f1", row.get("answer_f1")) for row in rows),
        "avg_bleu_1": mean(row.get("bleu_1") for row in rows),
        "avg_retrieval_recall": mean(row.get("retrieval_recall") for row in rows),
        "retrieval_hit_rate": mean(row.get("retrieval_hit") for row in rows),
        "complete_evidence_recall_rate": mean(row.get("complete_evidence_recall") for row in rows),
        "avg_prompt_gold_coverage": mean(row.get("prompt_gold_coverage") for row in rows),
        "prompt_hit_rate": mean(row.get("prompt_hit") for row in rows),
        "prompt_complete_evidence_recall_rate": mean(row.get("prompt_complete_evidence_recall") for row in rows),
        "avg_injection_loss": mean(row.get("injection_loss") for row in rows),
        "accuracy_given_gold_in_prompt": mean(row.get("accuracy_given_gold_in_prompt") for row in rows),
        "post_evidence_answer_failure_rate": mean(row.get("post_evidence_answer_failure") for row in rows),
        "retrieval_failure_rate": failure_rate(rows, "retrieval_failure"),
        "injection_failure_rate": failure_rate(rows, "injection_failure"),
        "utilization_failure_rate": failure_rate(rows, "post_evidence_answer_failure"),
        "partial_evidence_failure_rate": failure_rate(rows, "partial_evidence_failure"),
        "abstention_or_hallucination_failure_rate": failure_rate(rows, "abstention_or_hallucination_failure"),
        "success_rate": failure_rate(rows, "success"),
        "avg_latency": mean(row.get("latency") for row in rows),
        "avg_prompt_tokens": mean(row.get("prompt_tokens") for row in rows),
        "avg_completion_tokens": mean(row.get("completion_tokens") for row in rows),
        "avg_total_tokens": mean(row.get("total_tokens") for row in rows),
        "error_rate": mean(bool(row.get("error")) for row in rows),
    }


def mean(values: Any) -> float:
    numeric = [float(value) for value in values if value is not None and value != ""]
    return sum(numeric) / len(numeric) if numeric else 0.0


def failure_rate(rows: list[dict[str, Any]], failure_type: str) -> float:
    return sum(1 for row in rows if row.get("failure_type") == failure_type) / len(rows) if rows else 0.0


if __name__ == "__main__":
    main()
