from __future__ import annotations

import os
import hashlib
import gc
import json
import math
import re
import shutil
import time
import warnings
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

_MEM0_ROOT = Path(__file__).resolve().parents[1] / "storage" / "mem0_sdk" / f"process_{os.getpid()}_{uuid.uuid4().hex[:8]}"
_MEM0_ROOT.mkdir(parents=True, exist_ok=True)
# Mem0 captures MEM0_DIR at module import for its migration/config store.
# Set it before importing the SDK so it never falls back to a shared ~/.mem0.
os.environ.setdefault("MEM0_DIR", str(_MEM0_ROOT))

from mem0 import Memory

from map_platform.datasets.schema import UnifiedSample
from map_platform.memory_systems.base import (
    AnswerRecord,
    BaseMemorySystemAdapter,
    MemoryEntry,
    OrganizationState,
    PromptRecord,
    RetrievalResult,
    RetrievedMemory,
)
from map_platform.llm.gateway import (
    embed_texts,
    embedding_base_url,
    embedding_sdk_config,
    openai_base_url,
    openai_sdk_config,
    truncate_embedding_inputs_for_context_error,
    truncate_text_for_embedding,
)
from map_platform.utils.serialization import to_jsonable
from map_platform.utils.live_progress import update_live_progress


MEM0_OFFICIAL_CUSTOM_INSTRUCTIONS = """Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""

MEM0_OFFICIAL_ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {speaker_1}:

    {speaker_1_memories}

    Memories for user {speaker_2}:

    {speaker_2_memories}

    Question: {question}

    Answer:
    """


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _embedding_dims(model_name: str) -> int:
    override = os.getenv("MEM0_EMBED_DIMS") or os.getenv("EMBEDDING_DIMS")
    if override and override.isdigit():
        return int(override)
    normalized = model_name.lower()
    if "bge-m3" in normalized:
        return 1024
    if "qwen3-embedding-0.6b" in normalized:
        return 1024
    if "qwen3-embedding-4b" in normalized:
        return 2560
    if "qwen3-embedding-8b" in normalized:
        return 4096
    if "3-large" in normalized:
        return 3072
    return 1536


def _as_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return dict(payload)
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if hasattr(payload, "dict"):
        return payload.dict()
    return {"value": payload}


def _coerce_result_list(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [_as_dict(item) for item in payload]
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [_as_dict(item) for item in results]
        if "id" in payload or "memory" in payload or "data" in payload:
            return [_as_dict(payload)]
    return [_as_dict(payload)]


def _extract_content(raw: dict[str, Any]) -> str:
    return str(raw.get("memory") or raw.get("data") or raw.get("content") or "").strip()


def _extract_source_ids(raw: dict[str, Any]) -> list[str]:
    metadata = raw.get("metadata") or {}
    source_ids = (
        metadata.get("source_turn_ids")
        or metadata.get("source_ids")
        or metadata.get("source_batch_turn_ids")
        or []
    )
    if isinstance(source_ids, list):
        max_ids = int(os.getenv("MEM0_MAX_EXPLICIT_SOURCE_IDS", "10"))
        if len(source_ids) > max_ids:
            source_ids = source_ids[:max_ids]
        return [str(item) for item in source_ids if str(item).strip()]
    if source_ids:
        return [str(source_ids)]
    return []


def _flat_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    flattened = dict(metadata)
    for source_key, target_key in (
        ("timestamp", "session_date"),
        ("created_at", "created_at"),
        ("updated_at", "updated_at"),
        ("session_id", "session_id"),
        ("speaker", "speaker"),
    ):
        value = metadata.get(source_key)
        if value and target_key not in flattened:
            flattened[target_key] = value
    return flattened


def _is_truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _mem0_llm_max_tokens() -> int:
    return int(os.getenv("MEM0_LLM_MAX_TOKENS", "1024"))


def _strip_reasoning_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def _clean_llm_text(text: str) -> str:
    cleaned = _strip_reasoning_tags(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def _is_gpt5_like(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _token_param_name(model: str) -> str:
    override = os.getenv("MEM0_LLM_TOKEN_PARAM", "").strip()
    if override:
        return override
    return "max_completion_tokens" if _is_gpt5_like(model) else "max_tokens"


def _json_mode_disabled() -> bool:
    # selected provider currently returns intermittent 502 for response_format=json_object.
    # Keep local JSON parsing, but use prompt-constrained JSON at this gateway.
    base_url = (
        os.getenv("MEM0_LLM_BASE_URL", "").strip()
        or os.getenv("MEMORY_BUILD_OPENAI_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
    ).lower()
    return _is_truthy_env("MEM0_DISABLE_JSON_MODE", "false") or "selected provider" in base_url


def _mem0_debug_stdout_enabled() -> bool:
    return _is_truthy_env("MEM0_DEBUG_STDOUT", "false")


def _direct_extraction_enabled() -> bool:
    return _is_truthy_env("MEM0_USE_DIRECT_EXTRACTION", "false")


def _raw_turn_fallback_top_k() -> int:
    return max(0, int(os.getenv("MEM0_RAW_TURN_FALLBACK_TOP_K", "0")))


def _expand_retrieved_sources_enabled() -> bool:
    return _is_truthy_env("MEM0_EXPAND_RETRIEVED_SOURCE_TURNS", "false")


def _debug_mem0_llm_response(_llm: Any, _response: Any, params: dict[str, Any]) -> None:
    if not _mem0_debug_stdout_enabled():
        return
    messages = params.get("messages") or []
    summary = {
        "model": params.get("model"),
        "response_format": params.get("response_format"),
        "temperature": params.get("temperature"),
        "max_tokens": params.get("max_tokens"),
        "num_messages": len(messages),
        "message_roles": [item.get("role") for item in messages if isinstance(item, dict)],
        "message_contains_json": [
            "json" in str(item.get("content", "")).lower()
            for item in messages
            if isinstance(item, dict)
        ],
    }
    print(f"[mem0-llm-debug] {summary}", flush=True)


def _token_counts(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9']+", text.lower()))


def _lexical_score(query: str, content: str) -> float:
    query_counts = _token_counts(query)
    content_counts = _token_counts(content)
    if not query_counts or not content_counts:
        return 0.0
    overlap = sum(min(query_counts[token], content_counts[token]) for token in query_counts)
    return overlap / max(sum(query_counts.values()), 1)


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


class Mem0Adapter(BaseMemorySystemAdapter):
    system_name = "mem0"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._memory: Memory | None = None
        self._storage_dir = Path(
            os.getenv(
                "MEM0_STORAGE_DIR",
                str(Path(__file__).resolve().parents[1] / "storage" / "diagnostic_mem0"),
            )
        ).expanduser().resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = ""
        self._sample_dir = self._storage_dir
        self._sdk_provider = self._resolve_sdk_provider()
        self._retrieval_embedding_cache: dict[str, list[float]] = {}
        self._speaker_user_ids: list[tuple[str, str]] = []

    def reset(self, sample_id: str) -> None:
        self.close()
        super().reset(sample_id)
        self._retrieval_embedding_cache = {}
        self._ensure_provider_credentials()
        embed_hash = hashlib.sha1(self.embedding_model.encode("utf-8")).hexdigest()[:10]
        embed_dims = _embedding_dims(self.embedding_model)
        self._collection_name = f"diag_{_safe_component(sample_id)}_d{embed_dims}_{embed_hash}"
        # Qdrant's embedded store permits only one client per directory. The
        # evaluator may construct more than one adapter object in a process
        # during preflight/resume, so keep each adapter instance isolated.
        instance_root = self._storage_dir / f"adapter_{os.getpid()}_{id(self)}"
        sample_dir = instance_root / self._collection_name
        self._sample_dir = sample_dir
        # Keep the local Qdrant/history checkpoint across retries. A clean build
        # remains available through MEM0_RESET_SAMPLE_STORAGE=true.
        if sample_dir.exists() and os.getenv("MEM0_RESET_SAMPLE_STORAGE", "false").strip().lower() not in {"0", "false", "no"}:
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        sdk_dir = sample_dir / "mem0_sdk"
        sdk_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MEM0_DIR"] = str(sdk_dir)
        self._memory = Memory.from_config(
            {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": str(sample_dir / "qdrant"),
                        "collection_name": self._collection_name,
                        "embedding_model_dims": embed_dims,
                    },
                },
                "llm": {
                    "provider": self._sdk_provider,
                    "config": self._mem0_llm_config()
                    if self._sdk_provider == "openai"
                    else {"model": self.generation_model, "temperature": 0.0},
                },
                "embedder": {
                    "provider": self._sdk_provider,
                    "config": self._mem0_embedder_config()
                    if self._sdk_provider == "openai"
                    else {"model": self.embedding_model},
                },
                "history_db_path": str(sample_dir / "history.db"),
                "version": "v1.1",
                "custom_instructions": os.getenv(
                    "MEM0_EXTRACTION_PROMPT",
                    MEM0_OFFICIAL_CUSTOM_INSTRUCTIONS,
                ),
            }
        )
        self._patch_mem0_llm_client()
        self._patch_mem0_embedding_client()
        try:
            self._memory.delete_all(user_id=sample_id)
        except Exception as exc:
            warnings.warn(f"delete_all failed for {sample_id}: {exc}", RuntimeWarning, stacklevel=2)
        self._debug_state["sdk"] = {
            "provider": self._sdk_provider,
            "collection_name": self._collection_name,
            "embedding_model": self.embedding_model,
            "embedding_dims": embed_dims,
            "reset_sample_storage": os.getenv("MEM0_RESET_SAMPLE_STORAGE", "true"),
            "storage_dir": sample_dir,
            "mem0_dir": sdk_dir,
        }

    def close(self) -> None:
        if self._memory is None:
            return
        self._close_sdk_object(self._memory)
        self._memory = None
        gc.collect()

    def _close_sdk_object(self, obj: Any, *, seen: set[int] | None = None, depth: int = 0) -> None:
        if obj is None or depth > 4:
            return
        seen = seen or set()
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        for method_name in ("close", "disconnect"):
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
        for attr_name in (
            "vector_store",
            "vector_store_client",
            "client",
            "_client",
            "qdrant_client",
            "_qdrant_client",
            "db",
            "_db",
        ):
            try:
                child = getattr(obj, attr_name)
            except Exception:
                continue
            self._close_sdk_object(child, seen=seen, depth=depth + 1)

    def build_memory(self, history: list[dict], sample: UnifiedSample) -> None:
        if self._memory is None:
            self.reset(sample.sample_id)
        assert self._memory is not None
        messages = self._history_as_sourced_messages(history)
        batch_size = max(int(os.getenv("MEM0_BATCH_TURNS", "2")), 1)
        build_responses: list[dict[str, Any]] = []
        speakers = list(dict.fromkeys(str(turn.get("speaker") or "").strip() for turn in history if turn.get("speaker")))
        session_sizes = Counter(str(turn.get("session_id") or "session") for turn in history)
        total_batches = len(speakers) * sum(
            (session_size + batch_size - 1) // batch_size
            for session_size in session_sizes.values()
        )
        self._debug_state["build_memory_config"] = {
            "sample_id": sample.sample_id,
            "batch_size": batch_size,
            "num_messages": len(messages),
            "total_batches": total_batches,
            "completed_batches": 0,
        }
        update_live_progress(
            stage="build_memory", system=self.system_name, sample_id=sample.sample_id,
            current_operation="mem0_sdk_add", total_batches=total_batches,
            completed_batches=0, memory_entries=0, history_turns=len(history),
        )
        build_batches: list[dict[str, Any]] = []
        self._debug_state["build_memory_batches"] = build_batches
        checkpoint_path = self._sample_dir / "build_checkpoint.json"
        completed_batch_keys: set[str] = set()
        if checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if (
                    checkpoint.get("sample_id") == sample.sample_id
                    and int(checkpoint.get("batch_size", -1)) == batch_size
                ):
                    completed_batch_keys = {
                        str(item) for item in checkpoint.get("completed_batch_keys", [])
                    }
            except (OSError, ValueError, TypeError):
                completed_batch_keys = set()

        def persist_checkpoint() -> None:
            payload = {
                "sample_id": sample.sample_id,
                "batch_size": batch_size,
                "total_batches": total_batches,
                "completed_batch_keys": sorted(completed_batch_keys),
            }
            tmp = checkpoint_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, checkpoint_path)
        if _mem0_debug_stdout_enabled():
            print(
                "[mem0-build-config] "
                + str(self._debug_state["build_memory_config"]),
                flush=True,
            )
        if _direct_extraction_enabled():
            self._memory_entries = self._direct_extract_entries(history=history, sample=sample, batch_size=batch_size)
            self._debug_state["build_memory_mode"] = {
                "infer": True,
                "batch_size": batch_size,
                "implementation_mode": "direct_openai_json_extraction",
                "sdk_error": None,
            }
            self._debug_state["build_memory"] = {
                "implementation_mode": "direct_openai_json_extraction",
                "sdk_error": None,
                "num_entries": len(self._memory_entries),
            }
            self._org_state = OrganizationState(
                entries=list(self._memory_entries),
                levels=None,
                weights=None,
                priorities=None,
                raw={
                    "mode": "flat_vector_memory",
                    "sdk": "mem0ai",
                    "implementation_mode": "direct_openai_json_extraction",
                    "sdk_error": None,
                },
            )
            return
        active_batch_record: dict[str, Any] | None = None
        implementation_mode = "sdk"
        sdk_error = None
        try:
            speakers = list(dict.fromkeys(str(turn.get("speaker") or "").strip() for turn in history if turn.get("speaker")))
            if len(speakers) != 2:
                raise RuntimeError(f"Mem0 official contract requires exactly two speakers, found {speakers}")
            self._speaker_user_ids = [(speaker, f"{self._sample_id}:{_safe_component(speaker)}") for speaker in speakers]
            if _is_truthy_env("MEM0_REUSE_EXISTING_MEMORY", "false"):
                self._memory_entries = self._load_entries()
                if not self._memory_entries:
                    raise RuntimeError(
                        "MEM0_REUSE_EXISTING_MEMORY=true but the configured store contains no memories"
                    )
                implementation_mode = "sdk_reused_existing_memory"
                self._debug_state["build_memory_mode"] = {
                    "infer": True,
                    "batch_size": batch_size,
                    "implementation_mode": implementation_mode,
                    "sdk_error": None,
                    "reused_existing_memory": True,
                }
                self._debug_state["build_memory"] = {
                    "implementation_mode": implementation_mode,
                    "sdk_error": None,
                    "num_entries": len(self._memory_entries),
                    "reused_existing_memory": True,
                }
                self._org_state = OrganizationState(
                    entries=list(self._memory_entries),
                    levels=None,
                    weights=None,
                    priorities=None,
                    raw={
                        "mode": "flat_vector_memory",
                        "sdk": "mem0ai",
                        "implementation_mode": implementation_mode,
                        "sdk_error": None,
                        "reused_existing_memory": True,
                    },
                )
                return
            sessions: dict[str, list[dict[str, Any]]] = {}
            for turn in history:
                sessions.setdefault(str(turn.get("session_id") or "session"), []).append(turn)
            batch_number = 0
            for session_id, session_turns in sessions.items():
                for speaker, user_id in self._speaker_user_ids:
                    perspective = [
                        {
                            "role": "user" if str(turn.get("speaker") or "") == speaker else "assistant",
                            "content": f"{turn.get('speaker')}: {turn.get('text') or turn.get('content') or ''}",
                        }
                        for turn in session_turns
                    ]
                    for start in range(0, len(perspective), batch_size):
                        batch = perspective[start : start + batch_size]
                        batch_turns = session_turns[start : start + batch_size]
                        if not batch:
                            continue
                        batch_number += 1
                        batch_key = f"{session_id}|{speaker}|{start}|{len(batch)}"
                        if batch_key in completed_batch_keys:
                            self._debug_state["build_memory_config"]["completed_batches"] = len(completed_batch_keys)
                            update_live_progress(
                                stage="build_memory", system=self.system_name,
                                sample_id=sample.sample_id, session_id=session_id,
                                speaker=speaker, current_operation="mem0_checkpoint_skip",
                                batch=batch_number, total_batches=total_batches,
                                completed_batches=len(completed_batch_keys),
                                memory_entries=len(completed_batch_keys),
                            )
                            continue
                        batch_record: dict[str, Any] = {
                            "batch": batch_number,
                            "total_batches": total_batches,
                            "num_messages": len(batch),
                            "start": start,
                            "session_id": session_id,
                            "speaker": speaker,
                            "status": "started",
                        }
                        active_batch_record = batch_record
                        build_batches.append(batch_record)
                        update_live_progress(
                            stage="build_memory", system=self.system_name,
                            sample_id=sample.sample_id, session_id=session_id,
                            speaker=speaker, current_operation="mem0_sdk_add",
                            batch=batch_number, total_batches=total_batches,
                            completed_batches=batch_number - 1,
                            memory_entries=len(build_responses),
                        )
                        metadata = {
                                "sample_id": sample.sample_id,
                                "memory_user_id": user_id,
                                "dataset": sample.dataset,
                                "task_type": sample.task_type,
                                "timestamp": batch_turns[0].get("timestamp") if batch_turns else None,
                                "session_id": session_id,
                                "source_batch_turn_ids": [
                                    str(turn.get("turn_id") or f"turn_{start + index + 1}")
                                    for index, turn in enumerate(batch_turns)
                                ],
                                "source_batch_start": start,
                                "source_batch_size": len(batch_turns),
                            }
                        build_response = self._add_with_observation_timestamp(
                            batch,
                            user_id=user_id,
                            metadata=metadata,
                        )
                        jsonable_response = to_jsonable(_as_dict(build_response))
                        build_responses.append(jsonable_response)
                        completed_batch_keys.add(batch_key)
                        persist_checkpoint()
                        batch_record["status"] = "ok"
                        self._debug_state["build_memory_config"]["completed_batches"] = len(completed_batch_keys)
                        update_live_progress(
                            stage="build_memory", system=self.system_name,
                            sample_id=sample.sample_id, session_id=session_id,
                            speaker=speaker, current_operation="mem0_sdk_add",
                            batch=batch_number, total_batches=total_batches,
                            completed_batches=len(completed_batch_keys),
                            memory_entries=len(completed_batch_keys),
                        )
                        batch_record["response_preview"] = str(jsonable_response)[:500]
            self._memory_entries = self._load_entries()
        except Exception as exc:
            if active_batch_record is not None:
                active_batch_record["status"] = "error"
                active_batch_record["error"] = f"{type(exc).__name__}: {exc}"
            if os.getenv("MEM0_ALLOW_LOCAL_FALLBACK", "false").strip().lower() in {"0", "false", "no"}:
                raise
            implementation_mode = "local_fallback_after_sdk_error"
            sdk_error = f"{type(exc).__name__}: {exc}"
            self._memory_entries = self._fallback_entries(history)
        if (
            history
            and not self._memory_entries
            and os.getenv("MEMORY_EVAL_FORMAL_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
        ):
            raise RuntimeError("Mem0 completed memory construction but produced no memory entries")
        self._debug_state["build_memory_mode"] = {
            "infer": True,
            "batch_size": batch_size,
            "implementation_mode": implementation_mode,
            "sdk_error": sdk_error,
        }
        self._debug_state["build_memory_response"] = build_responses
        self._debug_state["build_memory"] = {
            "implementation_mode": implementation_mode,
            "sdk_error": sdk_error,
            "num_entries": len(self._memory_entries),
        }
        self._org_state = OrganizationState(
            entries=list(self._memory_entries),
            levels=None,
            weights=None,
            priorities=None,
            raw={
                "mode": "flat_vector_memory",
                "sdk": "mem0ai",
                "implementation_mode": implementation_mode,
                "sdk_error": sdk_error,
            }
        )

    def _add_with_observation_timestamp(
        self,
        messages: list[dict[str, Any]],
        *,
        user_id: str,
        metadata: dict[str, Any],
    ) -> Any:
        """Restore the hosted-v2 metadata timestamp binding in local mem0 2.x."""
        assert self._memory is not None
        import mem0.memory.main as mem0_main

        observation_timestamp = metadata.get("timestamp")
        original_generate_prompt = mem0_main.generate_additive_extraction_prompt

        def generate_prompt_with_timestamp(*args: Any, **kwargs: Any) -> str:
            kwargs["timestamp"] = observation_timestamp
            return original_generate_prompt(*args, **kwargs)

        mem0_main.generate_additive_extraction_prompt = generate_prompt_with_timestamp
        self._debug_state.setdefault("temporal_bindings", []).append(
            {
                "user_id": user_id,
                "session_id": metadata.get("session_id"),
                "observation_timestamp": observation_timestamp,
                "source": "official_metadata_timestamp",
                "local_compatibility_shim": True,
            }
        )
        try:
            return self._memory.add(
                messages,
                user_id=user_id,
                infer=True,
                metadata=metadata,
            )
        finally:
            mem0_main.generate_additive_extraction_prompt = original_generate_prompt

    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        assert self._memory is not None
        implementation_mode = self._debug_state.get("build_memory", {}).get("implementation_mode")
        if implementation_mode == "direct_openai_json_extraction":
            retrieved = self._direct_retrieve(query=query, top_k=top_k)
            self._retrieval_result = RetrievalResult(
                query=query,
                retrieved_entries=retrieved,
                top_k=top_k,
                raw={
                    "sdk": "mem0ai",
                    "implementation_mode": implementation_mode,
                    "retrieval_mode": os.getenv("MEM0_DIRECT_RETRIEVAL_MODE", "hybrid_vector"),
                },
            )
            return self._retrieval_result
        if implementation_mode == "local_fallback_after_sdk_error":
            retrieved = self._fallback_retrieve(query=query, top_k=top_k)
            self._retrieval_result = RetrievalResult(
                query=query,
                retrieved_entries=retrieved,
                top_k=top_k,
                raw={
                    "sdk": "mem0ai",
                    "implementation_mode": implementation_mode,
                    "sdk_error": self._debug_state.get("build_memory", {}).get("sdk_error"),
                },
            )
            return self._retrieval_result
        per_speaker_top_k = int(os.getenv("MEM0_OFFICIAL_TOP_K", "30"))
        raw_items: list[dict[str, Any]] = []
        for speaker, user_id in self._speaker_user_ids:
            raw_results = self._memory.search(query, top_k=per_speaker_top_k, filters={"user_id": user_id}, threshold=0.0)
            for item in _coerce_result_list(raw_results):
                item["_official_speaker"] = speaker
                raw_items.append(item)
        retrieved: list[RetrievedMemory] = []
        for rank, item in enumerate(raw_items, start=1):
            retrieved.append(
                RetrievedMemory(
                    entry_id=str(item.get("id") or f"mem0_{rank}"),
                    content=_extract_content(item),
                    score=float(item["score"]) if item.get("score") is not None else None,
                    rank=rank,
                    source_ids=_extract_source_ids(item),
                    metadata={
                        **_flat_metadata(item.get("metadata") or {}),
                        "mem0_event": item.get("event"),
                        "official_speaker": item.get("_official_speaker"),
                    },
                )
            )
        retrieved = self._augment_with_raw_turn_fallback(
            retrieved=retrieved,
            query=query,
            sample=sample,
            top_k=_raw_turn_fallback_top_k(),
        )
        self._retrieval_result = RetrievalResult(
            query=query,
            retrieved_entries=retrieved,
            top_k=len(retrieved),
            raw={
                "sdk": "mem0ai",
                "results": raw_items,
                "raw_turn_fallback_top_k": _raw_turn_fallback_top_k(),
            },
        )
        return self._retrieval_result

    def build_prompt(self, query: str, retrieved: RetrievalResult, sample: UnifiedSample) -> PromptRecord:
        if self._speaker_user_ids:
            grouped = {speaker: [] for speaker, _ in self._speaker_user_ids}
            for entry in retrieved.retrieved_entries:
                speaker = str(entry.metadata.get("official_speaker") or "")
                if speaker in grouped:
                    timestamp = entry.metadata.get("timestamp") or entry.metadata.get("session_date") or ""
                    grouped[speaker].append(f"{timestamp}: {entry.content}")
            speaker_1, speaker_2 = [item[0] for item in self._speaker_user_ids]
            full_prompt = MEM0_OFFICIAL_ANSWER_PROMPT.format(
                speaker_1=speaker_1,
                speaker_2=speaker_2,
                speaker_1_memories=json.dumps(grouped[speaker_1], ensure_ascii=False, indent=2),
                speaker_2_memories=json.dumps(grouped[speaker_2], ensure_ascii=False, indent=2),
                question=query,
            )
            self._prompt_record = PromptRecord(system_prompt=full_prompt, user_prompt="", memory_context="\n".join(sum(grouped.values(), [])), full_prompt=full_prompt, token_count=len(full_prompt.split()), raw={"system": "mem0", "method_contract": "official_locomo_open_source_sdk"})
            return self._prompt_record
        prompt_retrieval = retrieved
        if _expand_retrieved_sources_enabled():
            expanded_entries = self._expand_retrieved_source_turns(retrieved.retrieved_entries, sample=sample)
            if expanded_entries:
                prompt_retrieval = RetrievalResult(
                    query=retrieved.query,
                    retrieved_entries=[*retrieved.retrieved_entries, *expanded_entries],
                    top_k=retrieved.top_k + len(expanded_entries),
                    raw={
                        **(retrieved.raw if isinstance(retrieved.raw, dict) else {}),
                        "expanded_source_turns": len(expanded_entries),
                    },
                )
        prompt = super().build_prompt(query=query, retrieved=prompt_retrieval, sample=sample)
        prompt.raw.update({"system": "mem0", "sdk": "mem0ai"})
        return prompt

    def generate_answer(self, prompt: PromptRecord, sample: UnifiedSample) -> AnswerRecord:
        gateway = self._resolve_generation_gateway(fallback_provider=self._sdk_provider)
        response = gateway.complete_text(prompt=prompt.full_prompt, model=self.generation_model, temperature=0.0)
        self._answer_record = AnswerRecord(
            answer=response.text,
            raw_response={**response.to_dict(), "system": self.system_name},
            latency=response.latency,
            token_usage={
                "prompt_tokens_est": prompt.token_count,
                "completion_tokens_est": len(response.text.split()),
            },
        )
        return self._answer_record

    def _resolve_sdk_provider(self) -> str:
        provider = (
            os.getenv("MEM0_PROVIDER")
            or self.llm_provider_name
            or os.getenv("LLM_PROVIDER")
            or "openai"
        ).strip().lower()
        if provider == "mock":
            raise ValueError("Mem0 真实接入不支持 mock provider，请设置 OPENAI_API_KEY 并使用 openai。")
        return provider

    def _ensure_provider_credentials(self) -> None:
        if self._sdk_provider == "openai" and not (os.getenv("MEM0_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
            raise RuntimeError("Mem0 需要 OPENAI_API_KEY 才能完成真实 memory 提取与检索。")

    def _mem0_llm_config(self) -> dict[str, Any]:
        # mem0 2.x reads OPENAI_BASE_URL from the environment and its LLM
        # config does not accept explicit base_url/openai_base_url keys.
        model = os.getenv("MEM0_LLM_MODEL") or self.generation_model
        config = openai_sdk_config(model=model, temperature=0.0, include_base_url=False)
        api_key = os.getenv("MEM0_LLM_API_KEY")
        if api_key:
            config["api_key"] = api_key
        base_url = os.getenv("MEM0_LLM_BASE_URL") or openai_base_url()
        if base_url:
            config["openai_base_url"] = base_url
        config["max_tokens"] = _mem0_llm_max_tokens()
        if os.getenv("MEM0_DEBUG_LLM", "").strip().lower() in {"1", "true", "yes"}:
            config["response_callback"] = _debug_mem0_llm_response
        return config

    def _patch_mem0_llm_client(self) -> None:
        base_url = os.getenv("MEM0_LLM_BASE_URL") or openai_base_url()
        if self._memory is None or not base_url:
            return
        if os.getenv("MEM0_DIRECT_LLM_CLIENT", "true").strip().lower() in {"0", "false", "no", "off"}:
            return

        def generate_response(
            messages: list[dict[str, str]],
            response_format: Any = None,
            tools: list[dict[str, Any]] | None = None,
            tool_choice: str = "auto",
            **_kwargs: Any,
        ):
            from openai import OpenAI

            model = os.getenv("MEM0_LLM_MODEL") or self.generation_model
            max_tokens = _mem0_llm_max_tokens()
            client = OpenAI(
                api_key=(os.getenv("MEM0_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip(),
                base_url=base_url.strip(),
                timeout=float(os.getenv("MEM0_LLM_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "120"))),
                max_retries=int(os.getenv("MEM0_LLM_MAX_RETRIES", os.getenv("LLM_MAX_RETRIES", "2"))),
            )
            started = time.perf_counter()
            params: dict[str, Any] = {
                "model": model,
                "messages": messages,
                _token_param_name(model): max_tokens,
            }
            if response_format and not _json_mode_disabled():
                params["response_format"] = response_format
            # Some OpenAI-compatible gateways accept JSON mode but reject
            # tool schemas (or return a generic routing 500).  Keep a
            # deterministic plain-chat fallback available for those gateways;
            # the extraction prompt already requests the same JSON payload.
            if tools and os.getenv("MEM0_DISABLE_TOOLS", "false").strip().lower() not in {"1", "true", "yes", "on"}:
                params["tools"] = tools
                params["tool_choice"] = tool_choice
            if _mem0_debug_stdout_enabled():
                print(
                    "[mem0-direct-llm-debug] "
                    + str(
                        {
                            "model": model,
                            "base_url": base_url,
                            "response_format": params.get("response_format"),
                            "token_param": _token_param_name(model),
                            "max_tokens": max_tokens,
                            "tools": bool(tools),
                            "message_contains_json": [
                                "json" in str(item.get("content", "")).lower()
                                for item in messages
                            ],
                        }
                    ),
                    flush=True,
                )
            call_record: dict[str, Any] = {
                "model": model,
                "base_url": base_url,
                "response_format": params.get("response_format"),
                "token_param": _token_param_name(model),
                "max_tokens": max_tokens,
                "tools": bool(tools),
                "status": "started",
            }
            self._debug_state.setdefault("mem0_direct_llm_calls", []).append(call_record)
            try:
                response = client.chat.completions.create(**params)
            except Exception as exc:
                call_record["status"] = "error"
                call_record["error"] = f"{type(exc).__name__}: {exc}"
                call_record["latency"] = time.perf_counter() - started
                if _mem0_debug_stdout_enabled():
                    print(
                        "[mem0-direct-llm-error] "
                        + str(
                            {
                                "status": call_record["status"],
                                "latency": round(call_record["latency"], 3),
                                "error": call_record["error"],
                            }
                        ),
                        flush=True,
                    )
                raise
            if tools:
                message = response.choices[0].message
                result = {
                    "content": message.content,
                    "tool_calls": [
                        {
                            "name": call.function.name,
                            "arguments": json.loads(_clean_llm_text(call.function.arguments)),
                        }
                        for call in (message.tool_calls or [])
                    ],
                }
                call_record["status"] = "ok"
                call_record["latency"] = time.perf_counter() - started
                call_record["content_preview"] = str(result.get("content") or "")[:300]
                call_record["tool_call_count"] = len(result["tool_calls"])
                if _mem0_debug_stdout_enabled():
                    print(
                        "[mem0-direct-llm-result] "
                        + str(
                            {
                                "status": call_record["status"],
                                "latency": round(call_record["latency"], 3),
                                "tool_call_count": call_record["tool_call_count"],
                                "content_preview": call_record["content_preview"],
                            }
                        ),
                        flush=True,
                    )
                return result
            content = _clean_llm_text(response.choices[0].message.content or "")
            call_record["status"] = "ok"
            call_record["latency"] = time.perf_counter() - started
            call_record["content_preview"] = content[:300]
            if _mem0_debug_stdout_enabled():
                print(
                    "[mem0-direct-llm-result] "
                    + str(
                    {
                        "status": call_record["status"],
                        "latency": round(call_record["latency"], 3),
                        "content_preview": call_record["content_preview"],
                    }
                ),
                flush=True,
                )
            return content

        self._memory.llm.generate_response = generate_response

    def _patch_mem0_embedding_client(self) -> None:
        if self._memory is None:
            return
        embedder = getattr(self._memory, "embedding_model", None)
        if embedder is None or getattr(embedder, "_map_platform_truncation_patched", False):
            return

        original_embed = embedder.embed
        original_embed_batch = getattr(embedder, "embed_batch", None)

        def max_chars() -> int:
            return int(os.getenv("MEM0_EMBEDDING_MAX_INPUT_CHARS", os.getenv("EMBEDDING_MAX_INPUT_CHARS", "24000")))

        def truncate_one(text: Any) -> str:
            raw = str(text or "")
            safe = truncate_text_for_embedding(raw, max_chars=max_chars())
            if safe != raw:
                self._debug_state.setdefault("mem0_embedding_truncations", []).append(
                    {
                        "original_chars": len(raw),
                        "truncated_chars": len(safe),
                        "max_chars": max_chars(),
                    }
                )
            return safe

        def embed(text: Any, memory_action: Any = None):
            safe_text = truncate_one(text)
            try:
                return original_embed(safe_text, memory_action)
            except Exception as exc:
                retried_texts = truncate_embedding_inputs_for_context_error([safe_text], exc)
                if retried_texts == [safe_text]:
                    raise
                self._debug_state.setdefault("mem0_embedding_context_retries", []).append(
                    {
                        "memory_action": memory_action,
                        "original_chars": len(safe_text),
                        "retried_chars": len(retried_texts[0]),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return original_embed(retried_texts[0], memory_action)

        embedder.embed = embed

        if callable(original_embed_batch):

            def embed_batch(texts: Any, memory_action: Any = "add"):
                safe_texts = [truncate_one(text) for text in list(texts or [])]
                try:
                    return original_embed_batch(safe_texts, memory_action)
                except Exception as exc:
                    retried_texts = truncate_embedding_inputs_for_context_error(safe_texts, exc)
                    if retried_texts == safe_texts:
                        raise
                    self._debug_state.setdefault("mem0_embedding_context_retries", []).append(
                        {
                            "memory_action": memory_action,
                            "batch_size": len(safe_texts),
                            "original_chars": sum(len(text) for text in safe_texts),
                            "retried_chars": sum(len(text) for text in retried_texts),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    return original_embed_batch(retried_texts, memory_action)

            embedder.embed_batch = embed_batch

        embedder._map_platform_truncation_patched = True

    def _mem0_embedder_config(self) -> dict[str, Any]:
        config = embedding_sdk_config(model=self.embedding_model, include_base_url=False)
        base_url = embedding_base_url()
        if base_url:
            config["openai_base_url"] = base_url
        return config

    def _augment_with_raw_turn_fallback(
        self,
        *,
        retrieved: list[RetrievedMemory],
        query: str,
        sample: UnifiedSample,
        top_k: int,
    ) -> list[RetrievedMemory]:
        if top_k <= 0:
            return retrieved
        seen_source_ids = {
            str(source_id)
            for entry in retrieved
            for source_id in entry.source_ids
            if str(source_id).strip()
        }
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, turn in enumerate(sample.history):
            text = str(turn.get("text") or turn.get("content") or "").strip()
            if not text:
                continue
            turn_id = str(turn.get("turn_id") or turn.get("source_id") or turn.get("id") or f"turn_{index + 1}").strip()
            if turn_id in seen_source_ids:
                continue
            score = _lexical_score(query, text)
            if score <= 0.0:
                continue
            scored.append((score, index, turn))
        scored.sort(key=lambda item: (-item[0], item[1]))
        augmented = list(retrieved)
        for rank_offset, (score, index, turn) in enumerate(scored[:top_k], start=1):
            turn_id = str(turn.get("turn_id") or turn.get("source_id") or turn.get("id") or f"turn_{index + 1}").strip()
            speaker = str(turn.get("speaker") or "").strip()
            text = str(turn.get("text") or turn.get("content") or "").strip()
            augmented.append(
                RetrievedMemory(
                    entry_id=f"mem0:raw_turn_fallback:{turn_id}",
                    content=f"{speaker}: {text}" if speaker else text,
                    score=score,
                    rank=len(augmented) + 1,
                    source_ids=[turn_id],
                    metadata={
                        "memory_bucket": "raw_turn_fallback",
                        "session_id": turn.get("session_id"),
                        "session_date": turn.get("timestamp"),
                        "speaker": speaker,
                        "retrieval_mode": "mem0_plus_lexical_raw_turn_fallback",
                        "fallback_rank": rank_offset,
                    },
                )
            )
        return augmented

    def _expand_retrieved_source_turns(
        self,
        retrieved: list[RetrievedMemory],
        *,
        sample: UnifiedSample,
    ) -> list[RetrievedMemory]:
        history_by_id = {
            str(turn.get("turn_id") or turn.get("source_id") or turn.get("id") or "").strip(): turn
            for turn in sample.history
            if str(turn.get("turn_id") or turn.get("source_id") or turn.get("id") or "").strip()
        }
        expanded: list[RetrievedMemory] = []
        seen: set[str] = set()
        max_turns = max(0, int(os.getenv("MEM0_EXPANDED_SOURCE_TURN_LIMIT", "8")))
        for entry in retrieved:
            for source_id in entry.source_ids:
                turn_id = str(source_id).strip()
                if not turn_id or turn_id in seen or turn_id not in history_by_id:
                    continue
                if max_turns and len(expanded) >= max_turns:
                    return expanded
                seen.add(turn_id)
                turn = history_by_id[turn_id]
                speaker = str(turn.get("speaker") or "").strip()
                text = str(turn.get("text") or turn.get("content") or "").strip()
                if not text:
                    continue
                expanded.append(
                    RetrievedMemory(
                        entry_id=f"mem0:source_turn:{turn_id}",
                        content=f"{speaker}: {text}" if speaker else text,
                        score=entry.score,
                        rank=len(retrieved) + len(expanded) + 1,
                        source_ids=[turn_id],
                        metadata={
                            "memory_bucket": "expanded_source_turn",
                            "parent_memory_id": entry.entry_id,
                            "session_id": turn.get("session_id"),
                            "session_date": turn.get("timestamp"),
                            "speaker": speaker,
                        },
                    )
                )
        return expanded

    def _load_entries(self) -> list[MemoryEntry]:
        assert self._memory is not None
        top_k = int(os.getenv("MEM0_GET_ALL_LIMIT", "2000"))
        items: list[dict[str, Any]] = []
        for speaker, user_id in self._speaker_user_ids:
            response = self._memory.get_all(filters={"user_id": user_id}, top_k=top_k)
            for item in _coerce_result_list(response):
                item["_official_speaker"] = speaker
                items.append(item)
        entries: list[MemoryEntry] = []
        for index, item in enumerate(items, start=1):
            metadata = item.get("metadata") or {}
            entries.append(
                MemoryEntry(
                    entry_id=str(item.get("id") or f"mem0_{index}"),
                    content=_extract_content(item),
                    raw=item,
                    created_at=item.get("created_at"),
                    updated_at=item.get("updated_at"),
                    source_ids=_extract_source_ids(item),
                    metadata={
                        "metadata": metadata,
                        "user_id": item.get("user_id"),
                        "agent_id": item.get("agent_id"),
                        "run_id": item.get("run_id"),
                    },
                )
            )
        return entries

    def _direct_extract_entries(self, *, history: list[dict], sample: UnifiedSample, batch_size: int) -> list[MemoryEntry]:
        from openai import OpenAI

        base_url = os.getenv("MEM0_LLM_BASE_URL") or openai_base_url()
        model = os.getenv("MEM0_LLM_MODEL") or self.generation_model
        client = OpenAI(
            api_key=(os.getenv("MEM0_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip(),
            base_url=base_url.strip() if base_url else None,
            timeout=float(os.getenv("MEM0_LLM_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "120"))),
            max_retries=int(os.getenv("MEM0_LLM_MAX_RETRIES", os.getenv("LLM_MAX_RETRIES", "2"))),
        )
        entries: list[MemoryEntry] = []
        batch_records = self._debug_state.setdefault("build_memory_batches", [])
        for start in range(0, len(history), batch_size):
            batch_turns = history[start : start + batch_size]
            if not batch_turns:
                continue
            batch_number = start // batch_size + 1
            batch_record: dict[str, Any] = {
                "batch": batch_number,
                "num_messages": len(batch_turns),
                "start": start,
                "status": "started",
                "mode": "direct_openai_json_extraction",
            }
            batch_records.append(batch_record)
            prompt = self._direct_extraction_prompt(batch_turns)
            started = time.perf_counter()
            params: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                _token_param_name(model): _mem0_llm_max_tokens(),
            }
            try:
                response = client.chat.completions.create(**params)
                text = _clean_llm_text(response.choices[0].message.content or "")
                extracted = self._parse_direct_extraction_response(text)
            except Exception as exc:
                batch_record["status"] = "error"
                batch_record["error"] = f"{type(exc).__name__}: {exc}"
                batch_record["latency"] = time.perf_counter() - started
                raise
            for item_index, item in enumerate(extracted, start=1):
                content = str(item.get("content") or item.get("memory") or "").strip()
                if not content:
                    continue
                source_ids = item.get("source_turn_ids") or item.get("source_ids") or []
                if not isinstance(source_ids, list):
                    source_ids = [source_ids]
                source_ids = [str(source_id).strip() for source_id in source_ids if str(source_id).strip()]
                source_turn = next(
                    (
                        turn
                        for turn in batch_turns
                        if str(turn.get("turn_id") or turn.get("source_id") or turn.get("id") or "").strip()
                        in set(source_ids)
                    ),
                    {},
                )
                entries.append(
                    MemoryEntry(
                        entry_id=f"mem0:direct:{batch_number}:{item_index}",
                        content=content,
                        raw={"batch": batch_number, "item": item},
                        source_ids=source_ids,
                        metadata={
                            "memory_bucket": "semantic",
                            "implementation_mode": "direct_openai_json_extraction",
                            "sample_id": sample.sample_id,
                            "session_id": source_turn.get("session_id"),
                            "session_date": source_turn.get("timestamp"),
                            "speaker": source_turn.get("speaker"),
                        },
                    )
                )
            batch_record["status"] = "ok"
            batch_record["latency"] = time.perf_counter() - started
            batch_record["num_extracted"] = len(extracted)
            batch_record["response_preview"] = text[:300]
            if _mem0_debug_stdout_enabled():
                print(
                    "[mem0-direct-extract-result] "
                    + str(
                        {
                            "batch": batch_number,
                            "latency": round(batch_record["latency"], 3),
                            "num_extracted": len(extracted),
                        }
                    ),
                    flush=True,
                )
        return entries

    def _direct_extraction_prompt(self, turns: list[dict]) -> str:
        lines = []
        for index, turn in enumerate(turns, start=1):
            turn_id = str(turn.get("turn_id") or turn.get("source_id") or turn.get("id") or f"turn_{index}").strip()
            session_id = str(turn.get("session_id") or "").strip()
            timestamp = str(turn.get("timestamp") or "").strip()
            speaker = str(turn.get("speaker") or "unknown").strip()
            text = str(turn.get("text") or turn.get("content") or "").strip()
            if not text:
                continue
            lines.append(
                f"[turn_id={turn_id}; session_id={session_id}; session_date={timestamp}; speaker={speaker}]\n"
                f"  {text}"
            )
        dialogue = "\n\n".join(lines)
        return (
            "Extract durable semantic memories from the dialogue below.\n"
            "Keep concrete facts, preferences, events, plans, updates, and date-sensitive observations.\n"
            "Use session_date to resolve relative dates such as yesterday.\n"
            "Each memory must include source_turn_ids copied from the turn_id headers.\n"
            "Return only a JSON array. Each item must be an object with keys: content, source_turn_ids.\n\n"
            f"Dialogue:\n{dialogue}"
        )

    def _parse_direct_extraction_response(self, text: str) -> list[dict[str, Any]]:
        cleaned = _clean_llm_text(text)
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match:
            cleaned = match.group(0)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            for key in ("memories", "results", "items"):
                value = parsed.get(key)
                if isinstance(value, list):
                    parsed = value
                    break
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    def _fallback_entries(self, history: list[dict]) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        for index, turn in enumerate(history, start=1):
            content = str(turn.get("content") or turn.get("text") or "").strip()
            if not content:
                continue
            turn_id = str(
                turn.get("turn_id")
                or turn.get("source_id")
                or turn.get("id")
                or f"turn_{index}"
            )
            session_id = str(turn.get("session_id") or turn.get("session") or "")
            speaker = str(turn.get("speaker") or "").strip()
            source_ids = [turn_id]
            if session_id:
                source_ids.append(session_id)
            entries.append(
                MemoryEntry(
                    entry_id=f"mem0:fallback:{turn_id}",
                    content=f"{speaker}: {content}" if speaker else content,
                    source_ids=source_ids,
                    metadata={
                        "memory_bucket": "semantic",
                        "implementation_mode": "local_fallback_after_sdk_error",
                        "session_id": session_id,
                        "session_date": turn.get("timestamp"),
                        "speaker": speaker,
                    },
                    raw=to_jsonable(turn),
                )
            )
        return entries

    def _direct_retrieve(self, *, query: str, top_k: int) -> list[RetrievedMemory]:
        mode = os.getenv("MEM0_DIRECT_RETRIEVAL_MODE", "hybrid_vector").strip().lower()
        if mode in {"lexical", "keyword"}:
            return self._fallback_retrieve(query=query, top_k=top_k)

        try:
            self._ensure_entry_embeddings()
            query_vector = self._embedding_for_text(query)
            scored = []
            for index, entry in enumerate(self._memory_entries):
                vector_score = _cosine(query_vector, self._embedding_for_text(entry.content))
                lexical_score = _lexical_score(query, entry.content)
                score = vector_score if mode in {"vector", "embedding", "semantic"} else (0.85 * vector_score + 0.15 * lexical_score)
                scored.append((score, vector_score, lexical_score, index, entry))
            scored.sort(key=lambda item: (-item[0], item[3]))
            return [
                RetrievedMemory(
                    entry_id=entry.entry_id,
                    content=entry.content,
                    score=score,
                    rank=rank,
                    source_ids=list(entry.source_ids),
                    metadata={
                        **entry.metadata,
                        "implementation_mode": "direct_openai_json_extraction",
                        "retrieval_mode": mode,
                        "vector_score": vector_score,
                        "lexical_score": lexical_score,
                    },
                )
                for rank, (score, vector_score, lexical_score, _index, entry) in enumerate(scored[:top_k], start=1)
            ]
        except Exception as exc:
            self._debug_state.setdefault("direct_retrieval_errors", []).append(f"{type(exc).__name__}: {exc}")
            if os.getenv("MEMORY_EVAL_FORMAL_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}:
                raise RuntimeError(
                    f"Mem0 direct semantic retrieval failed; lexical fallback is disabled in formal mode: {exc}"
                ) from exc
            return self._fallback_retrieve(query=query, top_k=top_k)

    def _ensure_entry_embeddings(self) -> None:
        missing = [
            entry.content
            for entry in self._memory_entries
            if entry.content.strip() and entry.content.strip() not in self._retrieval_embedding_cache
        ]
        if not missing:
            return
        vectors = embed_texts(missing, model=self.embedding_model)
        for text, vector in zip(missing, vectors):
            self._retrieval_embedding_cache[text.strip()] = vector

    def _embedding_for_text(self, text: str) -> list[float]:
        key = text.strip()
        cached = self._retrieval_embedding_cache.get(key)
        if cached is not None:
            return cached
        vector = embed_texts([key], model=self.embedding_model)[0]
        self._retrieval_embedding_cache[key] = vector
        return vector

    def _fallback_retrieve(self, *, query: str, top_k: int) -> list[RetrievedMemory]:
        scored = [
            (_lexical_score(query, entry.content), index, entry)
            for index, entry in enumerate(self._memory_entries)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        retrieved: list[RetrievedMemory] = []
        for rank, (score, _index, entry) in enumerate(scored[:top_k], start=1):
            retrieved.append(
                RetrievedMemory(
                    entry_id=entry.entry_id,
                    content=entry.content,
                    score=score,
                    rank=rank,
                    source_ids=list(entry.source_ids),
                    metadata={
                        **entry.metadata,
                        "implementation_mode": "local_fallback_after_sdk_error",
                    },
                )
            )
        return retrieved

    def validate_setup(self) -> dict[str, Any]:
        provider = self._sdk_provider
        if provider == "openai" and not (os.getenv("MEM0_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
            return {"system": self.system_name, "ready": False, "provider": provider, "message": "缺少 OPENAI_API_KEY。"}
        return {"system": self.system_name, "ready": True, "provider": provider, "message": ""}
