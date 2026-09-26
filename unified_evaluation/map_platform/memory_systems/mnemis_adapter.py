from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Literal

from map_platform.datasets.schema import UnifiedSample
from map_platform.llm.gateway import token_param_name
from map_platform.memory_systems.base import (
    BaseMemorySystemAdapter,
    MemoryEntry,
    OrganizationState,
    PromptRecord,
    RetrievalResult,
    RetrievedMemory,
)


def _clean_json_llm_content(content: str) -> str:
    text = str(content or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text.lower().startswith("<think>"):
        marker = text.find("{")
        text = text[marker:] if marker >= 0 else ""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    if text and not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
    return text or "{}"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _rerank_texts(query: str, texts: list[str]) -> list[float] | None:
    """Score (query, text) pairs with a Qwen3-Reranker HTTP service.

    Official Mnemis uses Qwen3-Reranker-8B to reorder retrieved candidates before
    truncating to rag_top_k/graph_top_k. Enabled by MNEMIS_USE_RERANKER. Endpoint
    configured via MNEMIS_RERANKER_BASE_URL / MNEMIS_RERANKER_MODEL / MNEMIS_RERANKER_API_KEY.
    Tries a Jina/Cohere-style /rerank endpoint first, then a generic scoring call.
    Returns one relevance score per text, or None on any failure (caller keeps
    original order)."""
    if not texts:
        return None
    base_url = (
        _env("MNEMIS_RERANKER_BASE_URL", "")
        or _env("RERANKER_BASE_URL", "")
    )
    if not base_url:
        return None
    model = _env("MNEMIS_RERANKER_MODEL", "") or _env("RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")
    api_key = _env("MNEMIS_RERANKER_API_KEY", "") or _env("RERANKER_API_KEY", "") or _env("EMBEDDING_API_KEY", "") or "EMPTY"
    timeout = float(_env("MNEMIS_RERANKER_TIMEOUT", "60"))
    try:
        import requests

        api_path = _env("MNEMIS_RERANKER_API_PATH", "") or _env("RERANKER_API_PATH", "/v1/rerank")
        url = base_url.rstrip("/") + "/" + api_path.lstrip("/")
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "query": query, "documents": texts},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results") or data.get("data") or []
            scores = [0.0] * len(texts)
            for r in results:
                idx = r.get("index")
                score = r.get("relevance_score", r.get("score"))
                if isinstance(idx, int) and 0 <= idx < len(texts) and score is not None:
                    scores[idx] = float(score)
            return scores
    except Exception:
        return None
    return None


def _truncate_text(text: Any, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _parse_reference_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in (
        "%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y",
        "%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M",
    ):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _format_fact_with_range(fact: str, valid_at: Any, invalid_at: Any) -> str:
    """Render a Graphiti edge fact in the official Mnemis FACTS style.

    Official format puts the temporal scope as a trailing "(valid_at - invalid_at)"
    range, e.g. "Jon visited Paris recently. (2023-01-29 - 2023-02-08)". We must NOT
    use a leading "[valid_at] " prefix: the model reads a leading date as the time the
    statement was made, then re-derives any relative word still in the fact text
    ("yesterday", "last week") off that prefix, landing one day/week early. A trailing
    range disambiguates without triggering a second conversion.
    """
    fact = fact.strip()
    v = str(valid_at or "").strip()
    iv = str(invalid_at or "").strip()
    if not v and not iv:
        return fact
    if v and iv:
        scope = f"{v} - {iv}"
    elif v:
        scope = f"{v} - now"
    else:
        scope = f"until {iv}"
    return f"{fact} ({scope})"


def _format_rag_graph_chunks(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    max_items = _env_int("MNEMIS_CONTEXT_RAG_CHUNKS", _env_int("MNEMIS_RAG_TOP_K", len(results)))
    max_chars = _env_int("MNEMIS_CONTEXT_CHUNK_MAX_CHARS", 0)
    for idx, item in enumerate(results[:max_items]):
        content = _truncate_text(item.get("content") or item.get("fact") or "", max_chars)
        if not content:
            continue
        valid_at = str(item.get("valid_at") or "").strip()
        uuid = str(item.get("uuid") or "").strip()
        prefix = f"Message Chunk {idx}"
        if valid_at:
            prefix += f" [{valid_at}]"
        if uuid:
            prefix += f" ({uuid})"
        lines.append(f"{prefix}:\n{content}")
    return "\n\n".join(lines)


def _normalize_query_tokens(text: Any) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "with",
        "what", "when", "where", "who", "why", "how", "did", "does", "do", "was",
        "were", "is", "are", "be", "been", "being", "her", "his", "their", "she",
        "he", "they", "it", "that", "this",
    }
    return {token for token in tokens if len(token) > 2 and token not in stop}


def _history_rag_candidates(query: str, history: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Build official-style timestamped message chunks from raw conversation history.

    Official Mnemis RAG_GRAPH contexts include raw Message Chunk sections in addition
    to graph facts/entities. Using raw history chunks here keeps System-1 close to
    the released LoCoMo result files instead of depending only on Graphiti fact
    search output.
    """
    if limit <= 0:
        return []
    by_session: dict[str, list[dict[str, Any]]] = {}
    for turn in history:
        sid = str(turn.get("session_id") or "session").strip()
        by_session.setdefault(sid, []).append(turn)

    query_tokens = _normalize_query_tokens(query)
    candidates: list[dict[str, Any]] = []
    for sid, turns in by_session.items():
        session_index = 0
        match_indexes: list[int] = []
        for idx, turn in enumerate(turns):
            text = str(turn.get("text") or "")
            turn_tokens = _normalize_query_tokens(text)
            overlap = len(query_tokens & turn_tokens)
            if overlap:
                match_indexes.append(idx)
        if not match_indexes:
            # Keep one low-priority session summary candidate so reranker can still
            # surface paraphrased evidence that lexical overlap misses.
            match_indexes = [0]

        seen_windows: set[tuple[int, int]] = set()
        for match_idx in match_indexes:
            start = max(0, match_idx - 2)
            end = min(len(turns), match_idx + 3)
            window = (start, end)
            if window in seen_windows:
                continue
            seen_windows.add(window)
            selected = turns[start:end]
            lines = []
            source_ids = []
            for turn in selected:
                text = str(turn.get("text") or "").strip()
                if not text:
                    continue
                speaker = str(turn.get("speaker") or "unknown").strip()
                timestamp = str(turn.get("timestamp") or "").strip()
                prefix = f"[{timestamp}] " if timestamp else ""
                lines.append(f"{prefix}{speaker}: {text}")
                source_id = str(turn.get("turn_id") or "").strip()
                if source_id:
                    source_ids.append(source_id)
            if not lines:
                continue
            content = "\n".join(lines)
            score = 0.0
            normalized_content_tokens = _normalize_query_tokens(content)
            if query_tokens:
                score = len(query_tokens & normalized_content_tokens) / len(query_tokens)
            candidates.append({
                "uuid": f"{sid}:{session_index}",
                "content": content,
                "score": score,
                "valid_at": str(selected[0].get("timestamp") or ""),
                "session_id": sid,
                "source_ids": source_ids,
                "route": "history_rag",
            })
            session_index += 1

    candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    pre_limit = max(limit * 4, limit)
    narrowed = candidates[:pre_limit]
    scores = _rerank_texts(query, [item["content"] for item in narrowed])
    if scores:
        order = sorted(range(len(narrowed)), key=lambda idx: scores[idx], reverse=True)
        narrowed = [narrowed[idx] for idx in order]
        for idx, item in enumerate(narrowed):
            item["rerank_score"] = scores[order[idx]]
    return narrowed[:limit]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw.isdigit() else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _build_graphiti_llm_client():
    from graphiti_core.llm_client import OpenAIClient
    from graphiti_core.llm_client.config import LLMConfig

    class ChatCompletionsOpenAIClient(OpenAIClient):
        async def _create_structured_completion(
            self,
            model,
            messages,
            temperature,
            max_tokens,
            response_model,
            reasoning=None,
            verbosity=None,
        ):
            schema_instruction = {
                "role": "system",
                "content": (
                    "Return only one json object that exactly matches this JSON Schema. "
                    "Use the property names from the schema verbatim and do not add aliases:\n"
                    + json.dumps(response_model.model_json_schema(), ensure_ascii=False)
                ),
            }
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [schema_instruction, *messages],
                "response_format": {"type": "json_object"},
                token_param_name(str(model)): max_tokens,
            }
            # configured provider and other OpenAI-compatible gateways accept the plain
            # Chat Completions contract. Provider-specific reasoning/template
            # fields remain opt-in instead of being sent on every request.
            if _env_bool("MNEMIS_USE_EXTRA_BODY", False):
                request_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            effort = self._resolve_reasoning_effort(model, reasoning) if reasoning else None
            if effort:
                request_kwargs["reasoning_effort"] = effort
            if verbosity:
                request_kwargs["verbosity"] = verbosity
            response = await self.client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or "{}"
            content = _clean_json_llm_content(content)
            validated = response_model.model_validate_json(content)
            usage = getattr(response, "usage", None)
            return (
                validated.model_dump(),
                getattr(usage, "prompt_tokens", 0) if usage else 0,
                getattr(usage, "completion_tokens", 0) if usage else 0,
            )

        def _handle_structured_response(self, response):
            if isinstance(response, tuple):
                return response
            message = response.choices[0].message
            parsed = getattr(message, "parsed", None)
            if parsed is not None:
                payload = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
            elif message.content:
                payload = json.loads(_clean_json_llm_content(message.content))
            elif getattr(message, "refusal", None):
                raise RuntimeError(message.refusal)
            else:
                raise RuntimeError(f"Invalid structured response from LLM: {response}")

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            return payload, input_tokens or 0, output_tokens or 0

    base_url = (
        _env("MNEMIS_LLM_BASE_URL", "")
        or os.getenv("MEMORY_BUILD_OPENAI_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
    )
    api_key = (
        _env("MNEMIS_LLM_API_KEY", "")
        or os.getenv("MEMORY_BUILD_OPENAI_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("API_T", "").strip()
    )
    model = (
        _env("MNEMIS_LLM_MODEL", "")
        or os.getenv("MEMORY_BUILD_MODEL", "").strip()
        or os.getenv("LLM_MODEL", "").strip()
    )
    max_tokens = _env_int("MEMORY_BUILD_LLM_MAX_TOKENS", 16384)
    config = LLMConfig(
        api_key=api_key or None,
        base_url=base_url or None,
        model=model or None,
        small_model=model or None,
        max_tokens=max_tokens,
    )
    return ChatCompletionsOpenAIClient(
        config=config,
        max_tokens=max_tokens,
        reasoning=_env("MNEMIS_REASONING_EFFORT", ""),
        verbosity=_env("MNEMIS_VERBOSITY", ""),
    )


def _build_graphiti():
    from graphiti_core import Graphiti
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    class FloatEncodingOpenAIEmbedder(OpenAIEmbedder):
        """Graphiti embedder compatible with the local BGE-M3 endpoint.

        The OpenAI-compatible local server requires an explicit
        ``encoding_format=float`` field.  graphiti-core's stock embedder omits
        that optional field, which causes a 400 during graph ingestion even
        though the endpoint itself is healthy.
        """

        async def create(self, input_data):
            try:
                result = await self.client.embeddings.create(
                    input=input_data,
                    model=self.config.embedding_model,
                    encoding_format="float",
                )
            except Exception as exc:
                if "encoding_format" not in str(exc).lower():
                    raise
                result = await self.client.embeddings.create(
                    input=input_data,
                    model=self.config.embedding_model,
                )
            vector = list(result.data[0].embedding)
            if len(vector) != self.config.embedding_dim:
                raise ValueError(
                    f"Mnemis embedding dimension mismatch: got {len(vector)}, "
                    f"expected {self.config.embedding_dim}"
                )
            return vector

        async def create_batch(self, input_data_list):
            try:
                result = await self.client.embeddings.create(
                    input=input_data_list,
                    model=self.config.embedding_model,
                    encoding_format="float",
                )
            except Exception as exc:
                if "encoding_format" not in str(exc).lower():
                    raise
                result = await self.client.embeddings.create(
                    input=input_data_list,
                    model=self.config.embedding_model,
                )
            vectors = [list(embedding.embedding) for embedding in result.data]
            if vectors and any(len(vector) != self.config.embedding_dim for vector in vectors):
                raise ValueError(
                    f"Mnemis embedding dimension mismatch: expected {self.config.embedding_dim}"
                )
            return vectors

    embedding_base_url = _env("MNEMIS_EMBEDDING_BASE_URL", "") or os.getenv("EMBEDDING_BASE_URL", "").strip()
    embedding_api_key = _env("MNEMIS_EMBEDDING_API_KEY", "") or os.getenv("EMBEDDING_API_KEY", "").strip()
    embedding_model = _env("MNEMIS_EMBEDDING_MODEL", "") or os.getenv("EMBEDDING_MODEL", "").strip() or "text-embedding-3-small"
    embedder = FloatEncodingOpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=embedding_api_key or None,
            base_url=embedding_base_url or None,
            embedding_model=embedding_model,
            embedding_dim=_env_int("MNEMIS_EMBEDDING_DIMS", _env_int("EMBEDDING_DIMS", 1024)),
        )
    )
    return Graphiti(
        uri=_env("MNEMIS_NEO4J_URI", "bolt://localhost:7687"),
        user=_env("MNEMIS_NEO4J_USER", "neo4j"),
        password=_env("MNEMIS_NEO4J_PASSWORD", ""),
        llm_client=_build_graphiti_llm_client(),
        embedder=embedder,
    )


class _AsyncRunner:
    """Run all adapter async resources on one long-lived event loop."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        self._loop.close()

    def run(self, coro):
        if self._closed:
            coro.close()
            raise RuntimeError("Mnemis async runner is closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()


async def _close_async_client(owner: Any) -> None:
    client = getattr(owner, "client", None)
    if client is None:
        return
    close = getattr(client, "close", None) or getattr(client, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _infer_source_ids(content: str, history: list[dict[str, Any]], max_matches: int = 5) -> list[str]:
    if not content.strip():
        return []
    content_words = set(content.lower().split())
    if not content_words:
        return []
    scored: list[tuple[float, str]] = []
    for turn in history:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        turn_id = str(turn.get("turn_id") or turn.get("source_id") or turn.get("dia_id") or "")
        if not turn_id:
            continue
        turn_words = set(text.lower().split())
        if not turn_words:
            continue
        overlap = len(content_words & turn_words) / max(len(content_words), 1)
        if overlap > 0.15:
            scored.append((overlap, turn_id))
    scored.sort(key=lambda x: -x[0])
    return [tid for _, tid in scored[:max_matches]]


# Add Mnemis global_selection to path if available
_MNEMIS_DIR = Path(__file__).resolve().parents[1] / "Mnemis"
if _MNEMIS_DIR.exists() and str(_MNEMIS_DIR) not in sys.path:
    sys.path.insert(0, str(_MNEMIS_DIR))


NODE_SELECTION_PROMPT = """You are analyzing a hierarchical knowledge graph to help answer a user query.

Select all nodes that could help answer the query. A node is helpful if it:

- Directly relates to the query;
- Covers a clearly relevant topic, concept, or category;
- Provides useful background or context;
- Contains user-specific information (e.g. interests, goals, constraints);
- Likely has sub-nodes that may be helpful.

Do not be overly strict: include nodes that might provide context or personalization, even if they seem partially redundant.

For each selected node:
- "name" is the node's name.
- "uuid" is the node's unique identifier.
- "get_all_children" is an boolean value. Set true only if you're confident all its sub-nodes are helpful.
---
User Query:
"{query}"

Available Nodes:
{nodes_info}

Respond in JSON format: {{"selections": [{{"name": "...", "uuid": "...", "get_all_children": true/false}}, ...]}}
"""

HIERARCHY_CATEGORIZATION_SYSTEM_PROMPT = """You are an AI assistant specialized in semantic categorization of nodes.

Group indexed node names into semantically meaningful categories using both names and descriptions. Reuse an existing category when its attributes match; otherwise create a new one. Category names MUST NOT use the word "and" as a connector.

Return categories as JSON using this schema:
{"categories":[{"category":"specific category","indexes":[0,1,2]}]}

A node may belong to multiple categories. Use the minimal shared attributes that justify each grouping. There must be NO leftover nodes; single-member categories are allowed when necessary. The node "user" and first-person references such as "I" or "me" belong to "Speaker". Use only supplied integer indexes and do not repeat node names in the output."""


def _hierarchy_user_prompt(*, layer: int, items_desc: str, previous_categories: str) -> str:
    compression_guidance = (
        ""
        if layer == 1
        else "\nThis layer MUST compress the lower layer: return fewer categories than input nodes. "
        "Prefer 2-6 coherent super-categories and avoid merely renaming individual inputs."
    )
    return f"""<NODE INDEXED NAMES AND DESCRIPTIONS>
{items_desc}
</NODE INDEXED NAMES AND DESCRIPTIONS>

<EXISTING CATEGORIES>
{previous_categories or 'None'}
</EXISTING CATEGORIES>

<GUIDANCE ON CATEGORY GRANULARITY>
You are at Layer {layer}. Layer 1 must use specific, fine-grained categories. Higher layers group lower-layer categories into broader, more abstract super-categories. Categories may have multiple parents. Do not merge loosely related categories.
{compression_guidance}
</GUIDANCE ON CATEGORY GRANULARITY>

Return JSON only. Ensure every index occurs in at least one category."""


class MnemisAdapter(BaseMemorySystemAdapter):
    """Adapter for Mnemis: Dual-Route Retrieval on Hierarchical Graphs.

    Requires:
      - Neo4j running (MNEMIS_NEO4J_URI)
      - graphiti_core installed (pip install graphiti-core)
      - Data pre-ingested via graphiti_core into Neo4j with proper group_id

    The adapter supports two modes:
      1. Pre-built mode (default): assume graph is already built in Neo4j, only do retrieval
      2. Full mode (MNEMIS_BUILD_GRAPH=true): build graph from history via graphiti_core
    """

    system_name = "mnemis"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._async_runner = _AsyncRunner()
        self._driver: Any = None
        self._graphiti: Any = None
        self._global_selector: Any = None
        self._group_id: str = ""
        self._episode_turn_map: dict[str, list[str]] = {}

    def reset(self, sample_id: str) -> None:
        super().reset(sample_id)
        conv_id = sample_id.split("__")[0] if "__" in sample_id else sample_id
        self._group_id = f"{_env('MNEMIS_GROUP_ID_PREFIX', 'map_bench')}_{_safe_component(conv_id)}"
        self._episode_turn_map = {}
        self._initialize_driver()

    def _run_async(self, coro):
        return self._async_runner.run(coro)

    def _initialize_driver(self) -> None:
        if self._driver is not None:
            return
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            self._debug_state["init_error"] = "neo4j package not installed"
            return

        uri = _env("MNEMIS_NEO4J_URI", "bolt://localhost:7687")
        user = _env("MNEMIS_NEO4J_USER", "neo4j")
        password = _env("MNEMIS_NEO4J_PASSWORD", "")
        if not password:
            self._debug_state["init_error"] = "MNEMIS_NEO4J_PASSWORD not set"
            return

        try:
            self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            self._debug_state["init_error"] = f"Neo4j connection failed: {e}"
            return

    def _get_global_selector(self):
        if self._global_selector is not None:
            return self._global_selector

        if self._driver is None:
            return None

        try:
            from global_selection.global_selector import GlobalSelector, GlobalSelectorConfig
        except ImportError as e:
            self._debug_state["selector_error"] = (
                f"GlobalSelector import failed: {type(e).__name__}: {e}"
            )
            return None

        try:
            llm_client = _build_graphiti_llm_client()
            config = GlobalSelectorConfig(use_summary=False, use_tag=True)
            self._global_selector = GlobalSelector(self._driver, llm_client, config)
        except Exception as e:
            self._debug_state["selector_error"] = f"GlobalSelector init failed: {e}"
            return None

        return self._global_selector

    def build_memory(self, history: list[dict[str, Any]], sample: UnifiedSample) -> None:
        if self._driver is None:
            raise RuntimeError(
                "Mnemis Neo4j driver unavailable: "
                f"{self._debug_state.get('init_error', 'unknown initialization error')}"
            )
        if _env_bool("MNEMIS_BUILD_GRAPH", False):
            self._build_graph_from_history(history, sample)
        else:
            self._memory_entries = [
                MemoryEntry(
                    entry_id=f"mnemis_prebuilt_{self._group_id}",
                    content=f"Pre-built graph for group_id={self._group_id}",
                    raw={"group_id": self._group_id, "mode": "prebuilt"},
                    source_ids=[],
                )
            ]

    def _build_graph_from_history(self, history: list[dict[str, Any]], sample: UnifiedSample) -> None:
        try:
            import graphiti_core  # noqa: F401
        except ImportError:
            self._debug_state["build_error"] = "graphiti_core not installed"
            return

        if self._driver is None:
            self._debug_state["build_error"] = "Neo4j driver not available"
            return

        sessions = self._group_by_session(history)
        if _env_bool("MNEMIS_REUSE_EXISTING_GRAPH", True):
            existing_nodes = self._count_existing_group_nodes()
            if existing_nodes > 0 and self._hierarchy_is_valid():
                self._debug_state["reused_existing_graph"] = True
                self._debug_state["existing_graph_nodes"] = existing_nodes
                self._ensure_system2_selector()
                self._set_session_memory_entries(sessions)
                return
            if existing_nodes > 0:
                self._delete_group_graph()

        async def _ingest():
            graphiti = self._get_graphiti()
            for sess in sessions:
                messages = sess["messages"]
                episode_text = "\n".join(
                    f"{m.get('speaker', 'unknown')}: {m.get('text', '')}"
                    for m in messages if m.get("text")
                )
                if not episode_text.strip():
                    continue

                turn_ids = [str(m.get("turn_id") or m.get("dia_id") or "") for m in messages]
                timestamp = messages[0].get("timestamp") if messages else None
                reference_time = _parse_reference_time(timestamp)

                try:
                    await graphiti.add_episode(
                        name=f"session_{sess['session_id']}",
                        episode_body=episode_text,
                        group_id=self._group_id,
                        source_description="conversation",
                        reference_time=reference_time,
                    )
                    self._episode_turn_map[f"session_{sess['session_id']}"] = turn_ids
                except Exception as e:
                    self._debug_state.setdefault("ingest_errors", []).append({
                        "session_id": sess["session_id"],
                        "error": str(e),
                    })

        try:
            self._run_async(_ingest())
        except Exception as e:
            self._debug_state["build_error"] = str(e)
            raise RuntimeError(f"Mnemis graph ingestion failed: {e}") from e

        ingest_errors = self._debug_state.get("ingest_errors", [])
        if ingest_errors:
            raise RuntimeError(
                f"Mnemis graph ingestion incomplete: {len(ingest_errors)} session(s) failed. "
                f"First error: {ingest_errors[0]}"
            )

        # Build hierarchical graph after base graph is ready
        if _env_bool("MNEMIS_BUILD_HIERARCHY", True):
            try:
                self._run_async(self._build_hierarchy())
            except Exception as e:
                self._debug_state["hierarchy_build_error"] = str(e)
                raise RuntimeError(f"Mnemis hierarchy build failed: {e}") from e
            if not self._hierarchy_is_valid():
                self._debug_state["hierarchy_build_error"] = (
                    "Hierarchy validation failed after build"
                )
                raise RuntimeError("Mnemis hierarchy validation failed after build")

        self._ensure_system2_selector()
        self._set_session_memory_entries(sessions)

    def _ensure_system2_selector(self) -> None:
        if _env_bool("MNEMIS_USE_SYSTEM2", True) and self._get_global_selector() is None:
            raise RuntimeError(
                "Mnemis System-2 selector unavailable during build: "
                f"{self._debug_state.get('selector_error', 'unknown error')}"
            )

    def _count_existing_group_nodes(self) -> int:
        async def _count() -> int:
            records, _, _ = await self._driver.execute_query(
                "MATCH (n) WHERE n.group_id = $group_id RETURN count(n) AS count",
                group_id=self._group_id,
            )
            return int(records[0]["count"]) if records else 0

        try:
            return self._run_async(_count())
        except Exception as e:
            self._debug_state["existing_graph_check_error"] = str(e)
            return 0

    def _delete_group_graph(self) -> None:
        async def _delete() -> None:
            await self._driver.execute_query(
                "MATCH (n) WHERE n.group_id = $group_id DETACH DELETE n",
                group_id=self._group_id,
            )

        try:
            self._run_async(_delete())
            self._debug_state["deleted_incomplete_group"] = self._group_id
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete incomplete Mnemis group {self._group_id}: {e}"
            ) from e

    def _hierarchy_is_valid(self) -> bool:
        """Check that the existing graph has proper Category_N labels and CATEGORIZES edges."""
        async def _check() -> bool:
            # Use dynamic label/property checks so a fresh database does not emit
            # "label/property does not exist" notifications before hierarchy creation.
            records, _, _ = await self._driver.execute_query(
                "MATCH (c) "
                "WHERE c.group_id = $gid AND 'Category' IN labels(c) "
                "RETURN max(c['layer']) AS max_layer, count(c) AS cnt",
                gid=self._group_id,
            )
            if not records or not records[0]["cnt"]:
                return False
            max_layer = records[0]["max_layer"]
            if not max_layer or max_layer < 1:
                return False
            # Must have CATEGORIZES edges
            records2, _, _ = await self._driver.execute_query(
                "MATCH (p)-[r]->(c) "
                "WHERE p.group_id = $gid "
                "AND 'Category' IN labels(p) "
                "AND type(r) = 'CATEGORIZES' "
                "RETURN count(*) AS cnt",
                gid=self._group_id,
            )
            if not records2 or records2[0]["cnt"] < 3:
                return False
            # Check that Category_1 label exists (the format global_selector expects)
            records3, _, _ = await self._driver.execute_query(
                "MATCH (c) "
                "WHERE c.group_id = $gid AND 'Category_1' IN labels(c) "
                "RETURN count(c) AS cnt",
                gid=self._group_id,
            )
            if not records3 or records3[0]["cnt"] < 1:
                return False
            return True

        try:
            return self._run_async(_check())
        except Exception:
            return False

    def _set_session_memory_entries(self, sessions: list[dict[str, Any]]) -> None:
        self._memory_entries = [
            MemoryEntry(
                entry_id=f"mnemis_session_{sess['session_id']}",
                content=f"Session {sess['session_id']}: {len(sess['messages'])} messages",
                raw={"session_id": sess["session_id"], "group_id": self._group_id},
                source_ids=[str(m.get("turn_id") or m.get("dia_id") or "") for m in sess["messages"]],
            )
            for sess in sessions
        ]

    async def _build_hierarchy(self) -> None:
        """Build the paper-style many-to-many semantic category hierarchy."""
        import uuid as _uuid

        records, _, _ = await self._driver.execute_query(
            "MATCH (e:Entity) WHERE e.group_id = $gid RETURN e.uuid AS uuid, e.name AS name, e.summary AS summary",
            gid=self._group_id,
        )
        entities = [{"uuid": r["uuid"], "name": r["name"], "summary": r["summary"] or r["name"]} for r in records]
        if not entities:
            self._debug_state["hierarchy_skipped"] = "No entities"
            return

        max_layers = _env_int("MNEMIS_MAX_HIERARCHY_LAYERS", 12)
        current_items = entities
        current_layer = 0
        previous_categories = ""

        while len(current_items) > 1 and current_layer < max_layers:
            current_layer += 1
            items_desc = "\n".join(
                f"{idx}. {it['name']}: {str(it.get('summary') or '')[:240]}"
                for idx, it in enumerate(current_items)
            )
            base_prompt = _hierarchy_user_prompt(layer=current_layer, items_desc=items_desc, previous_categories=previous_categories)
            graphiti = self._get_graphiti()
            llm_client = graphiti.llm_client
            categories: list[dict[str, Any]] = []
            covered_indexes: set[int] = set()
            max_attempts = 3 if current_layer > 1 else 1
            for attempt in range(1, max_attempts + 1):
                retry_note = ""
                if attempt > 1:
                    retry_note = (
                        f"\nYour previous output produced {len(categories)} categories for "
                        f"{len(current_items)} inputs and did not compress. Regroup them into "
                        f"at most {max(1, len(current_items) - 1)} broader categories while "
                        "still covering every index.\n"
                    )
                try:
                    response = await llm_client.client.chat.completions.create(
                        model=llm_client.config.model or self.generation_model,
                        messages=[
                            {"role": "system", "content": HIERARCHY_CATEGORIZATION_SYSTEM_PROMPT},
                            {"role": "user", "content": base_prompt + retry_note},
                        ],
                        response_format={"type": "json_object"},
                        **{token_param_name(str(llm_client.config.model or self.generation_model)): 4096},
                    )
                    content = response.choices[0].message.content or "{}"
                    parsed = json.loads(_clean_json_llm_content(content))
                    raw_categories = parsed.get("categories", [])
                except Exception as e:
                    self._debug_state.setdefault("hierarchy_errors", []).append(
                        {"layer": current_layer, "attempt": attempt, "error": str(e)}
                    )
                    continue

                categories = []
                covered_indexes = set()
                seen_names: set[str] = set()
                banned_names = {"other", "others", "miscellaneous", "general", "unknown"}
                for raw_category in raw_categories if isinstance(raw_categories, list) else []:
                    if not isinstance(raw_category, dict):
                        continue
                    name = str(raw_category.get("category") or raw_category.get("name") or "").strip()
                    normalized_name = name.casefold()
                    if not name or " and " in normalized_name or normalized_name in banned_names or normalized_name in seen_names:
                        continue
                    members: list[int] = []
                    for value in raw_category.get("indexes", raw_category.get("members", [])):
                        try:
                            index = int(value)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= index < len(current_items) and index not in members:
                            members.append(index)
                    if not members:
                        continue
                    seen_names.add(normalized_name)
                    covered_indexes.update(members)
                    categories.append({"name": name, "members": members})

                for index in sorted(set(range(len(current_items))) - covered_indexes):
                    item_name = str(current_items[index].get("name") or "").strip()
                    fallback_name = "Speaker" if item_name.casefold() in {"user", "i", "me"} else item_name
                    categories.append({"name": fallback_name or f"Node {index}", "members": [index]})
                    covered_indexes.add(index)
                if categories and (current_layer == 1 or len(categories) < len(current_items)):
                    self._debug_state.setdefault("hierarchy_attempts", []).append(
                        {"layer": current_layer, "attempt": attempt, "categories": len(categories)}
                    )
                    break
            else:
                categories = []

            if not categories or (current_layer > 1 and len(categories) >= len(current_items)):
                self._debug_state.setdefault("hierarchy_errors", []).append(
                    {"layer": current_layer, "error": f"failed to compress after {max_attempts} attempt(s)"}
                )
                break

            next_layer_items: list[dict[str, Any]] = []

            for cat in categories:
                cat_uuid = str(_uuid.uuid4())
                cat_name = cat["name"]
                members = cat.get("members", [])

                # Create category node with BOTH :Category and :Category_{layer} labels
                await self._driver.execute_query(
                    f"CREATE (c:Category:Category_{current_layer} {{uuid: $uuid, name: $name, layer: $layer, "
                    "group_id: $gid, tag: $tag, summary: $summary})",
                    uuid=cat_uuid, name=cat_name, layer=current_layer,
                    gid=self._group_id,
                    tag=json.dumps([str(current_items[i].get("name") or "")[:60] for i in members[:5]]),
                    summary=f"{cat_name}: " + "; ".join(str(current_items[i].get("summary") or current_items[i].get("name") or "")[:160] for i in members[:5]),
                )

                for member_index in members:
                    member_uuid = current_items[member_index]["uuid"]
                    await self._driver.execute_query(
                        "MATCH (parent {uuid: $parent_uuid}) "
                        "MATCH (child {uuid: $child_uuid}) "
                        "MERGE (parent)-[:CATEGORIZES]->(child)",
                        parent_uuid=cat_uuid, child_uuid=member_uuid,
                    )

                next_layer_items.append({
                    "uuid": cat_uuid,
                    "name": cat_name,
                    "summary": f"{cat_name}: " + "; ".join(str(current_items[i].get("summary") or current_items[i].get("name") or "")[:160] for i in members[:5]),
                })

            if not next_layer_items:
                break
            self._debug_state.setdefault("hierarchy_layers", []).append({
                "layer": current_layer,
                "input_nodes": len(current_items),
                "category_nodes": len(next_layer_items),
                "covered_nodes": len(covered_indexes),
                "uncovered_nodes": len(current_items) - len(covered_indexes),
            })
            previous_categories = "\n".join(f"- {item['name']}: {item['summary']}" for item in next_layer_items)
            current_items = next_layer_items

        self._debug_state["hierarchy_layers_built"] = current_layer

    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        if self._driver is None:
            raise RuntimeError("Mnemis retrieval requested without a Neo4j driver")

        rag_top_k = _env_int("MNEMIS_RAG_TOP_K", 10)
        graph_top_k = _env_int("MNEMIS_GRAPH_TOP_K", 20)
        use_system2 = _env_bool("MNEMIS_USE_SYSTEM2", True)

        time_stats: dict[str, Any] = {}

        # System-1: official-style raw conversation RAG plus Graphiti similarity
        # fallback. The released Mnemis LoCoMo contexts expose timestamped raw
        # "Message Chunk" entries before FACTS/ENTITIES, so prefer raw chunks here.
        history_rag_results = _history_rag_candidates(query, sample.history, rag_top_k)
        graphiti_s1_results = self._system1_search(query, rag_top_k)
        s1_results = history_rag_results[:]
        seen_s1 = {str(item.get("content") or "").strip() for item in s1_results}
        for item in graphiti_s1_results:
            content = str(item.get("content") or item.get("fact") or "").strip()
            if content and content not in seen_s1:
                s1_results.append(item)
                seen_s1.add(content)
            if len(s1_results) >= rag_top_k:
                break
        time_stats["system1_history_count"] = len(history_rag_results)
        time_stats["system1_graphiti_count"] = len(graphiti_s1_results)
        time_stats["system1_count"] = len(s1_results)

        # System-2: Global selection via hierarchical graph
        s2_results: dict[str, Any] = {"episodes": [], "edges": [], "nodes": []}
        if use_system2:
            selector = self._get_global_selector()
            if selector is None:
                raise RuntimeError(
                    f"Mnemis System-2 selector unavailable: {self._debug_state.get('selector_error', 'unknown error')}"
                )
            try:
                s2_results, s2_time = self._run_async(
                    selector.global_selection(query, group_id=self._group_id)
                )
                time_stats["system2"] = s2_time
                time_stats["system2_episodes"] = len(s2_results.get("episodes", []))
                time_stats["system2_edges"] = len(s2_results.get("edges", []))
            except Exception as e:
                raise RuntimeError(f"Mnemis System-2 retrieval failed: {e}") from e

        # Optional Qwen3-Reranker pass (official Mnemis uses Qwen3-Reranker-8B to
        # reorder candidates before truncation). Reorders S2 episodes/edges and S1
        # results by relevance to the query; on any failure keeps original order.
        time_stats["reranked"] = False
        if _env_bool("MNEMIS_USE_RERANKER", False):
            time_stats["rerank_attempted"] = True
            def _rerank_list(items: list[dict[str, Any]], text_key: str) -> list[dict[str, Any]]:
                if not items:
                    return items
                texts = [str(it.get(text_key) or it.get("content") or it.get("fact") or "") for it in items]
                scores = _rerank_texts(query, texts)
                if not scores:
                    return items
                order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
                return [items[i] for i in order]
            try:
                s2_results["episodes"] = _rerank_list(s2_results.get("episodes", []), "content")
                s2_results["edges"] = _rerank_list(s2_results.get("edges", []), "fact")
                s1_results = _rerank_list(s1_results, "content")
                time_stats["reranked"] = True
            except Exception as e:
                self._debug_state["rerank_error"] = str(e)
        else:
            time_stats["rerank_attempted"] = False

        # Keep the two official retrieval budgets independent. The benchmark runner's
        # generic top_k belongs to single-route adapters and must not collapse the
        # paper's RAG Top-K + Graph Top-K contract into one shared allowance.
        faithful = _env_bool("MNEMIS_FAITHFUL_MERGE", True)
        retrieved: list[RetrievedMemory] = []
        seen_content: set[str] = set()
        rank = 1

        # Prepare S2 items: episodes, then facts (edges), then entity node summaries.
        s2_items: list[dict[str, Any]] = []
        for ep in s2_results.get("episodes", []):
            content = str(ep.get("content") or "").strip()
            valid_at = ep.get("valid_at", "")
            if content:
                s2_items.append({"content": f"[{valid_at}] {content}" if valid_at else content,
                                 "type": "episode", "uuid": ep.get("uuid", ""), "valid_at": valid_at,
                                 "raw": ep})
        for edge in s2_results.get("edges", []):
            fact = str(edge.get("fact") or "").strip()
            valid_at = edge.get("valid_at", "")
            invalid_at = edge.get("invalid_at", "")
            if fact:
                # Align with the official Mnemis FACTS format: the temporal scope is
                # rendered as a trailing "(valid_at - invalid_at)" range, NOT as a
                # "[valid_at] " prefix. A leading "[date]" prefix is read by the model
                # as the *speaking* time, so relative words still left in the fact
                # ("yesterday", "last week") get re-converted off that prefix and the
                # answer lands a day early. The trailing range avoids that ambiguity.
                display = _format_fact_with_range(fact, valid_at, invalid_at)
                s2_items.append({"content": display, "type": "fact", "uuid": edge.get("uuid", ""),
                                 "valid_at": valid_at, "invalid_at": invalid_at, "raw": edge})
        # Entity node summaries (dropped by the legacy merge). Faithful mode keeps them
        # as low-priority context after episodes/facts.
        if faithful:
            for node in s2_results.get("nodes", []):
                name = str(node.get("name") or "").strip()
                summary = str(node.get("summary") or "").strip()
                text = f"{name}: {summary}" if name and summary else (summary or name)
                if text:
                    s2_items.append({"content": text, "type": "entity", "uuid": node.get("uuid", ""),
                                     "valid_at": "", "invalid_at": "", "raw": node})

        selected_s2_items: list[dict[str, Any]] = []
        seen_s2: set[tuple[str, str]] = set()
        for item in s2_items:
            uuid = str(item.get("uuid") or "").strip()
            normalized = " ".join(str(item.get("content") or "").lower().split())
            key = (str(item.get("type") or ""), uuid or normalized)
            if not normalized or key in seen_s2:
                continue
            seen_s2.add(key)
            selected_s2_items.append(item)
            if len(selected_s2_items) >= graph_top_k:
                break

        selected_s1_results = s1_results[:rag_top_k]
        selected_s2_episodes = [item["raw"] for item in selected_s2_items if item["type"] == "episode"]
        selected_s2_edges = [item["raw"] for item in selected_s2_items if item["type"] == "fact"]
        selected_s2_nodes = [item["raw"] for item in selected_s2_items if item["type"] == "entity"]
        effective_top_k = len(selected_s1_results) + len(selected_s2_items)
        time_stats.update({
            "requested_runner_top_k": top_k,
            "rag_top_k": rag_top_k,
            "graph_top_k": graph_top_k,
            "selected_system1": len(selected_s1_results),
            "selected_system2": len(selected_s2_items),
            "effective_top_k": effective_top_k,
        })

        def _add(item_content: str, entry_id: str, *, route: str, score=None,
                 valid_at="", invalid_at="", memory_type=None,
                 source_ids: list[str] | None = None) -> bool:
            nonlocal rank
            content = str(item_content or "").strip()
            if not content or content in seen_content:
                return True
            seen_content.add(content)
            meta = {"route": route, "valid_at": valid_at, "adapter": self.system_name}
            if memory_type is not None:
                meta["memory_type"] = memory_type
            if invalid_at:
                meta["invalid_at"] = invalid_at
            retrieved.append(RetrievedMemory(
                entry_id=entry_id, content=content, score=score, rank=rank,
                source_ids=list(dict.fromkeys([*(source_ids or []), *_infer_source_ids(content, sample.history)])),
                metadata=meta,
            ))
            rank += 1
            return rank <= top_k

        if faithful:
            # Preserve both routes in full. Prompt construction still renders RAG
            # message chunks first and structured graph evidence second.
            for item in selected_s1_results:
                _add(item.get("content") or item.get("fact") or "",
                     f"mnemis:s1:{item.get('uuid', rank)}",
                     route=str(item.get("route") or "system1"),
                     score=item.get("rerank_score", item.get("score")),
                     valid_at=item.get("valid_at"),
                     source_ids=[str(source_id) for source_id in item.get("source_ids", [])])
            for it in selected_s2_items:
                _add(it["content"], f"mnemis:s2:{it['type']}:{it.get('uuid', rank)}",
                     route="system2", valid_at=it.get("valid_at", ""),
                     invalid_at=it.get("invalid_at", ""), memory_type=it["type"])
        else:
            # Legacy: interleave S1 and S2 one-by-one.
            s1_idx, s2_idx = 0, 0
            while s1_idx < len(selected_s1_results) or s2_idx < len(selected_s2_items):
                if s1_idx < len(selected_s1_results):
                    item = selected_s1_results[s1_idx]
                    s1_idx += 1
                    _add(item.get("content") or item.get("fact") or "",
                         f"mnemis:s1:{item.get('uuid', rank)}",
                         route=str(item.get("route") or "system1"),
                         score=item.get("rerank_score", item.get("score")),
                         valid_at=item.get("valid_at"),
                         source_ids=[str(source_id) for source_id in item.get("source_ids", [])])
                if s2_idx < len(selected_s2_items):
                    it = selected_s2_items[s2_idx]
                    s2_idx += 1
                    _add(it["content"], f"mnemis:s2:{it['type']}:{it.get('uuid', rank)}",
                         route="system2", valid_at=it.get("valid_at", ""),
                         invalid_at=it.get("invalid_at", ""), memory_type=it["type"])

        self._retrieval_result = RetrievalResult(
            query=query, retrieved_entries=retrieved, top_k=effective_top_k,
            raw={
                "time_stats": time_stats,
                "group_id": self._group_id,
                "s1_results": selected_s1_results,
                # Keep structured System-2 output so build_prompt can rebuild the
                # official three-section context (Episodes / FACTS / ENTITIES).
                "s2_episodes": selected_s2_episodes,
                "s2_edges": selected_s2_edges,
                "s2_nodes": selected_s2_nodes,
            },
        )
        return self._retrieval_result

    def build_prompt(self, query: str, retrieved: RetrievalResult, sample: UnifiedSample) -> PromptRecord:
        """Official Mnemis (Graphiti/Zep-style) three-section context, reconstructed
        from the structured System-2 output. Matches the `context` field format found
        in the official results JSON: Episodes (timestamped, by session) + <FACTS>
        (with valid_at-invalid_at ranges and temporal-semantics notes) + <ENTITIES>
        (name: summary). Enabled by MNEMIS_OFFICIAL_ANSWER (default true)."""
        if not _env_bool("MNEMIS_OFFICIAL_ANSWER", True):
            return super().build_prompt(query, retrieved, sample)

        raw = retrieved.raw or {}
        s1_results = raw.get("s1_results") or []
        episodes = raw.get("s2_episodes") or []
        edges = raw.get("s2_edges") or []
        nodes = raw.get("s2_nodes") or []

        parts: list[str] = [
            "# The following historical conversation is provided for your reference. "
            "If the information is useful for answering the current question, you may refer "
            "to it; otherwise, please disregard it."
        ]
        chunk_index = 0
        if s1_results:
            rag_context = _format_rag_graph_chunks([item for item in s1_results if isinstance(item, dict)])
            if rag_context:
                for block in rag_context.split("\n\n"):
                    if not block.strip():
                        continue
                    if block.startswith("Message Chunk "):
                        parts.append(re.sub(r"^Message Chunk \d+", f"Message Chunk {chunk_index}", block, count=1))
                    else:
                        parts.append(f"Message Chunk {chunk_index}:\n{block}")
                    chunk_index += 1
        for ep in episodes:
            content = str(ep.get("content") or "").strip()
            if not content:
                continue
            valid_at = ep.get("valid_at") or ""
            header = f"Message Chunk {chunk_index}"
            parts.append(f"{header}:\n{content}" if not valid_at else f"{header} [{valid_at}]:\n{content}")
            chunk_index += 1

        if edges:
            parts.append(
                "\n\nFACTS and ENTITIES represent relevant context to the current conversation."
                "\n\n# These are the most relevant facts for the conversation along with the "
                "datetime of the event that the fact refers to.\nIf a fact mentions something "
                "happening a week ago, then the datetime will be the date time of last week and "
                "not the datetime of when the fact was stated.\nTimestamps in memories represent "
                "the actual time the event occurred, not the time the event was mentioned in a message."
            )
            fact_lines = ["<FACTS>"]
            for edge in edges:
                fact = str(edge.get("fact") or "").strip()
                if not fact:
                    continue
                va = edge.get("valid_at") or ""
                iv = edge.get("invalid_at") or "now"
                fact_lines.append(f"  - {fact} ({va} - {iv})" if va else f"  - {fact}")
            fact_lines.append("</FACTS>")
            parts.append("\n".join(fact_lines))

        if nodes:
            ent_lines = ["# These are the most relevant entities", "# ENTITY_NAME: entity summary", "<ENTITIES>"]
            for node in nodes:
                name = str(node.get("name") or "").strip()
                summary = str(node.get("summary") or "").strip()
                if not name:
                    continue
                ent_lines.append(f"  - {name}: {summary}" if summary else f"  - {name}")
            ent_lines.append("</ENTITIES>")
            parts.append("\n".join(ent_lines))

        memory_context = "\n".join(parts)
        full_prompt = (
            "You are an intelligent memory assistant answering LoCoMo questions from historical "
            "conversation memories.\n\n"
            "Use the provided context as evidence. First identify the relevant message(s), facts, "
            "and timestamps, then answer the question directly. Synthesize across multiple memories "
            "when the question asks what someone would likely do, their status, identity, plans, "
            "or other inferred personal facts.\n\n"
            "Temporal reasoning is critical. Convert relative expressions to absolute dates, months, "
            "or years using the timestamp on the message or fact where the expression appears. "
            "Examples: if a 2023/05/08 message says 'yesterday', answer May 7, 2023; if a "
            "2023/05/25 message says 'last Saturday', answer May 20, 2023; if a late-May 2023 "
            "message says 'next month', answer June 2023. Do not leave answers as 'yesterday', "
            "'last week', or 'next month' when the timestamp is available.\n\n"
            "Do not over-compress the answer. Give the direct answer first, then include enough "
            "supporting detail to preserve exact names, dates, counts, list items, and the evidence "
            "chain used for inference. For list questions, include all supported items rather than "
            "only the first match. If the evidence supports an answer semantically but with different "
            "wording than the question, state the supported equivalent. Only answer 'Not enough "
            "information' when the context truly lacks evidence.\n\n"
            f"{memory_context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        self._prompt_record = PromptRecord(
            system_prompt=None,
            user_prompt=f"Question: {query}",
            memory_context=memory_context,
            full_prompt=full_prompt,
            injected_entry_ids=[item.entry_id for item in retrieved.retrieved_entries],
            token_count=self._estimate_token_count(full_prompt),
            raw={
                "adapter": self.system_name,
                "official_answer": True,
                "merge_method": "RAG_GRAPH",
                "includes_system1": bool(s1_results),
                "includes_system2": bool(episodes or edges or nodes),
            },
        )
        return self._prompt_record

    def _system1_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        try:
            from graphiti_core import Graphiti
        except ImportError:
            return []

        if self._driver is None:
            return []

        async def _search():
            graphiti = self._get_graphiti()
            results = await graphiti.search(
                query=query,
                group_ids=[self._group_id],
                num_results=top_k,
            )
            return [
                {
                    "uuid": getattr(r, "uuid", None) or str(i),
                    "content": getattr(r, "fact", None) or getattr(r, "content", None) or str(r),
                    "score": getattr(r, "score", None),
                    "valid_at": getattr(r, "valid_at", None),
                }
                for i, r in enumerate(results)
            ]

        try:
            return self._run_async(_search())
        except Exception as e:
            self._debug_state["system1_error"] = str(e)
            if (
                _env_bool("MEMORY_EVAL_FORMAL_MODE", False)
                or "TransactionStartFailed" in str(e)
                or "critical error" in str(e).lower()
            ):
                raise RuntimeError(f"Mnemis Neo4j retrieval failed critically: {e}") from e
            return []

    def validate_setup(self) -> dict[str, Any]:
        uri = _env("MNEMIS_NEO4J_URI", "bolt://localhost:7687")
        password = _env("MNEMIS_NEO4J_PASSWORD", "")

        if not password:
            return {
                "system": self.system_name,
                "ready": False,
                "provider": "neo4j+graphiti",
                "message": "MNEMIS_NEO4J_PASSWORD environment variable not set",
            }

        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=(_env("MNEMIS_NEO4J_USER", "neo4j"), password))
            driver.verify_connectivity()
            driver.close()
        except ImportError:
            return {
                "system": self.system_name,
                "ready": False,
                "provider": "neo4j+graphiti",
                "message": "neo4j package not installed (pip install neo4j)",
            }
        except Exception as e:
            return {
                "system": self.system_name,
                "ready": False,
                "provider": "neo4j+graphiti",
                "message": f"Cannot connect to Neo4j at {uri}: {e}",
            }

        try:
            import graphiti_core  # noqa: F401
        except ImportError:
            return {
                "system": self.system_name,
                "ready": False,
                "provider": "neo4j+graphiti",
                "message": "graphiti_core package not installed (pip install graphiti-core)",
            }

        return {"system": self.system_name, "ready": True, "provider": "neo4j+graphiti", "message": ""}

    def _get_graphiti(self):
        if self._graphiti is None:
            self._graphiti = _build_graphiti()
        return self._graphiti

    async def _close_resources(self) -> None:
        if self._graphiti is not None:
            try:
                await self._graphiti.close()
            finally:
                await _close_async_client(self._graphiti.llm_client)
                await _close_async_client(self._graphiti.embedder)
                await _close_async_client(self._graphiti.cross_encoder)

        selector_client = getattr(self._global_selector, "llm_client", None)
        if selector_client is not None:
            await _close_async_client(selector_client)

        if self._driver is not None:
            await self._driver.close()

        # Give transports a loop iteration to finish connection callbacks.
        await asyncio.sleep(0)

    def close(self) -> None:
        try:
            self._run_async(self._close_resources())
        except Exception:
            pass
        self._driver = None
        self._global_selector = None
        self._graphiti = None
        self._async_runner.close()

    def _group_by_session(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sessions_dict: dict[str, list[dict[str, Any]]] = {}
        for turn in history:
            sid = str(turn.get("session_id") or "default")
            sessions_dict.setdefault(sid, []).append(turn)
        return [
            {"session_id": sid, "messages": turns}
            for sid, turns in sessions_dict.items()
        ]
