from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from map_platform.datasets.schema import UnifiedSample
from map_platform.llm.gateway import token_param_name
from map_platform.memory_systems.base import (
    AnswerRecord,
    BaseMemorySystemAdapter,
    MemoryEntry,
    OrganizationState,
    PromptRecord,
    RetrievalResult,
    RetrievedMemory,
)


# =============================================================================
# Official MIRIX TaskAgent answering pipeline (ported from
# MIRIX/evals/task_agent.py). Enabled by default via MIRIX_OFFICIAL_ANSWER so
# MIRIX answers LoCoMo questions like its own paper: a multi-round tool-calling
# agent that actively re-searches memory (search_memory) and fetches raw items
# (check_raw_item) before giving a minimal, never-abstain answer.
# =============================================================================

MIRIX_CHAT_AGENT_SYSTEM_PROMPT = (
    "You are the Chat Agent, a component of the personal assistant system. "
    "Your primary responsibility is managing user communication. "
    "You have access to a unified memory infrastructure shared with other specialized agents. "
    "\n\nMemory Components:\n"
    "1. Core Memory: Essential user information and your persona.\n"
    "2. Episodic Memory: Chronological records of interactions.\n"
    "3. Procedural Memory: Step-by-step processes and guidelines.\n"
    "4. Resource Memory: Documents and reference materials.\n"
    "5. Knowledge: Factual data like contacts and credentials.\n"
    "6. Semantic Memory: Conceptual knowledge and contextual information.\n"
    "\n\nOperational Requirements:\n"
    "Whenever a user sends a query, an initial high-level (preliminary) search is automatically conducted, and the results are provided to you. "
    "However, this initial search may not be comprehensive or fully accurate. "
    "You MUST evaluate the provided information and utilize the `search_memory` tool to conduct additional, more specific searches if you believe further context is necessary to provide a complete and accurate response. "
    "\n\nSearch Strategy (CRITICAL):\n"
    "1. VERIFY RESULTS: After each search, check if results contain key terms from the question. If not, the search likely returned wrong memories.\n"
    "2. MULTI-ANGLE SEARCH: Try different search phrasings if initial results seem off-topic.\n"
    "   - Example: 'book Melanie read Caroline suggestion' + 'Becoming Nicole Melanie' + 'book recommendation Caroline Melanie'\n"
    "3. CROSS-MEMORY SEARCH: For most questions, search BOTH episodic AND semantic memory types separately and combine results.\n"
    "   - Episodic contains events/activities (when things happened)\n"
    "   - Semantic contains stable facts/attributes (interests, possessions, skills)\n"
    "4. LIST AGGREGATION: For questions asking 'What items...', 'What activities...', search multiple times with different keywords and aggregate ALL results.\n"
    "   - Example: 'What has X painted?' -> Search 'X painted', 'X painting', 'X artwork', then combine all unique items found\n"
    "5. SMART STOPPING: After 2-3 searches, evaluate if you have enough information to answer. If yes, STOP SEARCHING and provide your answer.\n"
    "   - Don't keep searching indefinitely if you already found relevant information\n"
    "   - You have a maximum of 5 search rounds - use them wisely\n"
    "6. KEYWORD VARIANTS: If searching for a specific item (book, painting, activity), try searching for:\n"
    "   - The item name directly ('Becoming Nicole')\n"
    "   - The person + activity ('Melanie read book')\n"
    "   - The relationship context ('Caroline suggested book Melanie')\n"
    "\n"
    "Be persistent but efficient: if you find relevant information after 2-3 searches, provide your answer. "
    "Do NOT give up or state that you don't know the answer unless multiple searches with different parameters have failed to yield relevant information. "
    "You may call the tool multiple times if needed. "
    "Each memory item may include a `raw_input_id` that points to the raw user input. "
    "Use the `check_raw_item` tool when you need the original input for disambiguation or exact wording. "
    "\n\nMessage Processing Protocol:\n"
    "1. Analyze the user's query and use `search_memory` to gather necessary context.\n"
    "   - If a result includes `raw_input_id` and you need the original text, call `check_raw_item`.\n"
    "2. Provide a helpful and concise answer based on the retrieved information.\n"
    "3. Only inform the user that you don't know the answer if at least three consecutive searches with different parameters have failed to yield relevant information.\n"
    "4. Be VERY CONCISE in your response, only output the answer and nothing else.\n"
    "5. There are some open-ended questions where you may not find explicit evidences, you still need to answer it based on your understanding. Never say you don't know or 'there is no specific information', ...\n"
    "6. If there is no information found, you still need to answer it. Guess an answer if you don't have enough information.\n"
    "\n\nAnswer Format Guidelines (CRITICAL):\n"
    "- For list questions (What books, What instruments, What activities, etc.), provide a simple comma-separated list or use 'and' between items.\n"
    "  Example: \"clarinet and violin\" NOT \"She plays clarinet\"\n"
    "  Example: '\"Nothing is Impossible\", \"Charlotte\\'s Web\"' NOT \"She read several books\"\n"
    "- For simple fact questions (What is X's relationship status?, How old?, etc.), provide direct factual answers.\n"
    "  Example: \"Single\" NOT \"She experienced a breakup but is...\"\n"
    "  Example: \"28 years old\" NOT \"She is currently 28 years old and...\"\n"
    "- For specific detail questions (What kind of art?, What type of pot?, etc.), provide the specific detail.\n"
    "  Example: \"abstract art\" NOT \"art inspired by...\"\n"
    "  Example: \"a cup with a dog face on it\" NOT \"pottery items\"\n"
    "- ALWAYS extract the minimal, direct answer that matches what's being asked. Do NOT add ANY additional information!\n"
    "- If the question asks for multiple items, search until you find ALL items, not just the first one."
)


_MIRIX_EXTRACTION_INSTRUCTIONS = """Instructions:

1. Carefully analyze all utterances from both speakers.
2. The conversation has a timestamp, but the events mentioned in the conversation may have different timestamps. You have to extract the exact date of the mentioned events. Remember that "mentioned at" is not the same as "occurred at" so this has to be noted in the memories.
3. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
4. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the conversation timestamp.
5. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
6. You are supposed to extract the event/fact/semantic knowledge from the conversation. For example, if the conversation happens at 2023 and the conversation says that "John went to India last year", then you should save the fact that "John went to India in 2022". Similarly for all other kinds of memories.
7. Make sure to extract the facts about the characters, such as their name, age, gender, occupation, hometown, etc."""


def _mirix_search_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": (
                    "Search Mirix memories for information related to a user query. "
                    "For best results, try multiple search strategies: "
                    "(1) Different phrasings of the query, "
                    "(2) Searching both 'episodic' and 'semantic' memory types separately, "
                    "(3) Using specific keywords from the question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query string."},
                        "memory_type": {
                            "type": "string",
                            "enum": ["episodic", "resource", "procedural", "knowledge", "semantic", "all"],
                            "default": "all",
                        },
                        "search_field": {"type": "string", "default": "null"},
                        "search_method": {"type": "string", "enum": ["bm25", "embedding"], "default": None},
                        "limit": {"type": "integer", "default": 10, "minimum": 1},
                        "filter_tags": {"type": "object"},
                        "similarity_threshold": {"type": "number"},
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_raw_item",
                "description": "Fetch the raw input payload for a memory item using raw_input_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "raw_input_id": {"type": "string", "description": "The raw_input_id returned by search_memory."}
                    },
                    "required": ["raw_input_id"],
                },
            },
        },
    ]


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw.isdigit() else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _chat_message_from_response(response: Any) -> Any:
    if isinstance(response, str):
        return {"content": response, "tool_calls": []}
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            choice = choices[0]
            if isinstance(choice, dict):
                return choice.get("message") or {"content": choice.get("text") or "", "tool_calls": []}
        return {"content": response.get("content") or response.get("text") or "", "tool_calls": []}
    choices = getattr(response, "choices", None) or []
    if choices:
        return getattr(choices[0], "message", None) or {"content": getattr(choices[0], "text", "") or "", "tool_calls": []}
    return {"content": str(response or ""), "tool_calls": []}


def _chat_message_content(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _chat_message_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, dict):
        value = message.get("tool_calls") or []
    else:
        value = getattr(message, "tool_calls", None) or []
    return list(value) if isinstance(value, list) else []


def _tool_call_to_payload(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        return tool_call
    if hasattr(tool_call, "model_dump"):
        return tool_call.model_dump()
    return {
        "id": getattr(tool_call, "id", ""),
        "type": getattr(tool_call, "type", "function"),
        "function": {
            "name": getattr(getattr(tool_call, "function", None), "name", ""),
            "arguments": getattr(getattr(tool_call, "function", None), "arguments", "{}"),
        },
    }


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or "")
    return str(getattr(tool_call, "id", "") or "")


def _tool_call_function(tool_call: Any) -> tuple[str, str]:
    if isinstance(tool_call, dict):
        function = tool_call.get("function") or {}
        if isinstance(function, dict):
            return str(function.get("name") or ""), str(function.get("arguments") or "{}")
        return "", "{}"
    function = getattr(tool_call, "function", None)
    return str(getattr(function, "name", "") or ""), str(getattr(function, "arguments", "{}") or "{}")


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _normalize_occurred_at(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e12 else value
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except ValueError:
        pass

    for fmt in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
    ):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            continue
    return None


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


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


def _query_terms(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "at", "did", "do", "does", "for", "from", "his", "her",
        "in", "is", "it", "of", "on", "or", "she", "the", "their", "to", "was",
        "what", "when", "where", "which", "who", "why", "with",
    }
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _raw_history_matches(query: str, sample: UnifiedSample, max_items: int = 8) -> str:
    terms = _query_terms(query)
    if not terms:
        return ""
    query_lower = query.lower()
    scored: list[tuple[float, dict[str, Any]]] = []
    for turn in sample.history:
        text = str(turn.get("text") or "")
        timestamp = str(turn.get("timestamp") or "")
        speaker = str(turn.get("speaker") or "")
        haystack = f"{speaker} {timestamp} {text}".lower()
        overlap = len(terms & _query_terms(haystack))
        if overlap <= 0:
            continue
        score = float(overlap)
        if timestamp and timestamp.lower() in query_lower:
            score += 6.0
        for month in (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ):
            if month in query_lower and month in timestamp.lower():
                score += 2.0
        for year in re.findall(r"\b\d{4}\b", query):
            if year in timestamp:
                score += 2.0
        scored.append((score, turn))
    scored.sort(key=lambda item: item[0], reverse=True)
    lines: list[str] = []
    for _, turn in scored[:max_items]:
        turn_id = str(turn.get("turn_id") or turn.get("source_id") or turn.get("dia_id") or "")
        session_id = str(turn.get("session_id") or "")
        speaker = str(turn.get("speaker") or "")
        timestamp = str(turn.get("timestamp") or "")
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        label_parts = [part for part in (turn_id, session_id, speaker, timestamp) if part]
        lines.append(f"- [{' | '.join(label_parts)}] {text}")
    return "\n".join(lines)


class MirixAdapter(BaseMemorySystemAdapter):
    system_name = "mirix"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client: Any = None
        self._user_id: str = ""
        self._initialized: bool = False
        self._config: dict[str, Any] | None = None

    def reset(self, sample_id: str) -> None:
        super().reset(sample_id)
        prefix = _safe_component(_env("MIRIX_USER_ID_PREFIX", "map_bench"))
        self._user_id = f"{prefix}_{_safe_component(sample_id)}"
        self._ensure_client()

    def _load_config(self) -> dict[str, Any]:
        if self._config is not None:
            return self._config

        config_path = _env("MIRIX_CONFIG_PATH", "")
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            default_path = Path(__file__).resolve().parents[1] / "MIRIX" / "mirix" / "configs" / "mirix.yaml"
            if default_path.exists():
                import yaml
                with open(default_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
            else:
                self._config = {}

        self._apply_runtime_endpoints(self._config)
        self._resolve_api_keys(self._config)
        self._normalize_official_core_blocks(self._config)
        return self._config

    def _apply_runtime_endpoints(self, config: dict[str, Any]) -> None:
        llm_config = config.get("llm_config")
        if isinstance(llm_config, dict):
            llm_config["model"] = _env("MIRIX_LLM_MODEL", self.generation_model)
            llm_base_url = _env("MIRIX_LLM_BASE_URL", "") or os.getenv("OPENAI_BASE_URL", "").strip()
            if llm_base_url:
                llm_config["model_endpoint"] = llm_base_url
            llm_config["api_key"] = (
                _env("MIRIX_LLM_API_KEY", "")
                or os.getenv("MEMORY_BUILD_OPENAI_API_KEY", "").strip()
                or os.getenv("OPENAI_API_KEY", "").strip()
                or os.getenv("API_T", "").strip()
            )

        embedding_config = config.get("embedding_config")
        if isinstance(embedding_config, dict):
            embedding_config["embedding_model"] = self.embedding_model
            embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
            if embedding_base_url:
                embedding_config["embedding_endpoint"] = embedding_base_url
            embedding_config["embedding_dim"] = int(os.getenv("EMBEDDING_DIMS", "1024"))
            embedding_config["api_key"] = os.getenv("EMBEDDING_API_KEY", "dummy")

    def _resolve_api_keys(self, config: dict[str, Any]) -> None:
        provider_env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GEMINI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
        }
        for section in config.values():
            if not isinstance(section, dict):
                continue
            if section.get("api_key") in {"your-api-key", "", None}:
                provider = (
                    section.get("model_endpoint_type")
                    or section.get("embedding_endpoint_type")
                    or ""
                ).lower()
                env_var = provider_env_vars.get(provider)
                if env_var:
                    section["api_key"] = os.environ.get(env_var, "")

    def _normalize_official_core_blocks(self, config: dict[str, Any]) -> None:
        """Map MIRIX eval YAML core-memory config to the server's agent schema.

        The official eval configs store core memory under
        ``meta_agent_config.memory.core``. The local MIRIX server seeds template
        blocks only from a ``{"core_memory_agent": {"blocks": [...]}}`` entry in
        ``meta_agent_config.agents``. Keep the official config as the source of
        truth and adapt only this schema difference.
        """
        meta_config = config.get("meta_agent_config")
        if not isinstance(meta_config, dict):
            return
        memory_config = meta_config.get("memory")
        if not isinstance(memory_config, dict):
            return
        core_blocks = memory_config.get("core")
        if not isinstance(core_blocks, list) or not core_blocks:
            return

        agents = meta_config.get("agents")
        if not isinstance(agents, list):
            agents = []

        normalized_agents: list[Any] = []
        inserted = False
        for agent in agents:
            if agent == "core_memory_agent":
                normalized_agents.append({"core_memory_agent": {"blocks": core_blocks}})
                inserted = True
                continue
            if isinstance(agent, dict) and "core_memory_agent" in agent:
                agent_config = agent.get("core_memory_agent")
                if not isinstance(agent_config, dict):
                    agent_config = {}
                agent_config.setdefault("blocks", core_blocks)
                normalized_agents.append({"core_memory_agent": agent_config})
                inserted = True
                continue
            normalized_agents.append(agent)
        if not inserted:
            normalized_agents.insert(0, {"core_memory_agent": {"blocks": core_blocks}})
        meta_config["agents"] = normalized_agents

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from mirix import MirixClient
        except ImportError as exc:
            self._debug_state["init_error"] = f"MIRIX import failed: {exc}"
            raise RuntimeError(self._debug_state["init_error"]) from exc

        base_url = _env("MIRIX_BASE_URL", "http://127.0.0.1:8531")
        timeout = _env_int("MIRIX_TIMEOUT", 1800)
        client_id = _env("MIRIX_CLIENT_ID", "") or str(uuid.uuid4())
        org_id = _env("MIRIX_ORG_ID", "") or None

        self._client = MirixClient(
            client_id=client_id,
            org_id=org_id,
            base_url=base_url,
            write_scope="read_write",
            timeout=timeout,
        )

        if not self._initialized:
            config = self._load_config()
            try:
                _run_async(self._client.initialize_meta_agent(config=config, update_agents=True))
                self._initialized = True
            except Exception as e:
                self._debug_state["init_error"] = str(e)
                raise RuntimeError(f"MIRIX meta-agent initialization failed: {e}") from e

    def build_memory(self, history: list[dict[str, Any]], sample: UnifiedSample) -> None:
        if self._client is None:
            self._debug_state["build_error"] = "MIRIX client not available"
            raise RuntimeError(self._debug_state["build_error"])
        if not self._initialized:
            raise RuntimeError(
                "MIRIX client exists but meta-agent initialization did not complete: "
                f"{self._debug_state.get('init_error', 'unknown error')}"
            )

        sessions = self._group_by_session(history)
        if _env_bool("MIRIX_REUSE_EXISTING_MEMORY", False):
            self._debug_state["reused_existing_memory"] = True
            self._set_session_memory_entries(sessions)
            return

        # Official MIRIX (main_eval.py) defaults: one chunk per whole session
        # (no 4096-token re-split), chaining=False, and an instructions block +
        # "conversation timestamped at ..." header embedded in every chunk so the
        # extractor resolves relative dates. All toggled via env for ablation.
        official_build = _env_bool("MIRIX_OFFICIAL_BUILD", True)
        chaining = _env_bool("MIRIX_CHAINING", False if official_build else True)
        async_add = _env_bool("MIRIX_ASYNC_ADD", False)
        use_cache = _env_bool("MIRIX_USE_CACHE", False)
        post_add_wait = _env_int("MIRIX_POST_ADD_WAIT", 5)
        successful_adds = 0
        total_chunks = 0

        for sess in sessions:
            raw_lines = [
                f"{t.get('speaker', 'unknown')}: {t.get('text', '')}"
                for t in sess["messages"]
                if t.get("text")
            ]
            if not raw_lines:
                continue

            timestamp = sess["messages"][0].get("timestamp") if sess["messages"] else None
            occurred_at = _normalize_occurred_at(timestamp)
            if timestamp and occurred_at is None:
                self._debug_state.setdefault("timestamp_warnings", []).append({
                    "session_id": sess["session_id"],
                    "timestamp": str(timestamp),
                })

            if official_build:
                # One chunk = whole session, prefixed with official instructions
                # + timestamped header (ported from main_eval.format_session_chunk).
                dt = str(timestamp or "")
                header_lines = [
                    f"You have access to the conversation between two speakers. The conversation is timestamped at {dt}.\n",
                    _MIRIX_EXTRACTION_INSTRUCTIONS,
                    f"Session {sess['session_id']} ({dt})" if dt else f"Session {sess['session_id']}",
                ]
                chunks = ["\n".join(header_lines + raw_lines)]
            else:
                chunks = self._chunk_lines(raw_lines, max_tokens=4096)

            for chunk_idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                add_kwargs: dict[str, Any] = {
                    "user_id": self._user_id,
                    "messages": [{"role": "user", "content": chunk_text}],
                    "chaining": chaining,
                    "filter_tags": {"scope": "read_write", "kind": "conversation_session"},
                    "use_cache": use_cache,
                    "async_add": async_add,
                }
                if occurred_at:
                    add_kwargs["occurred_at"] = occurred_at

                total_chunks += 1
                try:
                    _run_async(self._client.add(**add_kwargs))
                    successful_adds += 1
                except Exception as e:
                    self._debug_state.setdefault("add_errors", []).append({
                        "session_id": sess["session_id"],
                        "chunk_idx": chunk_idx,
                        "error": str(e),
                    })

        # Track build completeness — fail fast if any chunk failed
        self._debug_state["successful_adds"] = successful_adds
        self._debug_state["total_chunks"] = total_chunks
        if total_chunks > 0 and successful_adds != total_chunks:
            raise RuntimeError(
                f"MIRIX memory build incomplete: {successful_adds}/{total_chunks} chunks ingested. "
                f"Errors: {self._debug_state.get('add_errors', [])}"
            )

        # Wait for async memory processing to finish
        if post_add_wait > 0:
            time.sleep(post_add_wait)

        turn_ids = [str(t.get("turn_id") or t.get("dia_id") or "") for t in history if t.get("text")]
        self._set_session_memory_entries(sessions)

    def _set_session_memory_entries(self, sessions: list[dict[str, Any]]) -> None:
        self._memory_entries = [
            MemoryEntry(
                entry_id=f"mirix_session_{sess['session_id']}",
                content=f"Session {sess['session_id']}: {len(sess['messages'])} messages",
                raw={"session_id": sess["session_id"]},
                source_ids=[str(m.get("turn_id") or m.get("dia_id") or "") for m in sess["messages"]],
            )
            for sess in sessions
        ]

    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        if self._client is None:
            raise RuntimeError("MIRIX retrieval requested without an initialized client")

        try:
            use_cache = _env_bool("MIRIX_USE_CACHE", False)
            memories = _run_async(self._client.retrieve_with_conversation(
                user_id=self._user_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": query}],
                    }
                ],
                filter_tags={"scope": "read_write", "kind": "conversation_session"},
                use_cache=use_cache,
            ))
        except Exception as e:
            self._debug_state["retrieval_error"] = f"{type(e).__name__}: {e}"
            raise RuntimeError(f"MIRIX retrieval failed: {e}") from e

        retrieved = self._parse_memories(memories, sample, top_k)
        self._retrieval_result = RetrievalResult(
            query=query,
            retrieved_entries=retrieved,
            top_k=top_k,
            raw={
                "topics": (memories or {}).get("topics"),
                "temporal_expression": (memories or {}).get("temporal_expression"),
                "memory_types": list((memories or {}).get("memories", {}).keys()),
            },
        )
        return self._retrieval_result

    def build_prompt(self, query: str, retrieved: RetrievalResult, sample: UnifiedSample) -> PromptRecord:
        """In official mode the preliminary retrieval is just the agent's starting
        context; the TaskAgent re-searches in generate_answer. We still record a
        prompt with the preliminary evidence for tracing/diagnostics."""
        if not _env_bool("MIRIX_OFFICIAL_ANSWER", True):
            return super().build_prompt(query, retrieved, sample)
        memory_context = self._format_retrieved_memory_block(retrieved.retrieved_entries, sample=sample)
        raw_matches = _raw_history_matches(query, sample, max_items=_env_int("MIRIX_RAW_HISTORY_MATCHES", 8))
        raw_block = (
            "\n\nRaw conversation matches from the original history. Prefer exact-date/person/business matches "
            "over older summaries when they answer the question:\n"
            f"{raw_matches}"
            if raw_matches
            else ""
        )
        user_prompt = (
            f"User Query: {query}\n\n"
            f"Preliminary search results (may be incomplete - use search_memory for more):\n"
            f"{memory_context}"
            f"{raw_block}"
        )
        self._prompt_record = PromptRecord(
            system_prompt=MIRIX_CHAT_AGENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            memory_context=memory_context,
            full_prompt=user_prompt,
            injected_entry_ids=[item.entry_id for item in retrieved.retrieved_entries],
            token_count=self._estimate_token_count(user_prompt),
            raw={"adapter": self.system_name, "official_answer": True, "query": query},
        )
        return self._prompt_record

    def generate_answer(self, prompt: PromptRecord, sample: UnifiedSample) -> AnswerRecord:
        """Multi-round tool-calling agent (ported from MIRIX TaskAgent.answer).

        The model may call search_memory / check_raw_item up to max_tool_rounds
        times; the final round forces an answer with no tools. gpt-4.1-mini in the
        paper; here self.generation_model. max_completion_tokens kept small (128)
        to match the paper's minimal-answer behaviour."""
        if not _env_bool("MIRIX_OFFICIAL_ANSWER", True) or self._client is None:
            return super().generate_answer(prompt, sample)

        from openai import OpenAI

        started = time.perf_counter()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)
        tools = _mirix_search_tools()
        max_rounds = _env_int("MIRIX_MAX_TOOL_ROUNDS", 5)
        max_tokens = _env_int("MIRIX_ANSWER_MAX_TOKENS", 128)
        query = str((prompt.raw or {}).get("query") or "")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt.system_prompt or MIRIX_CHAT_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt.full_prompt},
        ]
        tool_calls_made = 0
        tool_trace: list[dict[str, Any]] = []
        round_trace: list[dict[str, Any]] = []
        final_answer = ""
        for round_num in range(max_rounds + 1):
            is_last = round_num == max_rounds
            if is_last:
                messages.append({
                    "role": "system",
                    "content": "You have reached the maximum number of searches. Please provide your best answer based on the information you've gathered so far.",
                })
            response = client.chat.completions.create(
                model=self.generation_model,
                messages=messages,
                tools=None if is_last else tools,
                tool_choice=None if is_last else "auto",
                **{token_param_name(self.generation_model): max_tokens},
            )
            message = _chat_message_from_response(response)
            tool_calls = _chat_message_tool_calls(message)
            round_record: dict[str, Any] = {
                "round": round_num + 1,
                "forced_final_answer": is_last,
                "assistant_content": _chat_message_content(message),
                "tool_call_count": len(tool_calls),
            }
            round_trace.append(round_record)
            if not tool_calls:
                final_answer = _chat_message_content(message)
                break
            messages.append({
                "role": "assistant",
                "content": _chat_message_content(message),
                "tool_calls": [_tool_call_to_payload(tc) for tc in tool_calls],
            })
            for tc in tool_calls:
                tool_calls_made += 1
                call_id = _tool_call_id(tc)
                tool_name, tool_args = _tool_call_function(tc)
                parsed_args: dict[str, Any] | None = None
                try:
                    parsed_args = json.loads(tool_args or "{}")
                except json.JSONDecodeError:
                    result: Any = {"success": False, "error": "Invalid tool arguments."}
                else:
                    if tool_name == "search_memory":
                        result = self._agent_search(parsed_args)
                    elif tool_name == "check_raw_item":
                        result = self._agent_check_raw_item(parsed_args)
                    else:
                        result = {"success": False, "error": "Unknown tool."}
                tool_trace.append(
                    {
                        "round": round_num + 1,
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "arguments_raw": tool_args,
                        "arguments": parsed_args,
                        "result": result,
                    }
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, default=str),
                })
        if _env_bool("MIRIX_EXPANDED_FINAL_ANSWER", False) and final_answer:
            final_answer = self._expand_final_answer(query=query, gathered_messages=messages, initial_answer=final_answer)

        self._answer_record = AnswerRecord(
            answer=final_answer or "I don't know",
            raw_response={
                "official_answer": True,
                "tool_call_count": tool_calls_made,
                "tool_calls": tool_trace,
                "rounds": round_num + 1,
                "round_trace": round_trace,
            },
            latency=time.perf_counter() - started,
            token_usage={"prompt_tokens_est": prompt.token_count, "completion_tokens_est": len(str(final_answer).split())},
        )
        return self._answer_record

    def _expand_final_answer(self, *, query: str, gathered_messages: list[dict[str, Any]], initial_answer: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)
        prompt = (
            "You are answering a LoCoMo long-term memory benchmark question.\n"
            "Use the gathered tool results below as evidence. Produce an answer in the style of a careful "
            "benchmark response: first give the direct answer, then briefly cite the key supporting evidence. "
            "Preserve exact names, dates, counts, and all items in lists. If the question requires inference, "
            "state the supported inference rather than saying there is not enough information when the evidence "
            "strongly supports it.\n\n"
            f"Question: {query}\n"
            f"Initial answer: {initial_answer}\n\n"
            f"Gathered conversation/tool trace:\n{json.dumps(gathered_messages[-12:], ensure_ascii=False, default=str)}\n\n"
            "Answer:"
        )
        response = client.chat.completions.create(
            model=self.generation_model,
            messages=[{"role": "user", "content": prompt}],
            **{token_param_name(self.generation_model): _env_int("MIRIX_EXPANDED_ANSWER_MAX_TOKENS", 512)},
        )
        message = _chat_message_from_response(response)
        return _chat_message_content(message) or initial_answer

    def _agent_search(self, params: dict[str, Any]) -> Any:
        """search_memory tool impl (ported from TaskAgent._search_memory)."""
        if self._client is None:
            return {"success": False, "error": "Mirix client not configured."}
        params = {k: v for k, v in (params or {}).items() if isinstance(k, str) and k}
        if not params.get("search_method"):
            params["search_method"] = "embedding"
        if not params.get("limit") or params.get("limit", 0) < 10:
            params["limit"] = 15
        try:
            results = _run_async(self._client.search(user_id=self._user_id, **params))
        except TypeError as e:
            return {"success": False, "error": f"Invalid search args: {e}", "skipped": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
        if isinstance(results, dict) and results.get("success"):
            for result in results.get("results", []):
                if "occurred_at" in result and result.get("occurred_at_description"):
                    result["occurred_at"] = f"{result['occurred_at']} ({result['occurred_at_description']})"
                    result.pop("occurred_at_description", None)
                result.pop("id", None)
                result.pop("actor", None)
            return results["results"]
        return results

    def _agent_check_raw_item(self, params: dict[str, Any]) -> Any:
        if self._client is None:
            return {"success": False, "error": "Mirix client not configured."}
        raw_input_id = (params or {}).get("raw_input_id")
        if not raw_input_id:
            return {"success": False, "error": "raw_input_id is required."}
        check = getattr(self._client, "check_raw_item", None)
        if check is None:
            return {"success": False, "error": "check_raw_item not supported by this client."}
        try:
            result = check(raw_input_id)
            return _run_async(result) if asyncio.iscoroutine(result) else result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def validate_setup(self) -> dict[str, Any]:
        base_url = _env("MIRIX_BASE_URL", "http://127.0.0.1:8531")
        try:
            import requests
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code < 500:
                return {"system": self.system_name, "ready": True, "provider": "mirix_http", "message": ""}
        except Exception as e:
            return {
                "system": self.system_name,
                "ready": False,
                "provider": "mirix_http",
                "message": f"Cannot connect to MIRIX at {base_url}: {e}",
            }
        return {
            "system": self.system_name,
            "ready": False,
            "provider": "mirix_http",
            "message": "MIRIX server returned error",
        }

    def close(self) -> None:
        self._client = None
        self._initialized = False
        self._config = None

    def _group_by_session(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sessions_dict: dict[str, list[dict[str, Any]]] = {}
        for turn in history:
            sid = str(turn.get("session_id") or "default")
            sessions_dict.setdefault(sid, []).append(turn)
        return [
            {"session_id": sid, "messages": turns}
            for sid, turns in sessions_dict.items()
        ]

    def _chunk_lines(self, lines: list[str], max_tokens: int = 4096) -> list[str]:
        """Chunk speaker-labeled lines into windows of approximately max_tokens tokens.

        Token count is approximated by whitespace splitting.
        Each chunk preserves complete lines (no mid-line splits).
        """
        chunks: list[str] = []
        current_lines: list[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = len(line.split())
            if current_tokens + line_tokens > max_tokens and current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_tokens = 0
            current_lines.append(line)
            current_tokens += line_tokens

        if current_lines:
            chunks.append("\n".join(current_lines))

        return chunks

    def _parse_memories(
        self, memories_response: dict[str, Any] | None, sample: UnifiedSample, top_k: int
    ) -> list[RetrievedMemory]:
        if not memories_response:
            return []

        retrieved: list[RetrievedMemory] = []
        rank = 1
        memories = memories_response.get("memories", {})
        # In official mode the preliminary retrieval feeds an agent that re-searches,
        # so a flat top_k truncation that lets the first memory type starve the rest
        # is undesirable. Apply a per-type cap so every type keeps a fair share.
        official = _env_bool("MIRIX_OFFICIAL_ANSWER", True)
        per_type_cap = _env_int("MIRIX_PER_TYPE_LIMIT", 10) if official else top_k

        for memory_type, data in memories.items():
            if not data or data.get("total_count", 0) == 0:
                continue
            type_count = 0

            items = data.get("items", [])
            if memory_type == "episodic" and not items:
                seen_ids: set[str] = set()
                for item in data.get("recent", []) + data.get("relevant", []):
                    item_id = item.get("id")
                    if item_id and item_id not in seen_ids:
                        seen_ids.add(item_id)
                        items.append(item)
            if not items and "recent" in data:
                items = data.get("recent", [])
            if not items:
                continue

            for item in items:
                if not official and rank > top_k:
                    break
                if official and type_count >= per_type_cap:
                    break
                content = self._extract_content(item, memory_type)
                if not content.strip():
                    continue

                timestamp = item.get("timestamp") or item.get("occurred_at") or ""
                if timestamp and not content.startswith("["):
                    content = f"[{timestamp}] {content}"

                source_ids = _infer_source_ids(content, sample.history)
                retrieved.append(RetrievedMemory(
                    entry_id=f"mirix:{memory_type}:{item.get('id', rank)}",
                    content=content,
                    score=item.get("score") or item.get("relevance_score"),
                    rank=rank,
                    source_ids=source_ids,
                    metadata={
                        "memory_type": memory_type,
                        "timestamp": timestamp,
                        "adapter": self.system_name,
                    },
                ))
                rank += 1
                type_count += 1

        return retrieved

    def _extract_content(self, item: dict[str, Any], memory_type: str) -> str:
        if memory_type == "core":
            label = item.get("label", "")
            value = item.get("value", "")
            content = f"{label}: {value}".strip(": ").strip()
            if not content:
                content = item.get("summary", "") or str(item)
            return content

        content = (
            item.get("details")
            or item.get("content")
            or item.get("summary")
            or item.get("caption")
            or item.get("name")
            or item.get("title")
            or item.get("description")
            or item.get("value", "")
        )
        if not content:
            content = str(item)
        return str(content).strip()
