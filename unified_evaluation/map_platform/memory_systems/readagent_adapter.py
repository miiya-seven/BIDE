from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any

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


class ReadAgentAdapter(BaseMemorySystemAdapter):
    system_name = "readagent"

    # Verbatim QMSum-style prompts from the official ReadAgent demo notebook.
    # Dialogue histories are mapped to transcript pages by this benchmark adapter.
    _GIST_PROMPT = (
        "Please shorten the following passage.\n"
        "Just give a shortened version. DO NOT explain your reasoning.\n\n"
        "Passage:\n{page_text}"
    )
    _PARALLEL_LOOKUP_PROMPT = (
        "The following text is what you remember from reading a meeting transcript, followed by a question about the transcript.\n"
        "You may read 1 or 2 pages of the transcript again to refresh your memory to prepare to answer the question.\n"
        "Please respond with which page(s) you would like to read.\n"
        "For example, if you would only like to read Page 8, respond with \"I want to look up Page [8] ...\"\n"
        "If you would like to read Page 7 and 12, respond with \"I want to look up Page [7, 12] ...\".\n"
        "Only select as many pages as you need, but no more than 2 pages.\n"
        "Don't answer the question yet.\n\n"
        "Text:\n{gist_block}\nEnd of text.\n\n"
        "Question:\n{query}\n\nWhich page(s) would you like to look up?"
    )
    _SEQUENTIAL_LOOKUP_PROMPT = (
        "The following text is what you remember from reading a meeting transcript, followed by a question about the transcript.\n"
        "You may read multiple pages of the transcript again to refresh your memory and prepare to answer the question.\n"
        "Each page that you re-read can significantly improve your chance of answering the question correctly.\n"
        "Please specify a SINGLE page you would like to read again or say \"STOP\".\n"
        "To read a page again, respond with \"Page $PAGE_NUM\", replacing $PAGE_NUM with the target page number.\n"
        "You can only specify a SINGLE page in your response at this time.\n"
        "DO NOT select more pages if you don't need to.\nTo stop, simply say \"STOP\".\n"
        "DO NOT answer the question in your response.\n\n"
        "Text:\n{gist_block}\nEnd of text.\n\n"
        "Pages re-read already (DO NOT ask to read them again):\n{selected_pages}\n\n"
        "Question:\n{query}\n\nSpecify a SINGLE page to read again, or say STOP:"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._gist_entries: list[MemoryEntry] = []
        self._page_entries: list[MemoryEntry] = []

    def reset(self, sample_id: str) -> None:
        super().reset(sample_id)
        self._gist_entries = []
        self._page_entries = []
        self._ensure_llm_ready()
        self._debug_state["official_source"] = {
            "site": "https://read-agent.github.io/",
            "repo": "https://github.com/read-agent/read-agent.github.io",
            "commit": "569dff3e47bb708ad084d267ac554f2534d21756",
            "local_snapshot": "external_systems/ReadAgent",
            "implementation_mode": "official_workflow_adapted_to_conversation_history",
            "official_notebook": "assets/read_agent_demo.ipynb",
            "gist_mode": self._gist_mode(),
            "lookup_mode": self._lookup_mode(),
            "variant": self._variant(),
            "pagination": "token_page_adapter_for_dialogue_history",
        }

    def build_memory(self, history: list[dict[str, Any]], sample: UnifiedSample) -> None:
        pages = self._paginate_history(history)
        gist_entries: list[MemoryEntry] = []
        page_entries: list[MemoryEntry] = []
        traces: list[dict[str, Any]] = []

        for page_id, page_turns in pages:
            page_text = self._render_page(page_turns)
            gist_text = self._make_gist(page_text)
            gist_entry = MemoryEntry(
                entry_id=f"{page_id}:gist",
                content=gist_text,
                raw={"page_id": page_id, "kind": "gist", "page_text": page_text},
                created_at=self._page_timestamp(page_turns),
                updated_at=self._page_timestamp(page_turns),
                source_ids=[str(turn.get("turn_id") or "") for turn in page_turns if str(turn.get("turn_id") or "").strip()],
                metadata={"memory_bucket": "gist", "page_id": page_id},
            )
            page_entry = MemoryEntry(
                entry_id=f"{page_id}:page",
                content=page_text,
                raw={"page_id": page_id, "kind": "raw_page"},
                created_at=self._page_timestamp(page_turns),
                updated_at=self._page_timestamp(page_turns),
                source_ids=[str(turn.get("turn_id") or "") for turn in page_turns if str(turn.get("turn_id") or "").strip()],
                metadata={"memory_bucket": "page", "page_id": page_id},
            )
            gist_entries.append(gist_entry)
            page_entries.append(page_entry)
            traces.append({"page_id": page_id, "gist": gist_text, "num_turns": len(page_turns)})

        self._gist_entries = gist_entries
        self._page_entries = page_entries
        self._memory_entries = gist_entries + page_entries
        self._org_state = OrganizationState(
            entries=list(self._memory_entries),
            levels={
                "gist": [entry.entry_id for entry in gist_entries],
                "page": [entry.entry_id for entry in page_entries],
            },
            priorities={entry.entry_id: (1.0 if entry.metadata.get("memory_bucket") == "gist" else 0.6) for entry in self._memory_entries},
            raw={"system": "readagent", "organization": "gist_plus_raw_pages"},
        )
        self._debug_state["build_memory"] = {"pages": traces}

    def retrieve(self, query: str, sample: UnifiedSample, top_k: int) -> RetrievalResult:
        gist_block = "\n".join(
            f"<Page {index}>\n{entry.content}"
            for index, entry in enumerate(self._gist_entries)
        )
        parallel_prompt = self._PARALLEL_LOOKUP_PROMPT.format(query=query, gist_block=gist_block)
        parallel_text = ""
        lookup_error = None
        if self._lookup_mode() == "lexical":
            selected_pages = self._lexical_page_indices(query=query, top_k=top_k)
        elif self._variant() == "sequential":
            selected_pages = []
        else:
            try:
                parallel_response = self.llm.complete_text(prompt=parallel_prompt, model=self.generation_model, temperature=0.0)
                parallel_text = parallel_response.text
                selected_pages = self._parse_page_indices(parallel_text, len(self._page_entries))
            except Exception as exc:
                lookup_error = f"{type(exc).__name__}: {exc}"
                if not self._allow_fallback():
                    raise RuntimeError(f"ReadAgent lookup failed in formal mode: {lookup_error}") from exc
                selected_pages = self._lexical_page_indices(query=query, top_k=top_k)

        sequential_traces: list[dict[str, Any]] = []
        while (
            self._variant() == "sequential"
            and lookup_error is None
            and self._lookup_mode() != "lexical"
            and len(selected_pages) < min(top_k, len(self._page_entries))
        ):
            prompt = self._SEQUENTIAL_LOOKUP_PROMPT.format(
                query=query,
                selected_pages=", ".join(str(item) for item in selected_pages) or "none",
                gist_block=gist_block,
            )
            try:
                response = self.llm.complete_text(prompt=prompt, model=self.generation_model, temperature=0.0)
                response_text = response.text
                next_pages = self._parse_page_indices(response_text, len(self._page_entries))
            except Exception as exc:
                lookup_error = f"{type(exc).__name__}: {exc}"
                if not self._allow_fallback():
                    raise RuntimeError(f"ReadAgent sequential lookup failed in formal mode: {lookup_error}") from exc
                next_pages = []
                response_text = ""
            sequential_traces.append({"prompt": prompt, "response": response_text, "parsed": next_pages, "error": lookup_error})
            new_pages = [item for item in next_pages if item not in selected_pages]
            if not new_pages:
                break
            selected_pages.extend(new_pages[:1])
        if not selected_pages and self._allow_fallback():
            selected_pages = self._lexical_page_indices(query=query, top_k=top_k)

        selection_budget = min(top_k, 2) if self._variant() == "parallel" else top_k
        selected_pages = selected_pages[:selection_budget]
        retrieved: list[RetrievedMemory] = []
        raw_rows: list[dict[str, Any]] = []
        for rank, page_index in enumerate(selected_pages, start=1):
            page_entry = self._page_entries[page_index]
            gist_entry = self._gist_entries[page_index]
            retrieved.append(
                RetrievedMemory(
                    entry_id=page_entry.entry_id,
                    content=page_entry.content,
                    score=float(len(selected_pages) - rank + 1),
                    rank=rank,
                    source_ids=list(page_entry.source_ids),
                    metadata={
                        "memory_bucket": "page",
                        "page_id": page_entry.metadata.get("page_id"),
                        "session_date": page_entry.created_at,
                        "created_at": page_entry.created_at,
                        "updated_at": page_entry.updated_at,
                        "gist_entry_id": gist_entry.entry_id,
                        "gist": gist_entry.content,
                    },
                )
            )
            raw_rows.append(
                {
                    "page_index": page_index,
                    "page_entry_id": page_entry.entry_id,
                    "gist_entry_id": gist_entry.entry_id,
                }
            )

        self._retrieval_result = RetrievalResult(
            query=query,
            retrieved_entries=retrieved,
            top_k=top_k,
            raw={
                "system": "readagent",
                "lookup_mode": self._lookup_mode(),
                "variant": self._variant(),
                "lookup_error": lookup_error,
                "parallel_lookup_prompt": parallel_prompt,
                "parallel_lookup_response": parallel_text,
                "selected_pages": selected_pages,
                "sequential_lookup": sequential_traces,
                "results": raw_rows,
            },
        )
        return self._retrieval_result

    def build_prompt(self, query: str, retrieved: RetrievalResult, sample: UnifiedSample) -> PromptRecord:
        history_index = self._history_index(sample)
        retrieved_by_page = {
            str(item.metadata.get("page_id")): item for item in retrieved.retrieved_entries
        }
        expanded_pages: list[str] = []
        injected_ids: list[str] = []
        injection_positions: dict[str, str] = {}
        for index, (gist_entry, page_entry) in enumerate(zip(self._gist_entries, self._page_entries)):
            page_id = str(page_entry.metadata.get("page_id"))
            retrieved_item = retrieved_by_page.get(page_id)
            if retrieved_item is not None:
                expanded_pages.append(
                    f"<Page {index}>\n{self._format_retrieved_memory(retrieved_item, history_index=history_index)}"
                )
                injected_ids.append(page_entry.entry_id)
                injection_positions[page_entry.entry_id] = "replacement_page"
            else:
                expanded_pages.append(f"<Page {index}>\n{gist_entry.content}")
                injected_ids.append(gist_entry.entry_id)
                injection_positions[gist_entry.entry_id] = "gist_page"

        # Official ReadAgent expands memory by replacing selected gists with
        # their raw pages, rather than appending duplicate raw pages.
        memory_context = "\n\n".join(expanded_pages)

        full_prompt = (
            "Read the question and text below and then answer the question.\n\n"
            f"Question:\n{query}\n\n"
            f"Text:\n{memory_context.strip()}\nEnd of Text.\n\n"
            "Answer the question based on the above passage and retrieved pages. "
            "Your answer should be short and concise."
        )
        self._prompt_record = PromptRecord(
            system_prompt="Prefer direct evidence from expanded pages when available.",
            user_prompt=f"Question: {query}",
            memory_context=memory_context.strip(),
            full_prompt=full_prompt,
            injected_entry_ids=injected_ids,
            token_count=self._estimate_token_count(full_prompt),
            injection_positions=injection_positions,
            raw={"system": "readagent", "prompt_layout": "selected_gists_replaced_by_raw_pages"},
        )
        return self._prompt_record

    def generate_answer(self, prompt: PromptRecord, sample: UnifiedSample) -> AnswerRecord:
        response = self.llm.complete_text(prompt=prompt.full_prompt, model=self.generation_model, temperature=0.0)
        self._answer_record = AnswerRecord(
            answer=response.text,
            raw_response=response.to_dict(),
            latency=response.latency,
            token_usage={"prompt_tokens_est": prompt.token_count, "completion_tokens_est": len(response.text.split())},
        )
        return self._answer_record

    def _ensure_llm_ready(self) -> None:
        if (self.llm_provider_name or "mock") == "mock":
            raise ValueError("ReadAgent 真实适配器不支持 mock provider，请使用 openai 并设置 OPENAI_API_KEY。")
        if (self.llm_provider_name or "").lower() != "openai":
            raise ValueError("当前 ReadAgent 真实适配器仅支持 openai provider。")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("ReadAgent 真实适配器需要 OPENAI_API_KEY。")

    def _paginate_history(self, history: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
        pages: list[tuple[str, list[dict[str, Any]]]] = []
        current: list[dict[str, Any]] = []
        current_tokens = 0
        max_tokens = 350
        for index, turn in enumerate(history, start=1):
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            turn_tokens = len(text.split())
            if current and current_tokens + turn_tokens > max_tokens:
                page_id = f"page_{len(pages) + 1:04d}"
                pages.append((page_id, list(current)))
                current = []
                current_tokens = 0
            current.append(turn)
            current_tokens += turn_tokens
        if current:
            page_id = f"page_{len(pages) + 1:04d}"
            pages.append((page_id, list(current)))
        if not pages:
            pages.append(("page_0001", []))
        return pages

    def _render_page(self, turns: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for turn in turns:
            speaker = str(turn.get("speaker") or "unknown")
            text = str(turn.get("text") or "").strip()
            if text:
                turn_id = str(turn.get("turn_id") or turn.get("source_id") or "").strip()
                session_id = str(turn.get("session_id") or "").strip()
                timestamp = str(turn.get("timestamp") or "").strip()
                header_parts = []
                if turn_id:
                    header_parts.append(f"turn_id={turn_id}")
                if session_id:
                    header_parts.append(f"session_id={session_id}")
                if timestamp:
                    header_parts.append(f"session_date={timestamp}")
                header = f"[{'; '.join(header_parts)}] " if header_parts else ""
                lines.append(f"{header}{speaker}: {text}")
        return "\n".join(lines)

    def _make_gist(self, page_text: str) -> str:
        if self._gist_mode() == "extractive":
            return self._extractive_gist(page_text)
        prompt = self._GIST_PROMPT.format(page_text=page_text)
        try:
            response = self._complete_text_for_stage(prompt=prompt, stage="build_memory", temperature=0.0)
            gist = response.text.strip()
            if gist:
                return gist
            if not self._allow_fallback():
                raise RuntimeError("ReadAgent gist generation returned an empty response in formal mode")
            self._debug_state.setdefault("gist_errors", []).append("empty_response")
            return self._extractive_gist(page_text)
        except Exception as exc:
            self._debug_state.setdefault("gist_errors", []).append(f"{type(exc).__name__}: {exc}")
            if not self._allow_fallback():
                raise RuntimeError(f"ReadAgent gist generation failed in formal mode: {type(exc).__name__}: {exc}") from exc
            return self._extractive_gist(page_text)

    def _gist_mode(self) -> str:
        return os.getenv("READAGENT_GIST_MODE", "llm").strip().lower()

    def _lookup_mode(self) -> str:
        return os.getenv("READAGENT_LOOKUP_MODE", "llm").strip().lower()

    def _variant(self) -> str:
        variant = os.getenv("READAGENT_VARIANT", "parallel").strip().lower()
        if variant not in {"parallel", "sequential"}:
            raise ValueError("READAGENT_VARIANT must be 'parallel' or 'sequential'")
        return variant

    def _allow_fallback(self) -> bool:
        return os.getenv("READAGENT_ALLOW_FALLBACK", "false").strip().lower() in {"1", "true", "yes", "on"}

    def _extractive_gist(self, page_text: str) -> str:
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        gist = " ".join(lines[:3]).strip()
        return gist[:420] if gist else page_text[:220]

    def _lexical_page_indices(self, *, query: str, top_k: int) -> list[int]:
        query_counts = self._token_counts(query)
        scored: list[tuple[float, int]] = []
        for index, (gist_entry, page_entry) in enumerate(zip(self._gist_entries, self._page_entries)):
            haystack_counts = self._token_counts(f"{gist_entry.content} {page_entry.content}")
            overlap = sum(min(query_counts[token], haystack_counts[token]) for token in query_counts)
            score = overlap / max(sum(query_counts.values()), 1)
            scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [index for score, index in scored if score > 0]
        if not selected:
            selected = [index for _score, index in scored]
        return selected[: min(top_k, len(self._page_entries))]

    def _token_counts(self, text: str) -> Counter[str]:
        return Counter(re.findall(r"[a-z0-9']+", text.lower()))

    def _parse_page_indices(self, text: str, max_page: int) -> list[int]:
        upper = text.strip().upper()
        if upper.startswith("NONE") or upper.startswith("STOP"):
            return []
        bracket_match = re.search(r"\[([^\]]+)\]", text)
        parse_target = bracket_match.group(1) if bracket_match else text
        indices = [int(match) for match in re.findall(r"\d+", parse_target)]
        output: list[int] = []
        for index in indices:
            if 0 <= index < max_page and index not in output:
                output.append(index)
        return output

    def _page_timestamp(self, turns: list[dict[str, Any]]) -> str | None:
        for turn in turns:
            timestamp = str(turn.get("timestamp") or "").strip()
            if timestamp:
                return timestamp
        return None

    def validate_setup(self) -> dict[str, Any]:
        provider = (self.llm_provider_name or "mock").lower()
        if provider == "mock":
            return {"system": self.system_name, "ready": False, "provider": provider, "message": "不支持 mock provider。"}
        if provider != "openai":
            return {"system": self.system_name, "ready": False, "provider": provider, "message": "当前仅支持 openai provider。"}
        if not os.getenv("OPENAI_API_KEY"):
            return {"system": self.system_name, "ready": False, "provider": provider, "message": "缺少 OPENAI_API_KEY。"}
        return {"system": self.system_name, "ready": True, "provider": provider, "message": ""}
