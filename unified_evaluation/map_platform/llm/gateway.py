from __future__ import annotations

import os
import re
import time
import base64
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from map_platform.utils.hashing import sha256_json
try:
    from scheme_a_method.cell_transport import active_scope, durable_response_call, reserve_logical_cell
except ModuleNotFoundError:
    def active_scope() -> None:
        return None

    @contextmanager
    def reserve_logical_cell():
        yield None

    def durable_response_call(**_kwargs: Any):
        raise RuntimeError("Scheme-A durable transport is unavailable outside an active scope")


def normalize_provider(provider: str | None) -> tuple[str, str | None]:
    raw = str(provider or os.getenv("LLM_PROVIDER", "mock")).strip()
    if raw.lower().startswith(("http://", "https://")):
        return "openai", raw
    return raw.lower() or "mock", openai_base_url()


def openai_base_url() -> str | None:
    return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_API_BASE_URL") or None


def embedding_base_url() -> str | None:
    return os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_EMBEDDING_BASE_URL") or openai_base_url()


def openai_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("API_T") or None,
        "timeout": float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        "max_retries": int(os.getenv("LLM_MAX_RETRIES", "2")),
    }
    base_url = openai_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def embedding_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "api_key": os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
        "timeout": float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "120"))),
        "max_retries": int(os.getenv("EMBEDDING_MAX_RETRIES", os.getenv("LLM_MAX_RETRIES", "2"))),
    }
    base_url = embedding_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def openai_sdk_config(*, model: str, temperature: float | None = None, include_base_url: bool = True) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": model,
        "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("API_T") or None,
    }
    if temperature is not None:
        config["temperature"] = temperature
    base_url = openai_base_url()
    if include_base_url and base_url:
        config["base_url"] = base_url
        config["openai_base_url"] = base_url
    return config


def embedding_sdk_config(*, model: str, include_base_url: bool = True) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": model,
        "api_key": os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
    }
    base_url = embedding_base_url()
    if include_base_url and base_url:
        config["base_url"] = base_url
        config["openai_base_url"] = base_url
    return config


def embed_texts(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    from openai import OpenAI

    embedding_model = model or os.getenv("EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL_NAME") or "BAAI/bge-m3"
    client = OpenAI(**embedding_client_kwargs())
    safe_texts = truncate_embedding_inputs(texts)
    try:
        response = client.embeddings.create(model=embedding_model, input=safe_texts, encoding_format="float")
    except Exception as exc:
        # Some OpenAI-compatible embedding servers implement the endpoint but
        # reject the optional ``encoding_format`` field.  Retry the standard
        # request before treating the error as a context-length failure.
        message = str(exc).lower()
        if any(marker in message for marker in ("encoding_format", "unexpected keyword", "extra fields", "unknown field")):
            response = client.embeddings.create(model=embedding_model, input=safe_texts)
        else:
            retried_texts = truncate_embedding_inputs_for_context_error(safe_texts, exc)
            if retried_texts is safe_texts:
                raise
            response = client.embeddings.create(model=embedding_model, input=retried_texts)
    vectors = [list(item.embedding) for item in response.data]
    expected_dims = int(os.getenv("EMBEDDING_DIMS", "0") or 0)
    if expected_dims and vectors and len(vectors[0]) != expected_dims:
        raise ValueError(
            f"embedding dimension mismatch: service returned {len(vectors[0])}, "
            f"but EMBEDDING_DIMS={expected_dims}; update the setting or use a matching model"
        )
    return vectors


def truncate_embedding_inputs(texts: list[str]) -> list[str]:
    max_chars = int(os.getenv("EMBEDDING_MAX_INPUT_CHARS", "24000"))
    if max_chars <= 0:
        return texts
    return [truncate_text_for_embedding(text, max_chars=max_chars) for text in texts]


def truncate_text_for_embedding(text: str, *, max_chars: int | None = None) -> str:
    limit = max_chars if max_chars is not None else int(os.getenv("EMBEDDING_MAX_INPUT_CHARS", "24000"))
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n...\n"
    if limit <= len(marker):
        return text[:limit]
    content_limit = limit - len(marker)
    head = int(content_limit * 0.65)
    tail = max(0, content_limit - head)
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def truncate_embedding_inputs_for_context_error(texts: list[str], exc: Exception) -> list[str]:
    message = str(exc)
    max_match = re.search(r"maximum context length is (\d+) tokens", message)
    requested_match = re.search(r"requested (\d+) tokens", message)
    if not max_match or not requested_match:
        return texts
    max_tokens = int(max_match.group(1))
    requested_tokens = int(requested_match.group(1))
    if max_tokens <= 0 or requested_tokens <= max_tokens:
        return texts
    ratio = max(0.1, min(0.9, (max_tokens / requested_tokens) * 0.85))
    retried = [truncate_text_for_embedding(text, max_chars=max(1, int(len(text) * ratio))) for text in texts]
    return retried if retried != texts else texts


def strip_reasoning_tags(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    if cleaned.lower().startswith("<think>") and "</think>" not in cleaned.lower():
        return ""
    return cleaned


def looks_like_extra_body_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "extra_body" in text
        or "chat_template_kwargs" in text
        or "bad_response_status_code" in text
        or "openai_error" in text
        or "403" in text
    )


def is_retryable_openai_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if any(marker in name for marker in ("authentication", "permissiondenied", "badrequest")):
        return False
    if any(marker in text for marker in ("401", "403", "invalid api key")):
        return False
    return any(
        marker in name or marker in text
        for marker in (
            "apiconnection",
            "apitimeout",
            "ratelimit",
            "internalserver",
            "timeout",
            "connection reset",
            "temporarily unavailable",
            "retryable",
            "429",
            "500",
            "502",
            "503",
            "504",
            "520",
            "522",
            "524",
        )
    )


def openai_retry_delay(exc: Exception, attempt: int) -> float:
    text = str(exc)
    match = re.search(r"['\"]retry_after['\"]\s*:\s*(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    base = float(os.getenv("LLM_RETRY_BASE_SECONDS", "5"))
    cap = float(os.getenv("LLM_RETRY_MAX_SECONDS", "60"))
    return min(cap, base * (2 ** max(0, attempt - 1)))


def is_gpt5_like(model: str) -> bool:
    return model.strip().lower().startswith(("gpt-5", "o1", "o3", "o4"))


def is_qwen_like(model: str) -> bool:
    return "qwen" in model.strip().lower()


def thinking_disabled() -> bool:
    return os.getenv("LLM_DISABLE_THINKING", "1").strip().lower() not in {"0", "false", "no", "off"}


def append_no_think_directive(prompt: str, model: str) -> str:
    if not thinking_disabled() or not is_qwen_like(model):
        return prompt
    if os.getenv("LLM_APPEND_NO_THINK", "1").strip().lower() in {"0", "false", "no", "off"}:
        return prompt
    if "/no_think" in prompt.lower():
        return prompt
    return f"{prompt.rstrip()}\n/no_think"


def token_param_name(model: str) -> str:
    override = os.getenv("LLM_TOKEN_PARAM", "").strip()
    if override:
        return override
    return "max_completion_tokens" if is_gpt5_like(model) else "max_tokens"


def chat_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return str(message.get("content") or "")
                return str(first.get("text") or first.get("content") or "")
        return str(response.get("text") or response.get("content") or response)
    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            return str(getattr(message, "content", "") or "")
        return str(getattr(first, "text", "") or getattr(first, "content", "") or "")
    return str(response or "")


class LLMMetricsStore:
    """记录最小必要的模型调用元数据。"""

    def __init__(self, metrics_dir: Path) -> None:
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def record(self, payload: dict[str, Any]) -> Path:
        path = self.metrics_dir / f"{int(time.time() * 1000)}.json"
        path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


@dataclass
class LLMResponse:
    text: str
    provider: str
    requested_model: str
    actual_model: str
    temperature: float
    latency: float
    prompt_hash: str
    raw: dict[str, Any]
    usage: dict[str, Any] | None = None
    attempts: int = 1
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "requested_model": self.requested_model,
            "actual_model": self.actual_model,
            "temperature": self.temperature,
            "latency": self.latency,
            "prompt_hash": self.prompt_hash,
            "raw": dict(self.raw),
            "usage": dict(self.usage or {}),
            "attempts": self.attempts,
            "request_id": self.request_id,
        }


class LLMGateway:
    def __init__(
        self,
        *,
        provider: str | None = None,
        metrics_dir: str | Path | None = None,
    ) -> None:
        self.provider, provider_base_url = normalize_provider(provider)
        if provider_base_url:
            os.environ.setdefault("OPENAI_BASE_URL", provider_base_url)
        self.metrics_store = LLMMetricsStore(Path(metrics_dir or "storage/llm_metrics"))
        self._openai_clients: dict[tuple[Any, ...], Any] = {}
        self._last_call_meta: dict[str, Any] = {}

    def complete_text(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.0,
    ) -> LLMResponse:
        prompt_hash = sha256_json({"prompt": prompt, "model": model, "temperature": temperature, "provider": self.provider})
        started = time.perf_counter()
        provider = self._build_provider(model)
        self._last_call_meta = {}
        error = ""
        success = False
        text = ""
        try:
            text = provider(prompt, temperature)
            if not str(text or "").strip():
                raise RuntimeError(
                    "Empty LLM response after reasoning cleanup. "
                    "The model may have exhausted max_tokens inside a thinking block."
                )
            success = True
            return self._record_response(
                text=text,
                provider=self.provider,
                requested_model=model,
                actual_model=f"{self.provider}/{model}",
                temperature=temperature,
                latency=time.perf_counter() - started,
                prompt_hash=prompt_hash,
                raw=dict(self._last_call_meta.get("raw") or {}),
                usage=dict(self._last_call_meta.get("usage") or {}),
                attempts=int(self._last_call_meta.get("attempts") or 1),
                request_id=self._last_call_meta.get("request_id"),
                success=True,
                error="",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if not success:
                self.metrics_store.record(
                    {
                        "provider": self.provider,
                        "model": model,
                        "requested_model": model,
                        "actual_model": self.provider,
                        "temperature": temperature,
                        "prompt_hash": prompt_hash,
                        "duration_ms": (time.perf_counter() - started) * 1000.0,
                        "usage": dict(self._last_call_meta.get("usage") or {}),
                        "attempts": int(self._last_call_meta.get("attempts") or 1),
                        "success": False,
                        "error": error,
                    }
                )

    def _record_response(
        self,
        *,
        text: str,
        provider: str,
        requested_model: str,
        actual_model: str,
        temperature: float,
        latency: float,
        prompt_hash: str,
        raw: dict[str, Any],
        usage: dict[str, Any] | None = None,
        attempts: int = 1,
        request_id: str | None = None,
        success: bool,
        error: str,
    ) -> LLMResponse:
        payload = {
            "provider": provider,
            "model": actual_model,
            "requested_model": requested_model,
            "actual_model": actual_model,
            "temperature": temperature,
            "prompt_hash": prompt_hash,
            "duration_ms": latency * 1000.0,
            "success": success,
            "error": error,
            "usage": dict(usage or {}),
            "attempts": attempts,
            "request_id": request_id,
        }
        self.metrics_store.record(payload)
        return LLMResponse(
            text=text,
            provider=provider,
            requested_model=requested_model,
            actual_model=actual_model,
            temperature=temperature,
            latency=latency,
            prompt_hash=prompt_hash,
            raw=raw,
            usage=dict(usage or {}),
            attempts=attempts,
            request_id=request_id,
        )

    def _build_provider(self, model: str):
        if self.provider == "mock":
            return self._mock_complete
        if self.provider == "openai":
            return lambda prompt, temperature: self._openai_complete(prompt, model, temperature)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _mock_complete(self, prompt: str, temperature: float) -> str:
        if "strict evaluation judge" in prompt.lower():
            return "FAIL"
        lower = prompt.lower()
        if "memory block:" in lower:
            memory_block = prompt.split("Memory Block:", 1)[1].split("Question:", 1)[0].strip()
            lines = [line.strip("- ").strip() for line in memory_block.splitlines() if line.strip()]
            if lines:
                return " ".join(lines[:2])
        return "Mock response."

    def _openai_complete(self, prompt: str, model: str, temperature: float) -> str:
        with reserve_logical_cell():
            return self._openai_complete_inner(prompt, model, temperature)

    def _openai_complete_inner(self, prompt: str, model: str, temperature: float) -> str:
        from openai import OpenAI

        client_kwargs = openai_client_kwargs()
        if active_scope() is not None:
            # SDK retries are otherwise invisible to the exact-attempt ledger.
            client_kwargs["max_retries"] = 0
        client_key = (
            client_kwargs.get("api_key"),
            client_kwargs.get("base_url"),
            client_kwargs.get("timeout"),
            client_kwargs.get("max_retries"),
        )
        client = self._openai_clients.get(client_key)
        if client is None:
            client = OpenAI(**client_kwargs)
            self._openai_clients[client_key] = client
        extra_body = None
        use_extra_body = (
            os.getenv("LLM_USE_EXTRA_BODY", "").strip().lower() in {"1", "true", "yes", "on"}
            or not is_gpt5_like(model)
        )
        if use_extra_body and thinking_disabled():
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        user_prompt = append_no_think_directive(prompt, model)
        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user_prompt}],
            token_param_name(model): int(os.getenv("LLM_MAX_TOKENS", "128")),
        }
        if not is_gpt5_like(model) or os.getenv("LLM_FORCE_TEMPERATURE", "").strip().lower() in {"1", "true", "yes", "on"}:
            params["temperature"] = temperature
        if extra_body:
            params["extra_body"] = extra_body
        def create_response(call_params: dict[str, Any]) -> Any:
            if active_scope() is None:
                return client.chat.completions.create(**call_params)
            raw_response, durable, _ = durable_response_call(
                request={"route": "chat/completions", "params": call_params},
                invoke=lambda: client.chat.completions.with_raw_response.create(**call_params),
                serialize_response=lambda value: {
                    "raw_body_b64": base64.b64encode(value.http_response.content).decode(),
                    "raw_body_sha256": __import__("hashlib").sha256(value.http_response.content).hexdigest(),
                    "http_status": int(value.http_response.status_code),
                    "response_headers": dict(value.http_response.headers),
                },
            )
            if raw_response is not None:
                return raw_response.parse()  # strictly after durable_response_call fsync
            raw = base64.b64decode((durable.get("response") or {})["raw_body_b64"])
            if __import__("hashlib").sha256(raw).hexdigest() != (durable.get("response") or {}).get("raw_body_sha256"):
                raise RuntimeError("durable provider raw body hash mismatch")
            return __import__("json").loads(raw)
        attempts = max(1, int(os.getenv("LLM_OUTER_MAX_ATTEMPTS", "3")))
        completed_attempts = 0
        for attempt in range(1, attempts + 1):
            completed_attempts = attempt
            try:
                response = create_response(params)
                break
            except Exception as exc:
                if extra_body and looks_like_extra_body_error(exc) and active_scope() is None:
                    retry_params = dict(params)
                    retry_params.pop("extra_body", None)
                    try:
                        response = create_response(retry_params)
                        break
                    except Exception as retry_exc:
                        exc = retry_exc
                if attempt >= attempts or not is_retryable_openai_error(exc):
                    raise
                time.sleep(openai_retry_delay(exc, attempt))
        usage = self._usage_dict(
            response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        )
        request_id = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
        self._last_call_meta = {
            "usage": usage,
            "attempts": completed_attempts,
            "request_id": request_id,
            "raw": {"usage": usage, "request_id": request_id} if usage or request_id else {},
        }
        raw_text = chat_response_text(response)
        cleaned = strip_reasoning_tags(raw_text)
        if not cleaned and raw_text.strip().lower().startswith("<think>") and thinking_disabled() and is_qwen_like(model):
            retry_params = dict(params)
            retry_params.pop("extra_body", None)
            retry_params[token_param_name(model)] = max(
                int(params[token_param_name(model)]),
                int(os.getenv("LLM_NO_THINK_RETRY_MAX_TOKENS", "256")),
            )
            response = create_response(retry_params)
            usage = self._usage_dict(
                response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
            )
            request_id = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
            self._last_call_meta.update({
                "usage": usage,
                "request_id": request_id,
                "raw": {"usage": usage, "request_id": request_id} if usage or request_id else {},
            })
            cleaned = strip_reasoning_tags(chat_response_text(response))
        empty_attempts = max(1, int(os.getenv("LLM_EMPTY_RESPONSE_MAX_ATTEMPTS", "3")))
        empty_token_cap = max(
            int(params[token_param_name(model)]),
            int(os.getenv("LLM_EMPTY_RESPONSE_MAX_TOKENS", "8192")),
        )
        for empty_attempt in range(2, empty_attempts + 1):
            if cleaned:
                break
            retry_params = dict(params)
            retry_params[token_param_name(model)] = min(
                empty_token_cap,
                int(params[token_param_name(model)]) * (2 ** (empty_attempt - 1)),
            )
            response = create_response(retry_params)
            usage = self._usage_dict(
                response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
            )
            request_id = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
            self._last_call_meta.update({
                "usage": usage,
                "request_id": request_id,
                "raw": {"usage": usage, "request_id": request_id} if usage or request_id else {},
            })
            cleaned = strip_reasoning_tags(chat_response_text(response))
        return cleaned

    @staticmethod
    def _usage_dict(usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        def value(name: str) -> Any:
            if isinstance(usage, dict):
                return usage.get(name)
            return getattr(usage, name, None)
        prompt = value("prompt_tokens")
        completion = value("completion_tokens")
        total = value("total_tokens")
        prompt = prompt if prompt is not None else value("input_tokens")
        completion = completion if completion is not None else value("output_tokens")
        total = total if total is not None else (
            int(prompt) + int(completion) if prompt is not None and completion is not None else None
        )
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "source": "api",
        }
