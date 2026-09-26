"""Frozen, fail-closed model-service and retrieval-cache contracts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROTOCOL_VERSION = "ab-model-services-v1"
EXTERNAL_AUTH_BY_BASE_URL = {}

def external_auth_env(base_url: str) -> str:
    return os.getenv("GENERATION_AUTH_ENV", "GENERATION_API_KEY")

def external_credential(base_url: str) -> tuple[str, str]:
    auth_env = external_auth_env(base_url)
    key = str(os.getenv(auth_env) or "")
    if not key:
        raise RuntimeError(f"{auth_env} is required for the configured service")
    return key, auth_env

SECRET_KEYS = {
    "OPENAI_API_KEY", "MEMORY_BUILD_OPENAI_API_KEY", "GENERATION_API_KEY",
    "MEMORY_BUILD_GENERATION_API_KEY", "API_KEY",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ServiceSpec:
    role: str
    provider: str
    base_url: str
    model: str
    required: bool
    api_path: str = ""
    auth_env: str = ""
    instruction_sha256: str = ""


@dataclass(frozen=True)
class ModelServiceProtocol:
    generation: ServiceSpec
    judge: ServiceSpec
    embedding: ServiceSpec
    reranker: ServiceSpec
    config_sha256: str

    @classmethod
    def load(cls, path: Path) -> "ModelServiceProtocol":
        raw = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(raw) or {}
        if payload.get("schema_version") != PROTOCOL_VERSION:
            raise ValueError("model service protocol version mismatch")
        leaked = []
        for key, value in (payload.get("env") or {}).items():
            if key in SECRET_KEYS and str(value or "").strip(): leaked.append(key)
        if leaked: raise ValueError(f"secret material is forbidden in config: {sorted(leaked)}")
        services = payload.get("services") or {}
        def spec(role: str) -> ServiceSpec:
            row = services.get(role) or {}
            instruction = str(row.get("instruction") or "")
            return ServiceSpec(
                role, str(row.get("provider") or ""), str(row.get("base_url") or "").rstrip("/"),
                str(row.get("model") or ""), bool(row.get("required")), str(row.get("api_path") or ""),
                str(row.get("auth_env") or ""), hashlib.sha256(instruction.encode()).hexdigest() if instruction else "",
            )
        protocol = cls(spec("generation"), spec("judge"), spec("embedding"), spec("reranker"),
                       hashlib.sha256(raw.encode()).hexdigest())
        for service in (protocol.generation, protocol.judge):
            if service.required and not service.auth_env:
                raise ValueError(f"{service.role} must declare auth_env")
        for service in (protocol.embedding, protocol.reranker):
            if service.required and (not service.base_url or not service.model):
                raise ValueError(f"required {service.role} service is incomplete")
        return protocol

    def external_credential(self, service: ServiceSpec) -> str:
        return external_credential(service.base_url)[0]

    def generation_credential(self) -> str:
        """Compatibility wrapper for the configured generation service."""
        return self.external_credential(self.generation)


REQUIRED_CACHE_BINDINGS = {
    "snapshot_sha256", "questions_sha256", "model", "instruction_sha256",
    "compiler_sha256", "service_protocol_sha256", "request_manifest_sha256",
    "response_manifest_sha256",
}


def validate_retrieval_cache_receipt(receipt: dict[str, Any], *, kind: str,
                                     expected: dict[str, str]) -> None:
    if receipt.get("schema_version") != f"ab-{kind}-cache-receipt-v1":
        raise ValueError(f"{kind} cache schema mismatch")
    if receipt.get("status") != "COMPLETE" or receipt.get("service_fallback_used") is not False:
        raise ValueError(f"{kind} cache is incomplete or used a fallback")
    missing = REQUIRED_CACHE_BINDINGS - set(receipt)
    if missing: raise ValueError(f"{kind} cache missing bindings: {sorted(missing)}")
    drift = {key: (receipt.get(key), value) for key, value in expected.items() if receipt.get(key) != value}
    if drift: raise ValueError(f"{kind} cache binding drift: {sorted(drift)}")
    if int(receipt.get("request_count") or 0) <= 0 or receipt.get("request_count") != receipt.get("response_count"):
        raise ValueError(f"{kind} cache request/response lineage is not closed")


def build_retrieval_cache_receipt(*, kind: str, bindings: dict[str, str],
                                  requests: list[dict[str, Any]],
                                  responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Close a one-request/one-terminal-response retrieval-cache lineage."""
    if kind not in {"dense", "reranker"}: raise ValueError("unsupported cache kind")
    request_ids = [str(row.get("cell_id") or "") for row in requests]
    response_ids = [str(row.get("cell_id") or "") for row in responses]
    if not request_ids or len(set(request_ids)) != len(request_ids) or set(request_ids) != set(response_ids):
        raise ValueError("cache cells are missing, duplicated, or unmatched")
    response_by_id = {str(row["cell_id"]): row for row in responses}
    for request in requests:
        cell_id = str(request["cell_id"]); response = response_by_id[cell_id]
        if response.get("status") != "terminal_ok": raise ValueError(f"cache cell is not terminal_ok: {cell_id}")
        if response.get("request_sha256") != canonical_sha256(request.get("request")):
            raise ValueError(f"cache request hash mismatch: {cell_id}")
        if not str(response.get("response_sha256") or ""): raise ValueError(f"cache response hash absent: {cell_id}")
    receipt = {
        "schema_version": f"ab-{kind}-cache-receipt-v1", "status": "COMPLETE",
        **bindings, "request_count": len(requests), "response_count": len(responses),
        "request_manifest_sha256": canonical_sha256(requests),
        "response_manifest_sha256": canonical_sha256(responses),
        "service_fallback_used": False,
    }
    validate_retrieval_cache_receipt(receipt, kind=kind, expected=bindings)
    return receipt
