from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
from typing import Any


OPTIONAL_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "mem0": ("mem0ai", "qdrant-client"),
    "langmem": ("langmem", "langgraph", "langchain-openai"),
    "letta": ("letta-client",),
    "memgpt": ("letta-client",),
    "mnemis": ("graphiti-core", "neo4j"),
    "mirix": ("pyyaml",),
    "everos": ("requests",),
}


MEMORY_SYSTEM_REGISTRY: dict[str, tuple[str, str]] = {
    "no_memory": ("map_platform.memory_systems.no_memory", "NoMemoryAdapter"),
    "full_context": ("map_platform.memory_systems.full_context", "FullContextAdapter"),
    "simple_vector": ("map_platform.memory_systems.simple_vector", "SimpleVectorAdapter"),
    "current_memory": ("map_platform.memory_systems.current_memory", "CurrentMemoryAdapter"),
    "mem0": ("map_platform.memory_systems.mem0_adapter", "Mem0Adapter"),
    "memorybank": ("map_platform.memory_systems.memorybank_adapter", "MemoryBankAdapter"),
    "readagent": ("map_platform.memory_systems.readagent_adapter", "ReadAgentAdapter"),
    "memgpt": ("map_platform.memory_systems.memgpt_adapter", "MemGPTAdapter"),
    "letta": ("map_platform.memory_systems.memgpt_adapter", "LettaAdapter"),
    "langmem": ("map_platform.memory_systems.langmem_adapter", "LangMemAdapter"),
    "mnemis": ("map_platform.memory_systems.mnemis_adapter", "MnemisAdapter"),
    "mirix": ("map_platform.memory_systems.mirix_adapter", "MirixAdapter"),
    "everos": ("map_platform.memory_systems.everos_adapter", "EverOSAdapter"),
    "external": ("map_platform.memory_systems.generic_adapter", "GenericExternalAdapter"),
}


def load_system_manifest() -> dict[str, Any]:
    path = Path(__file__).with_name("system_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def list_systems(*, include_historical: bool = False, include_pending: bool = False) -> list[str]:
    """Return registered systems allowed by the repository lifecycle manifest."""
    metadata = load_system_manifest().get("systems", {})
    selected: list[str] = []
    for name in sorted(MEMORY_SYSTEM_REGISTRY):
        lifecycle = metadata.get(name, {}).get("lifecycle", "unclassified")
        if lifecycle == "historical" and not include_historical:
            continue
        if lifecycle == "pending" and not include_pending:
            continue
        selected.append(name)
    return selected


def build_memory_adapter(system_name: str, **kwargs: Any):
    normalized = system_name.strip().lower()
    target = MEMORY_SYSTEM_REGISTRY.get(normalized)
    if target is None:
        raise ValueError(f"Unsupported memory system: {system_name}")
    module_name, class_name = target
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        packages = OPTIONAL_DEPENDENCIES.get(normalized)
        if packages:
            install_cmd = f"python -m pip install {' '.join(packages)}"
            raise ModuleNotFoundError(
                f"Missing optional dependencies for memory system '{normalized}': "
                f"{', '.join(packages)}. Install them in the same Python environment "
                f"used to run the benchmark with: {install_cmd}"
            ) from exc
        raise
    adapter_cls = getattr(module, class_name)
    return adapter_cls(**kwargs)
