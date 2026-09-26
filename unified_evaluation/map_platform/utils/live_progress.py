"""Small, process-local live progress writer for benchmark runs.

The file deliberately contains only operational counters and no prompt/API
content.  It is written atomically so the external progress viewer never reads
half a JSON document while an adapter is updating a batch.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_state: dict[str, Any] = {}


def update_live_progress(**fields: Any) -> None:
    path_value = os.getenv("MEMORY_EVAL_PROGRESS_PATH", "").strip()
    if not path_value:
        return
    _state.update(fields)
    _state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _state.setdefault("pid", os.getpid())
    path = Path(path_value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(_state, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except OSError:
        # Progress must never change the benchmark result.  The runner log and
        # final JSONL remain authoritative if a filesystem is read-only/full.
        return
