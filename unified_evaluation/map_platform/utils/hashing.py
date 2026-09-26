"""配置与 trace 载荷的哈希辅助函数。"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_json(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
