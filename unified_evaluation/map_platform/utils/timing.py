"""用于实验阶段计时的轻量辅助工具。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Timer:
    """可重复使用的简单计时器对象。"""

    started_at: float = field(default_factory=time.perf_counter)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0
