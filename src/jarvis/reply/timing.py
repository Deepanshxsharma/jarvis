"""Per-reply stage timing.

One :class:`ReplyTiming` is active per reply (held in a context variable so
concurrent replies on different threads never mix). Engine call sites wrap
their work in ``stage("router")`` and the like; the summary line and the
benchmark read the finished record.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator, Optional

# Display order for the summary line; unknown stages follow in insertion order.
STAGE_ORDER = (
    "fast_path",
    "mcp_refresh",
    "router",
    "planner",
    "profile",
    "memory",
    "resolve",
    "tools",
    "digest",
    "llm",
)


@dataclass
class ReplyTiming:
    started: float = field(default_factory=perf_counter)
    finished: Optional[float] = None
    stages: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    prompt_tokens: int = 0
    output_tokens: int = 0
    first_text_at: Optional[float] = None

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + seconds
        self.calls[name] = self.calls.get(name, 0) + 1

    def record_llm_usage(self, response) -> None:
        if not isinstance(response, dict):
            return
        self.prompt_tokens += int(response.get("prompt_eval_count") or 0)
        self.output_tokens += int(response.get("eval_count") or 0)

    def mark_first_text(self) -> None:
        if self.first_text_at is None:
            self.first_text_at = perf_counter()

    def finish(self) -> None:
        if self.finished is None:
            self.finished = perf_counter()

    @property
    def total(self) -> float:
        end = self.finished if self.finished is not None else perf_counter()
        return end - self.started

    def format(self) -> str:
        names = [n for n in STAGE_ORDER if n in self.stages]
        names += [n for n in self.stages if n not in STAGE_ORDER]
        parts = []
        for name in names:
            part = f"{name}={_fmt(self.stages[name])}"
            if self.calls.get(name, 0) > 1:
                part += f"x{self.calls[name]}"
            parts.append(part)
        if self.prompt_tokens or self.output_tokens:
            parts.append(f"tokens={self.prompt_tokens}in/{self.output_tokens}out")
        parts.append(f"total={_fmt(self.total)}")
        return "REPLY " + " ".join(parts)


def _fmt(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


_current: contextvars.ContextVar[Optional[ReplyTiming]] = contextvars.ContextVar(
    "jarvis_reply_timing", default=None
)
_last: Optional[ReplyTiming] = None


def begin() -> ReplyTiming:
    global _last
    timing = ReplyTiming()
    _current.set(timing)
    _last = timing
    return timing


def current() -> Optional[ReplyTiming]:
    return _current.get()


def last_reply_timing() -> Optional[ReplyTiming]:
    """The most recently started reply's timing (for benchmarks and tests)."""
    return _last


@contextmanager
def stage(name: str) -> Iterator[None]:
    timing = _current.get()
    started = perf_counter()
    try:
        yield
    finally:
        if timing is not None:
            timing.add(name, perf_counter() - started)
