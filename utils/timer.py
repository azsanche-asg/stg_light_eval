"""Simple timing helpers for quick benchmarking runs."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

Sink = Callable[[str], None]


def _default_sink(message: str) -> None:
    print(message)


@dataclass
class Timer(contextlib.AbstractContextManager[float]):
    """Context manager that measures elapsed time using perf_counter."""

    label: str | None = None
    sink: Sink = field(default=_default_sink)
    enabled: bool = True
    _start: float = field(init=False, default=0.0)
    _elapsed: float = field(init=False, default=0.0)

    def __enter__(self) -> "Timer":
        if self.enabled:
            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        if self.enabled:
            self._elapsed = time.perf_counter() - self._start
            if self.label:
                self.sink(f"{self.label}: {self._elapsed:.4f}s")
        return False

    @property
    def elapsed(self) -> float:
        return self._elapsed


@contextlib.contextmanager
def time_block(label: str | None = None, sink: Sink = _default_sink, enabled: bool = True) -> Iterator[Timer]:
    """Convenience wrapper mirroring Timer but using `with time_block(...)` style."""
    timer = Timer(label=label, sink=sink, enabled=enabled)
    with timer:
        yield timer

