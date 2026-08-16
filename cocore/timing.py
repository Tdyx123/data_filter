"""Stable completion timing events for Cocore stages."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


TimingCallback = Callable[[str, float], None]


def emit_completed_timing(step: str, elapsed_seconds: float) -> None:
    """Write one completed Cocore timing event to stderr."""

    print(
        f"cocore_timing step={step} seconds={float(elapsed_seconds):.6f} status=completed",
        file=sys.stderr,
        flush=True,
    )


@contextmanager
def timed_step(step: str, callback: TimingCallback | None) -> Iterator[None]:
    """Report elapsed time only when the wrapped step completes successfully."""

    if callback is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        raise
    else:
        callback(step, time.perf_counter() - started)
