"""Retry utilities with exponential backoff and jitter."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from core.exceptions import AutomationError

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    *,
    max_attempts: int = 3,
    backoff_base_seconds: float = 1.0,
    backoff_jitter_seconds: float = 0.5,
    retryable_exceptions: tuple[type[Exception], ...] = (AutomationError,),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Execute *func* with retries on retryable exceptions."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            delay = backoff_base_seconds * (2 ** (attempt - 1))
            delay += random.uniform(0, backoff_jitter_seconds)
            if on_retry:
                on_retry(attempt, exc)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
