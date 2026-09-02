"""重试与熔断（路线文档 §9.2 第 2/3 条）。

重试的关键不是"重试几次"，而是**只重试可重试的错误**：
4xx 参数错误与内容策略拒绝重试必然再失败，纯浪费配额与时间。
可重试性已在 `app/llm/base.py` 的异常类上定死，这里只读不判。
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from app.llm.base import CircuitOpenError, LLMError, LLMRateLimited


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 1.0
    factor: float = 2.0
    jitter: float = 0.3  # ±30%
    max_delay_s: float = 20.0

    def delay_for(self, attempt: int, *, retry_after_s: float | None = None) -> float:
        """第 attempt 次失败后的等待时长（attempt 从 1 开始）。

        供应商给了 `Retry-After` 就听它的 —— 比我们自己猜更准。
        """
        if retry_after_s is not None:
            return min(retry_after_s, self.max_delay_s)
        raw = self.base_delay_s * (self.factor ** (attempt - 1))
        raw = min(raw, self.max_delay_s)
        spread = raw * self.jitter
        return max(0.0, raw + random.uniform(-spread, spread))


async def call_with_retry[T](
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    on_retry: Callable[[int, LLMError, float], None] | None = None,
) -> T:
    """按策略重试。不可重试的错误立即抛出。"""
    last: LLMError | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await fn()
        except LLMError as exc:
            last = exc
            if not exc.retryable or attempt >= policy.max_attempts:
                raise
            retry_after = getattr(exc, "retry_after_s", None) if isinstance(
                exc, LLMRateLimited
            ) else None
            delay = policy.delay_for(attempt, retry_after_s=retry_after)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)

    raise last if last is not None else RuntimeError("unreachable")


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class CircuitBreaker:
    """滑动窗口熔断器（§9.2 第 3 条）。

    避免下游故障时请求堆积拖垮自己。半开状态放行一个试探请求。
    """

    failure_rate_threshold: float = 0.5
    window_size: int = 20
    min_samples: int = 5
    cooldown_s: float = 30.0
    name: str = "default"

    _outcomes: list[bool] = field(default_factory=list)
    _opened_at: float | None = None
    _half_open_inflight: bool = False

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def before_call(self) -> None:
        state = self.state
        if state is BreakerState.OPEN:
            remaining = self.cooldown_s - (time.monotonic() - (self._opened_at or 0.0))
            raise CircuitOpenError(
                f"熔断器 {self.name} 处于 OPEN，{remaining:.1f}s 后半开试探"
            )
        if state is BreakerState.HALF_OPEN:
            if self._half_open_inflight:
                raise CircuitOpenError(f"熔断器 {self.name} 半开，已有试探请求在途")
            self._half_open_inflight = True

    def record(self, *, success: bool) -> None:
        was_half_open = self.state is BreakerState.HALF_OPEN
        self._half_open_inflight = False

        if was_half_open:
            if success:
                self._reset()
            else:
                self._opened_at = time.monotonic()  # 试探失败，重新计时
            return

        self._outcomes.append(success)
        if len(self._outcomes) > self.window_size:
            self._outcomes.pop(0)

        if len(self._outcomes) < self.min_samples:
            return
        failures = sum(1 for ok in self._outcomes if not ok)
        if failures / len(self._outcomes) >= self.failure_rate_threshold:
            self._opened_at = time.monotonic()

    def _reset(self) -> None:
        self._outcomes.clear()
        self._opened_at = None
        self._half_open_inflight = False

    @property
    def failure_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        return sum(1 for ok in self._outcomes if not ok) / len(self._outcomes)
