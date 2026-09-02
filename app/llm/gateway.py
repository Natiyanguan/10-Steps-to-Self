"""LLMGateway —— 所有 LLM 调用的唯一入口（路线文档 §9.2）。

硬约束（§16.2）：**业务代码禁止直接调用 LLM SDK**，一律经此。

Gateway 负责的七件事，缺一件都会在生产暴露：
  1. 超时分级（不同任务用同一超时值是常见错误）
  2. 选择性重试（只重试可重试错误）
  3. 熔断
  4. 降级链 + **降级可见**（§9.5：静默降级最危险）
  5. 能力门控（§2.4.3：档位不足拒绝路由，而不是"先试试看"）
  6. 结构化输出的容错与重试（§6.2）
  7. token 与成本计量（§9.2 第 7 条）
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel

from app.llm import capability as cap
from app.llm.base import (
    CallRecord,
    ChatProvider,
    ChatRequest,
    ChatResponse,
    InsufficientCapabilityError,
    LLMError,
    LLMTimeout,
    LLMUnavailable,
    Message,
    Role,
    StructuredOutputError,
    TaskKind,
)
from app.llm.retry import CircuitBreaker, RetryPolicy, call_with_retry
from app.llm.structured import (
    MAX_PARSE_ATTEMPTS,
    build_retry_hint,
    parse_json_lenient,
    validate_into,
)
from app.schemas.enums import DegradeReason

T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------
# 超时分级（§9.2 第 1 条）
#
# 注意：这些是**任务内层截止时间**，与 LLM_TIMEOUT_SECONDS（HTTP 外层上限）
# 不是重复配置 —— 外层防连接悬挂，内层防单任务拖垮整体节奏。
# 若目标模型是推理型，需按 §17 探测得到的实测 P95 上调，不要照抄。
# --------------------------------------------------------------------------
DEFAULT_TIMEOUTS: dict[TaskKind, float] = {
    TaskKind.SAFETY: 3.0,
    TaskKind.QUALITY: 5.0,
    TaskKind.RERANK_JUDGE: 5.0,
    TaskKind.QUERY_REWRITE: 8.0,
    TaskKind.ASK: 15.0,
    TaskKind.PROBE: 15.0,
    TaskKind.PLAN: 20.0,
    TaskKind.SUMMARIZE: 20.0,
    TaskKind.CRITIQUE: 30.0,
    TaskKind.ANALYZE: 60.0,  # 报告分段后单段 60s（§6.5）
}

# 分析类任务需要更低温度（§6.4 生成前防线）
LOW_TEMPERATURE_TASKS: frozenset[TaskKind] = frozenset(
    {TaskKind.ANALYZE, TaskKind.CRITIQUE, TaskKind.SAFETY, TaskKind.QUALITY}
)


@dataclass(slots=True)
class Degradation:
    """一次降级事件。必须能回传给用户（§9.5）。"""

    reason: DegradeReason
    detail: str
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class CallOutcome:
    """调用结果 + 降级痕迹。业务层据此设置 `degraded` 标志。"""

    response: ChatResponse
    degradations: list[Degradation] = field(default_factory=list)
    attempts: int = 1

    @property
    def degraded(self) -> bool:
        return bool(self.degradations)


@dataclass(slots=True)
class Budget:
    """成本护栏（§9.2 第 7 条 / §3.7 防滥用）。"""

    max_tokens_per_session: int = 400_000
    spent_tokens: int = 0

    def charge(self, tokens: int) -> None:
        self.spent_tokens += tokens

    @property
    def exhausted(self) -> bool:
        return self.spent_tokens >= self.max_tokens_per_session

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens_per_session - self.spent_tokens)


class BudgetExceeded(LLMError):
    retryable = False


@dataclass(slots=True)
class Route:
    """任务到 provider 的一条候选路径。顺序即降级顺序。"""

    provider: ChatProvider
    label: str  # primary / fallback_model / fallback_provider / local


class LLMGateway:
    """任务 → provider 的路由器，带门控、重试、熔断、降级、计量。"""

    def __init__(
        self,
        *,
        routes: Sequence[Route],
        default_temperature: float = 0.3,
        retry_policy: RetryPolicy | None = None,
        timeouts: dict[TaskKind, float] | None = None,
        budget: Budget | None = None,
        on_record: Callable[[CallRecord], None] | None = None,
    ) -> None:
        if not routes:
            raise ValueError("至少需要一条 Route")
        self._routes = list(routes)
        self._default_temperature = default_temperature
        self._retry = retry_policy or RetryPolicy()
        self._timeouts = {**DEFAULT_TIMEOUTS, **(timeouts or {})}
        self._budget = budget or Budget()
        self._on_record = on_record
        self._breakers: dict[str, CircuitBreaker] = {
            r.provider.name: CircuitBreaker(name=r.provider.name) for r in self._routes
        }
        self.records: list[CallRecord] = []
        # 指标：llm_capability_rejected_total{task,provider}（§2.4.3 第 5 条）
        self.capability_rejections: list[tuple[TaskKind, str, str]] = []

    # ---- 路由 ----

    @property
    def budget(self) -> Budget:
        return self._budget

    def eligible_routes(self, task: TaskKind) -> list[Route]:
        """返回能承担该任务的路由（已过能力门控）。"""
        out: list[Route] = []
        for route in self._routes:
            result = cap.check(task, route.provider.caps, provider_name=route.provider.name)
            if result.ok:
                out.append(route)
            else:
                self.capability_rejections.append(
                    (task, route.provider.name, result.reason)
                )
        return out

    def _timeout_for(self, task: TaskKind) -> float:
        return self._timeouts.get(task, 30.0)

    def _temperature_for(self, task: TaskKind, override: float | None) -> float:
        if override is not None:
            return override
        if task in LOW_TEMPERATURE_TASKS:
            return min(self._default_temperature, 0.2)
        return self._default_temperature

    # ---- 单次调用 ----

    async def complete(
        self,
        task: TaskKind,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        prompt_version: str | None = None,
    ) -> CallOutcome:
        if self._budget.exhausted:
            raise BudgetExceeded(
                f"会话 token 预算已耗尽（{self._budget.spent_tokens}），拒绝继续调用"
            )

        routes = self.eligible_routes(task)
        if not routes:
            # §2.4.3 第 2 条：档位不足 → 拒绝路由，而不是降级到弱模型顶替
            reasons = [
                f"{name}: {why}"
                for t, name, why in self.capability_rejections[-len(self._routes) :]
                if t is task
            ]
            raise InsufficientCapabilityError(
                f"没有 provider 能承担任务 {task.value}。\n  "
                + "\n  ".join(reasons)
                + "\n  这不是 bug：宁可显式失败，也不让用户拿到劣质输出（§2.4.3）。"
            )

        req = ChatRequest(
            task=task,
            messages=tuple(messages),
            json_mode=json_mode,
            temperature=self._temperature_for(task, temperature),
            max_output_tokens=max_output_tokens,
            timeout_s=self._timeout_for(task),
            prompt_version=prompt_version,
        )

        degradations: list[Degradation] = []
        last_error: LLMError | None = None

        for idx, route in enumerate(routes):
            breaker = self._breakers.setdefault(
                route.provider.name, CircuitBreaker(name=route.provider.name)
            )

            if idx > 0:
                # 降级必须可见（§9.2 第 4 条 / §9.5）
                reason = (
                    DegradeReason.LLM_FALLBACK_MODEL
                    if route.label == "fallback_model"
                    else DegradeReason.LLM_FALLBACK_PROVIDER
                )
                degradations.append(
                    Degradation(
                        reason=reason,
                        detail=f"主路径不可用，已切换到 {route.provider.name}（{route.label}）",
                    )
                )

            attempts = 0

            # route / breaker 用默认参数绑定**当前迭代**的值，而不是闭包引用。
            # 闭包引用会在调用时才解析：目前恰好在同一轮迭代内 await，行为是对的，
            # 但一旦重试被推迟到循环之后（比如改成收集任务再批量执行），
            # 就会静默地对着最后一个 provider 重试 —— 而日志里记的还是原来那个。
            # 这类 bug 不会抛异常，只会让故障转移悄悄失效，所以在这里就绑死。
            async def _once(
                route: Route = route, breaker: CircuitBreaker = breaker
            ) -> ChatResponse:
                nonlocal attempts
                attempts += 1
                breaker.before_call()
                try:
                    resp = await asyncio.wait_for(
                        route.provider.complete(req), timeout=req.timeout_s
                    )
                except TimeoutError as exc:
                    breaker.record(success=False)
                    raise LLMTimeout(
                        f"任务 {task.value} 超过内层截止时间 {req.timeout_s}s"
                    ) from exc
                except LLMError:
                    breaker.record(success=False)
                    raise
                breaker.record(success=True)
                return resp

            try:
                resp = await call_with_retry(_once, self._retry)
            except LLMError as exc:
                last_error = exc
                continue

            self._budget.charge(resp.usage.total)
            record = CallRecord(
                task=task,
                provider=resp.provider,
                model=resp.model,
                usage=resp.usage,
                latency_ms=resp.latency_ms,
                attempt=attempts,
                degraded=bool(degradations),
                at=datetime.now(UTC),
            )
            self.records.append(record)
            if self._on_record is not None:
                self._on_record(record)

            if resp.truncated:
                degradations.append(
                    Degradation(
                        reason=DegradeReason.CONTEXT_TRUNCATED,
                        detail="模型输出触顶被截断，建议进一步分段（§6.5）",
                    )
                )

            return CallOutcome(response=resp, degradations=degradations, attempts=attempts)

        raise LLMUnavailable(
            f"任务 {task.value} 的降级链全部耗尽（{len(routes)} 条路径）。最后错误：{last_error}"
        ) from last_error

    # ---- 结构化调用（业务层主要用这个）----

    async def complete_structured(
        self,
        task: TaskKind,
        messages: Sequence[Message],
        model: type[T],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        prompt_version: str | None = None,
    ) -> tuple[T, CallOutcome]:
        """带 schema 校验与修复重试的调用（§6.2）。

        三次失败后**显式失败**，不猜测填充默认值 ——
        静默降级会产出"看起来正常但内容错误"的报告，比报错危险得多。
        """
        convo = list(messages)
        degradations: list[Degradation] = []
        last_outcome: CallOutcome | None = None
        last_error = ""

        for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
            outcome = await self.complete(
                task,
                convo,
                json_mode=True,
                # 重试时把温度压到 0（§6.2 第 3 层）
                temperature=0.0 if attempt > 1 else temperature,
                max_output_tokens=max_output_tokens,
                prompt_version=prompt_version,
            )
            last_outcome = outcome
            degradations.extend(outcome.degradations)
            text = outcome.response.text

            parsed = parse_json_lenient(text)
            if parsed.ok:
                obj, err = validate_into(parsed.data, model)
                if obj is not None:
                    merged = CallOutcome(
                        response=outcome.response,
                        degradations=degradations,
                        attempts=attempt,
                    )
                    return obj, merged
                last_error = err
            else:
                last_error = parsed.error

            if attempt < MAX_PARSE_ATTEMPTS:
                convo = [
                    *messages,
                    Message(Role.ASSISTANT, text[:2000]),
                    Message(Role.USER, build_retry_hint(last_error, model)),
                ]

        raw = last_outcome.response.text[:1000] if last_outcome else ""
        raise StructuredOutputError(
            f"任务 {task.value} 连续 {MAX_PARSE_ATTEMPTS} 次结构化输出失败：{last_error}",
            raw_text=raw,
        )

    # ---- 流式（仅用于问题文本，§9.4）----

    async def stream(
        self,
        task: TaskKind,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        routes = self.eligible_routes(task)
        if not routes:
            raise InsufficientCapabilityError(f"没有 provider 能承担流式任务 {task.value}")

        req = ChatRequest(
            task=task,
            messages=tuple(messages),
            json_mode=False,
            temperature=self._temperature_for(task, temperature),
            max_output_tokens=max_output_tokens,
            timeout_s=self._timeout_for(task),
        )
        provider = routes[0].provider
        async for chunk in provider.stream(req):
            yield chunk

    # ---- 观测 ----

    def total_usage(self) -> tuple[int, int]:
        return (
            sum(r.usage.prompt_tokens for r in self.records),
            sum(r.usage.completion_tokens for r in self.records),
        )

    def breaker_state(self, provider_name: str) -> str:
        breaker = self._breakers.get(provider_name)
        return breaker.state.value if breaker else "unknown"

    async def aclose(self) -> None:
        for route in self._routes:
            await route.provider.aclose()
