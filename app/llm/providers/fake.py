"""FakeLLM —— 整套测试的地基（路线文档 §12 硬要求 1）。

它让状态机测试确定、快速、免费。必须支持：
预设返回序列、模拟超时、返回非法 JSON、模拟拒答。

注意（§12 硬要求 6）：FakeLLM 可以声明任意 caps，
因此**不能**用它替代能力门控测试 —— 门控测试要验证
"caps 不足时 Gateway 确实拒绝"，而不是"FakeLLM 返回了正确结果"。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.llm.base import (
    ChatRequest,
    ChatResponse,
    LLMBadRequest,
    LLMRateLimited,
    LLMRefused,
    LLMServerError,
    LLMTimeout,
    TaskKind,
    Usage,
)
from app.llm.capability import ModelTier, ProviderCaps, ZhQuality


@dataclass(slots=True)
class Scripted:
    """一次预设响应。`error` 优先于 `text`。"""

    text: str | None = None
    error: Exception | None = None
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=lambda: Usage(120, 80))


def _caps(tier: ModelTier) -> ProviderCaps:
    return ProviderCaps(
        tier=tier,
        context_window=64_000,
        max_output_tokens=4_096,
        supports_json_mode=True,
        supports_function_calling=False,
        supports_temperature=True,
        supports_streaming=True,
        zh_quality=ZhQuality.HIGH,
        tokenizer_hint="cl100k_base",
        cost_per_1k_in=0.0,
        cost_per_1k_out=0.0,
        probed_at=datetime.now(UTC),
    )


class FakeProvider:
    """可编程的假 provider。

    用法：
        fake = FakeProvider()
        fake.script(TaskKind.ASK, Scripted(text='{"text": "你最怕失去什么？"}'))
        fake.script(TaskKind.ASK, Scripted(error=LLMTimeout("boom")))  # 第二次调用超时
    """

    def __init__(
        self,
        *,
        name: str = "fake",
        model: str = "fake-model",
        tier: ModelTier = ModelTier.STRONG,
        caps: ProviderCaps | None = None,
        default_text: str = "{}",
    ) -> None:
        self._name = name
        self._model = model
        self._caps = caps or _caps(tier)
        self._default_text = default_text
        self._scripts: dict[TaskKind, list[Scripted]] = defaultdict(list)
        self.calls: list[ChatRequest] = []
        self.closed = False

    # ---- 编程接口 ----

    @property
    def name(self) -> str:
        return self._name

    @property
    def caps(self) -> ProviderCaps:
        return self._caps

    def script(self, task: TaskKind, *responses: Scripted) -> FakeProvider:
        self._scripts[task].extend(responses)
        return self

    def script_json(self, task: TaskKind, payload: object, **kw: object) -> FakeProvider:
        return self.script(
            task, Scripted(text=json.dumps(payload, ensure_ascii=False), **kw)  # type: ignore[arg-type]
        )

    def script_timeout(self, task: TaskKind, times: int = 1) -> FakeProvider:
        return self.script(task, *[Scripted(error=LLMTimeout("fake timeout"))] * times)

    def script_rate_limited(self, task: TaskKind, times: int = 1) -> FakeProvider:
        return self.script(
            task, *[Scripted(error=LLMRateLimited("fake 429", retry_after_s=0.0))] * times
        )

    def script_server_error(self, task: TaskKind, times: int = 1) -> FakeProvider:
        return self.script(task, *[Scripted(error=LLMServerError("fake 500"))] * times)

    def script_bad_request(self, task: TaskKind) -> FakeProvider:
        return self.script(task, Scripted(error=LLMBadRequest("fake 400")))

    def script_refusal(self, task: TaskKind) -> FakeProvider:
        return self.script(task, Scripted(error=LLMRefused("fake content policy")))

    def script_invalid_json(self, task: TaskKind, times: int = 1) -> FakeProvider:
        """返回带 markdown 围栏与尾逗号的脏 JSON —— 触发 §6.2 的容错修复路径。"""
        dirty = '```json\n{"a": 1, "b": [1,2,],}\n```'
        return self.script(task, *[Scripted(text=dirty)] * times)

    def script_truncated(
        self, task: TaskKind, partial: str = '{"underlying_needs": [{"stat'
    ) -> FakeProvider:
        """模拟输出触顶截断 —— §6.5 要处理的核心失败模式。"""
        return self.script(task, Scripted(text=partial, finish_reason="length"))

    def call_count(self, task: TaskKind | None = None) -> int:
        if task is None:
            return len(self.calls)
        return sum(1 for c in self.calls if c.task is task)

    # ---- provider 协议 ----

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.calls.append(req)
        queue = self._scripts.get(req.task)
        item = queue.pop(0) if queue else Scripted(text=self._default_text)

        if item.error is not None:
            raise item.error

        return ChatResponse(
            text=item.text if item.text is not None else self._default_text,
            usage=item.usage,
            model=self._model,
            provider=self._name,
            finish_reason=item.finish_reason,
            latency_ms=1.0,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[str]:
        resp = await self.complete(req)
        # 按字符切块，模拟流式增量
        for i in range(0, len(resp.text), 8):
            yield resp.text[i : i + 8]

    async def aclose(self) -> None:
        self.closed = True
