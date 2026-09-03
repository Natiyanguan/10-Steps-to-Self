"""压缩接线集成测试（路线文档 §6.3 / HANDOFF.md 第 2 节）。

单元测试覆盖了 `plan_compression` 与 `Summarizer.summarize` 的纯逻辑，
这里验证**粘合层**：`listen._maybe_compress` 是否真的
—— 在水位触发时调用 summarizer、写入 `state.summary`、标记降级，
—— 在水位未到时什么都不做。

直接调用模块级函数 `_maybe_compress`，用一个小上下文窗口的 FakeProvider
让水位确定性触发，避免走完整状态机（那需要 script 十几种响应）。
"""

from __future__ import annotations

import json

import pytest

from app.domain.technique import Technique
from app.graph.deps import Deps
from app.graph.nodes import listen as listen_node
from app.graph.state import GraphState
from app.llm.base import TaskKind
from app.llm.capability import ModelTier, ProviderCaps, ZhQuality
from app.llm.gateway import LLMGateway, Route
from app.llm.providers.fake import FakeProvider, Scripted
from app.safety.resources import load_crisis_resources
from app.schemas.enums import DegradeReason, Sharpness
from app.schemas.session import Answer, QAPair, Question
from tests.conftest import PROJECT_ROOT, make_state

_LONG_ANSWER = "这是一段用于测试压缩水位触发的较长回答，包含足够多的字符来推动历史占用。"
# 逐字引用从这里复制，确保 Summarizer 的校验通过
_QUOTE = "测试压缩水位触发的较长回答"


def _pair(index: int) -> QAPair:
    return QAPair(
        question=Question(
            index=index,
            text=f"第{index}个问题，请说说你的想法？",
            technique=Technique.SCALING,
            sharpness=Sharpness.MODERATE,
        ),
        answer=Answer(question_index=index, text=_LONG_ANSWER * 10),
    )


def _small_deps(settings, *, context_window: int) -> tuple[Deps, FakeProvider]:
    """构造一个指定上下文窗口的 STRONG provider（能过 ASK/SUMMARIZE 门控）。"""
    caps = ProviderCaps(
        tier=ModelTier.STRONG,
        context_window=context_window,
        # 必须 ≥ 800：SUMMARIZE 任务的 min_output 是 800，低于它会触发压缩
        # 却被能力门控拒绝 summarize（正确降级，但不是本测试目标）。
        max_output_tokens=1024,
        supports_json_mode=True,
        supports_function_calling=False,
        supports_temperature=True,
        supports_streaming=True,
        zh_quality=ZhQuality.HIGH,
        tokenizer_hint="cl100k_base",
        cost_per_1k_in=0.0,
        cost_per_1k_out=0.0,
    )
    fake = FakeProvider(name="small_window", caps=caps)
    gw = LLMGateway(routes=[Route(provider=fake, label="primary")])
    deps = Deps.build(
        settings=settings,
        gateway=gw,
        prompts_root=PROJECT_ROOT / "prompts",
        crisis=load_crisis_resources(settings.crisis_resources_file),
    )
    return deps, fake


def _state_with_history(n_pairs: int) -> GraphState:
    state = make_state()
    pairs = [_pair(i) for i in range(1, n_pairs + 1)]
    return state.model_copy(update={"qa_pairs": pairs})


@pytest.mark.asyncio
async def test_no_compression_below_watermark(settings) -> None:
    """窗口足够大时，历史未达水位，不调用 summarizer、不写 summary。"""
    deps, fake = _small_deps(settings, context_window=64_000)
    state = _state_with_history(4)
    update: dict = {}

    await listen_node._maybe_compress(state, deps, update, state.qa_pairs)

    assert "summary" not in update
    assert fake.call_count(TaskKind.SUMMARIZE) == 0


@pytest.mark.asyncio
async def test_compression_triggers_and_writes_summary(settings) -> None:
    """窗口小、历史长 → 水位触发，写入 summary 并标记降级。

    窗口必须 ≥ 8000：SUMMARIZE 任务的 min_context 是 8000，
    小于它会触发压缩却被能力门控拒绝 summarize（那是对的降级，但不是本测试目标）。
    """
    deps, fake = _small_deps(settings, context_window=8_000)

    # 预置 SUMMARIZE 响应：一条真实逐字原话 + 一条 facts
    fake.script(
        TaskKind.SUMMARIZE,
        Scripted(
            text=json.dumps(
                {
                    "by_dimension": [
                        {"dimension": "value", "facts": ["重视稳定"], "emotions": []}
                    ],
                    "preserved_quotes": [_QUOTE],
                },
                ensure_ascii=False,
            )
        ),
    )

    state = _state_with_history(10)
    update: dict = {}

    await listen_node._maybe_compress(state, deps, update, state.qa_pairs)

    assert fake.call_count(TaskKind.SUMMARIZE) == 1
    assert "summary" in update, "压缩成功后应写入 summary"
    assert update["summary"].preserved_quotes == [_QUOTE]
    assert DegradeReason.HISTORY_COMPRESSED in update["degrade_reasons"]
    # 降级必须发可见事件（§9.5）
    assert any(
        e.event == "degraded"
        and e.data.get("reason") == DegradeReason.HISTORY_COMPRESSED.value
        for e in update["emissions"]
    )


@pytest.mark.asyncio
async def test_compression_failure_does_not_block(settings) -> None:
    """压缩失败时（LLM 超时）不写 summary、不抛异常 —— 保留原文继续。"""
    deps, fake = _small_deps(settings, context_window=8_000)
    fake.script_timeout(TaskKind.SUMMARIZE)

    state = _state_with_history(10)
    update: dict = {}

    # 不应抛异常
    await listen_node._maybe_compress(state, deps, update, state.qa_pairs)

    assert "summary" not in update
    # 超时是可重试错误，会走 gateway 的重试链，所以调用次数可能 > 1。
    # 测试只关心"确实尝试压缩了，且失败没写 summary、没抛异常"。
    assert fake.call_count(TaskKind.SUMMARIZE) >= 1


@pytest.mark.asyncio
async def test_no_ask_provider_skips_compression(settings) -> None:
    """没有能承担 ASK 的 provider 时，压缩直接跳过（不报错）。"""
    # 用 FAST 档 provider：能过 SUMMARIZE，但过不了 ASK（要求 STRONG）
    caps = ProviderCaps(
        tier=ModelTier.FAST,
        context_window=6_000,
        max_output_tokens=512,
        supports_json_mode=True,
        supports_temperature=True,
        supports_streaming=True,
        zh_quality=ZhQuality.HIGH,
    )
    fake = FakeProvider(name="fast_only", caps=caps)
    gw = LLMGateway(routes=[Route(provider=fake, label="primary")])
    deps = Deps.build(
        settings=settings,
        gateway=gw,
        prompts_root=PROJECT_ROOT / "prompts",
        crisis=load_crisis_resources(settings.crisis_resources_file),
    )

    state = _state_with_history(8)
    update: dict = {}
    await listen_node._maybe_compress(state, deps, update, state.qa_pairs)

    assert "summary" not in update
    assert fake.call_count(TaskKind.SUMMARIZE) == 0
