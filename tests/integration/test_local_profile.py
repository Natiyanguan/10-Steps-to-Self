"""`local` 档冒烟（路线文档 §12 硬要求 5 / §13.2.1）。

这是 minimind 在本项目最有价值的用途：
让"provider adapter / 流式 / 超时 / 重试链路真的通"这件事**每次提交都被验证**，
且完全离线免费。

**断言只针对流程与 schema，不针对内容质量**（§2.5）——
DemoProvider 产出的内容毫无意义，这是刻意的。
"""

from __future__ import annotations

import pytest

from app.domain.dimension import TOTAL_QUESTIONS, validate_coverage
from app.domain.mode import AskMode
from app.domain.technique import is_compatible
from app.graph.build import build_graph
from app.graph.deps import Deps
from app.graph.state import GraphState
from app.llm.base import InsufficientCapabilityError, TaskKind
from app.llm.capability import ModelTier
from app.llm.gateway import LLMGateway, Route
from app.llm.providers.demo import DemoProvider, demo_caps, demo_caps_strong
from app.safety.evidence import verify_quote
from app.schemas.enums import Phase, Sharpness
from tests.conftest import PROJECT_ROOT, answer_text
from tests.integration.test_flow import drive


@pytest.fixture
def demo_deps(settings) -> Deps:
    """用 DemoProvider 组装依赖 —— 等价于 `tss chat --fake`。"""
    gw = LLMGateway(routes=[Route(provider=DemoProvider(), label="local")])
    return Deps.build(
        settings=settings,
        gateway=gw,
        prompts_root=PROJECT_ROOT / "prompts",
    )


@pytest.mark.asyncio
async def test_local_profile_completes_full_flow(demo_deps: Deps) -> None:
    """离线全流程冒烟：走完 10 问并产出报告。"""
    graph = build_graph(demo_deps)
    answers = [answer_text(i) for i in range(1, TOTAL_QUESTIONS + 1)]
    final = await drive(graph, _state(), answers)

    assert final.phase is Phase.DONE, f"停在 {final.phase}"
    assert final.answered_count == TOTAL_QUESTIONS
    assert final.report is not None
    # 只断言结构存在，不断言内容有意义
    assert final.report.surface_concern
    assert final.report.suggestions


@pytest.mark.asyncio
async def test_demo_plan_satisfies_domain_constraints(demo_deps: Deps) -> None:
    """DemoProvider 只能从提示词解析约束 ——

    它产出的规划合法，反过来证明两件事：
      1. 提示词把"维度 → 配套技术"的配对关系讲清楚了
      2. 领域约束是**可满足的**（存在至少一个合法解）

    这条测试曾抓到真实缺陷：提示词最初把允许维度与可用技术列成两份
    独立列表，导致模型组合出 emotion+exception 这类非法配对，
    每次都要重规划、白烧 token。
    """
    graph = build_graph(demo_deps)
    answers = [answer_text(i) for i in range(1, TOTAL_QUESTIONS + 1)]
    final = await drive(graph, _state(), answers)

    assert final.plan is not None
    dims = [s.dimension for s in final.plan.slots]
    assert validate_coverage(dims) == [], "规划应满足全部维度约束"

    for slot in final.plan.slots:
        assert is_compatible(
            slot.technique,
            slot.dimension,
            sharpness=int(Sharpness.MODERATE),
            crisis_mode=False,
        ), f"第 {slot.index} 问：{slot.technique.value} 不适用于 {slot.dimension.value}"


@pytest.mark.asyncio
async def test_demo_evidence_survives_verification(demo_deps: Deps) -> None:
    """Demo 从提示词里提取真实原话作为依据 —— 走的是真实校验路径。

    若 Demo 取错了片段，这条会以"洞察被丢弃"的形式失败，
    而不是静默产出一份空报告。
    """
    graph = build_graph(demo_deps)
    answers = [answer_text(i) for i in range(1, TOTAL_QUESTIONS + 1)]
    final = await drive(graph, _state(), answers)

    assert final.report is not None
    assert final.report.underlying_needs, "依据校验后应仍有洞察存活"

    user_texts = final.user_texts()
    for ins in final.report.underlying_needs:
        for ev in ins.evidence:
            assert verify_quote(ev.quote, user_texts).ok, (
                f"报告里的依据未能通过校验：{ev.quote!r}"
            )


def test_demo_caps_default_is_fast_not_strong() -> None:
    """默认档位必须是 FAST —— 这是本地小模型的真实定位（§2.5）。

    `demo_caps_strong()` 是"假模型假装够强"，仅用于离线跑通全流程；
    真实本地模型不应这样声明。
    """
    assert demo_caps().tier is ModelTier.FAST
    assert demo_caps_strong().tier is ModelTier.STRONG


@pytest.mark.asyncio
async def test_fast_tier_demo_blocked_from_strong_tasks(settings) -> None:
    """FAST 档的 DemoProvider 不得承担 STRONG 任务（§2.4.3）。

    这条保证：即使有人把本地模型接进来，能力门控仍会挡住它 ——
    而不是让它产出格式完整但内容失效的心理分析。
    """
    gw = LLMGateway(
        routes=[Route(provider=DemoProvider(caps=demo_caps()), label="local")]
    )
    deps = Deps.build(settings=settings, gateway=gw, prompts_root=PROJECT_ROOT / "prompts")

    # 门控层面：STRONG 任务无可用路由
    for task in (TaskKind.PLAN, TaskKind.ASK, TaskKind.ANALYZE, TaskKind.CRITIQUE):
        assert gw.eligible_routes(task) == [], f"{task.value} 不该有可用路由"

    # FAST 任务仍然可用 —— 降级不等于不可用
    for task in (TaskKind.SAFETY, TaskKind.QUALITY, TaskKind.SUMMARIZE):
        assert gw.eligible_routes(task), f"{task.value} 应可用"

    # 流程层面：进入规划时显式失败并保留会话，而不是硬崩
    graph = build_graph(deps)
    final = await drive(graph, _state())
    assert final.phase is Phase.PAUSED
    codes = [e.data.get("code") for e in final.emissions if e.event == "error"]
    assert "INSUFFICIENT_CAPABILITY" in codes, f"应报能力不足，实际 {codes}"


@pytest.mark.asyncio
async def test_toy_tier_hard_blocked_even_if_misconfigured(settings) -> None:
    """TOY 档承担 STRONG 任务是无条件断言失败（§2.4.3 第 3 条）。

    即使有人在配置里把 minimind 标成能干活的档位，这条也必须挡住。
    """
    from app.llm.capability import ProviderCaps, ZhQuality

    toy = ProviderCaps(
        tier=ModelTier.TOY,
        context_window=1_024,
        max_output_tokens=512,
        supports_json_mode=True,   # 即便声称支持 JSON
        zh_quality=ZhQuality.HIGH,  # 即便声称中文很好
    )
    gw = LLMGateway(routes=[Route(provider=DemoProvider(caps=toy), label="local")])

    for task in (TaskKind.PLAN, TaskKind.ASK, TaskKind.ANALYZE, TaskKind.CRITIQUE):
        assert gw.eligible_routes(task) == []

    # SAFETY 也必须拒绝：安全不能建立在能力不足的模型上（§2.5）
    assert gw.eligible_routes(TaskKind.SAFETY) == []

    with pytest.raises(InsufficientCapabilityError):
        from app.llm.base import Message, Role

        await gw.complete(TaskKind.ANALYZE, [Message(Role.USER, "x")])


def _state(**kw) -> GraphState:
    from tests.conftest import make_state

    return make_state(mode=AskMode.BATCHED, **kw)
