"""状态机集成测试（路线文档 §12）。

用 FakeLLM 跑完整流程：三种模式 + 危机分支 + 追问 + Critic 否决 + 能力门控。
全部离线、确定、免费。
"""

from __future__ import annotations

import json

import pytest

from app.domain.dimension import TOTAL_QUESTIONS
from app.domain.mode import AskMode
from app.graph.build import build_graph
from app.graph.deps import Deps
from app.graph.state import GraphState
from app.llm.base import TaskKind
from app.llm.capability import ModelTier, ProviderCaps, ZhQuality
from app.llm.gateway import LLMGateway, Route
from app.llm.providers.fake import FakeProvider, Scripted
from app.safety.resources import load_crisis_resources
from app.schemas.enums import DegradeReason, Phase
from tests.conftest import (
    PROJECT_ROOT,
    answer_text,
    make_state,
    script_analysis,
    script_plans,
    script_quality,
    script_questions,
)

MAX_TURNS = 60  # 防御：真实流程远小于此


async def drive(graph, state: GraphState, answers: list[str] | None = None) -> GraphState:
    """驱动图直到终止或需要输入而无输入可给。

    模拟真实调用方：每轮把一个回答放进 inbox，跑到 turn_done，再进下一轮。
    """
    pending = list(answers or [])
    current = state

    for _ in range(MAX_TURNS):
        raw = await graph.ainvoke(current)
        current = GraphState.model_validate(raw)

        if current.phase in (Phase.DONE, Phase.CRISIS):
            return current

        # 需要回答且还有回答可给
        if current.open_questions and pending:
            current = current.model_copy(
                update={"inbox": pending.pop(0), "turn_done": False}
            )
            continue

        if current.phase is Phase.PAUSED:
            return current

        # 无输入可给且未终止 → 停止驱动
        if not current.inbox:
            return current

    raise AssertionError("驱动超过最大轮数，疑似死循环")


@pytest.mark.asyncio
async def test_batched_full_flow(deps: Deps, fake: FakeProvider) -> None:
    """BATCHED 模式跑完 10 问并产出报告。"""
    script_plans(fake, AskMode.BATCHED)
    script_questions(fake, TOTAL_QUESTIONS + 2)
    script_quality(fake, TOTAL_QUESTIONS + 2)
    script_analysis(fake)

    graph = build_graph(deps)
    answers = [answer_text(i) for i in range(1, TOTAL_QUESTIONS + 1)]
    final = await drive(graph, make_state(mode=AskMode.BATCHED), answers)

    assert final.phase is Phase.DONE, f"未走到 DONE，停在 {final.phase}"
    assert final.answered_count == TOTAL_QUESTIONS
    assert final.report is not None
    assert final.report.underlying_needs, "报告应含深层诉求"
    assert final.report.inner_conflicts, "报告应含内在冲突"
    assert final.report.suggestions, "报告应含建议"
    # 建议必须关联到真实存在的洞察编号（§6.5 代码校验）
    total_insights = len(final.report.underlying_needs) + len(final.report.inner_conflicts)
    for s in final.report.suggestions:
        assert all(0 <= i < total_insights for i in s.linked_insights)


@pytest.mark.asyncio
async def test_progressive_mode(deps: Deps, fake: FakeProvider) -> None:
    script_plans(fake, AskMode.PROGRESSIVE)
    script_questions(fake, TOTAL_QUESTIONS + 2)
    script_quality(fake, TOTAL_QUESTIONS + 2)
    script_analysis(fake)

    graph = build_graph(deps)
    answers = [answer_text(i) for i in range(1, TOTAL_QUESTIONS + 1)]
    final = await drive(graph, make_state(mode=AskMode.PROGRESSIVE), answers)

    assert final.phase is Phase.DONE
    assert final.answered_count == TOTAL_QUESTIONS


@pytest.mark.asyncio
async def test_one_shot_emits_all_questions_at_once(deps: Deps, fake: FakeProvider) -> None:
    script_plans(fake, AskMode.ONE_SHOT)
    script_questions(fake, TOTAL_QUESTIONS + 2)
    script_quality(fake, TOTAL_QUESTIONS + 2)
    script_analysis(fake)

    graph = build_graph(deps)
    state = make_state(mode=AskMode.ONE_SHOT)

    # 第一轮就应发出全部 10 问
    raw = await graph.ainvoke(state)
    after_first = GraphState.model_validate(raw)
    assert len(after_first.open_questions) == TOTAL_QUESTIONS, (
        f"ONE_SHOT 应一次发出 {TOTAL_QUESTIONS} 问，实际 {len(after_first.open_questions)}"
    )

    answers = [answer_text(i) for i in range(1, TOTAL_QUESTIONS + 1)]
    final = await drive(graph, after_first.model_copy(update={"turn_done": False}), answers)
    assert final.phase is Phase.DONE


@pytest.mark.asyncio
async def test_crisis_short_circuits_immediately(deps: Deps, fake: FakeProvider) -> None:
    """危机信号出现时立即终止提问，并输出配置中的资源（§3.4）。"""
    script_plans(fake, AskMode.BATCHED)
    script_questions(fake, 5)
    script_quality(fake, 5)

    graph = build_graph(deps)
    state = make_state(concern="我最近一直在想自杀，撑不下去了")
    final = await drive(graph, state)

    assert final.phase is Phase.CRISIS
    assert not final.open_questions, "危机分支必须清空待答问题"
    assert final.report is None, "危机分支不应生成报告"

    events = [e.event for e in final.emissions]
    assert "crisis_stopped" in events

    payload = next(e for e in final.emissions if e.event == "crisis_stopped").data
    # 资源来自配置，不由模型生成（§16.2）
    assert "12356" in payload["text"] or "热线" in payload["text"]
    # 规则层零 LLM 依赖：不应为此调用任何 LLM 判级之外的东西
    assert fake.call_count(TaskKind.ASK) == 0, "危机时不应生成任何问题"


@pytest.mark.asyncio
async def test_crisis_mid_session_stops_questioning(deps: Deps, fake: FakeProvider) -> None:
    """会话中途出现危机信号也必须立即短路。"""
    script_plans(fake, AskMode.BATCHED)
    script_questions(fake, 6)
    script_quality(fake, 6)

    graph = build_graph(deps)
    answers = ["第一个正常回答，说了一些具体情况。", "我已经买好了药，不想活了"]
    final = await drive(graph, make_state(), answers)

    assert final.phase is Phase.CRISIS
    # 回答仍被保存 —— 它是审计依据
    assert final.qa_pairs, "危机时的回答仍应保存以供审计"


@pytest.mark.asyncio
async def test_probe_does_not_consume_quota(deps: Deps, fake: FakeProvider) -> None:
    """追问不占用 10 问配额（§4.3）。"""
    script_plans(fake, AskMode.PROGRESSIVE)
    script_questions(fake, TOTAL_QUESTIONS + 6)
    # 第一个回答质量低 → 触发追问
    fake.script(
        TaskKind.QUALITY,
        Scripted(
            text=json.dumps(
                {
                    "score": 0.2,
                    "is_evasive": True,
                    "is_too_brief": False,
                    "missing_aspects": ["没说具体在意什么"],
                }
            )
        ),
    )
    script_quality(fake, TOTAL_QUESTIONS + 6)
    script_analysis(fake)

    graph = build_graph(deps)
    answers = [answer_text(i) for i in range(1, TOTAL_QUESTIONS + 3)]
    final = await drive(graph, make_state(mode=AskMode.PROGRESSIVE), answers)

    probes = [qa for qa in final.qa_pairs if qa.question.is_probe]
    assert probes, "低质量回答应触发追问"
    assert final.answered_count == TOTAL_QUESTIONS, "追问不应占用正题配额"
    assert len(final.qa_pairs) > TOTAL_QUESTIONS, "总问答数应多于 10（含追问）"


@pytest.mark.asyncio
async def test_consent_gate_blocks_without_consent(deps: Deps, fake: FakeProvider) -> None:
    script_plans(fake, AskMode.BATCHED)
    graph = build_graph(deps)
    final = await drive(graph, make_state(with_consent=False))

    assert final.phase is Phase.CONSENT
    assert fake.call_count(TaskKind.PLAN) == 0, "未同意前不应开始规划"
    assert any(e.event == "consent_required" for e in final.emissions)


@pytest.mark.asyncio
async def test_stale_policy_version_requires_reconsent(
    deps: Deps, fake: FakeProvider
) -> None:
    """条款版本变更需重新征求同意（§3.2）。"""
    graph = build_graph(deps)
    state = make_state()
    stale = state.consent.model_copy(update={"policy_version": "old-version"})
    final = await drive(graph, state.model_copy(update={"consent": stale}))

    assert final.phase is Phase.CONSENT
    assert fake.call_count(TaskKind.PLAN) == 0


@pytest.mark.asyncio
async def test_toy_provider_cannot_run_strong_tasks(deps: Deps, settings) -> None:
    """能力门控：TOY 档不得承担 STRONG 任务，显式失败而非静默劣化（§2.4.3）。"""
    toy = FakeProvider(
        name="toy_local",
        tier=ModelTier.TOY,
        caps=ProviderCaps(
            tier=ModelTier.TOY,
            context_window=1024,
            max_output_tokens=512,
            supports_json_mode=False,
            zh_quality=ZhQuality.LOW,
        ),
    )
    gw = LLMGateway(routes=[Route(provider=toy, label="local")])
    toy_deps = Deps.build(
        settings=settings,
        gateway=gw,
        prompts_root=PROJECT_ROOT / "prompts",
        crisis=load_crisis_resources(settings.crisis_resources_file),
    )
    graph = build_graph(toy_deps)
    final = await drive(graph, make_state())

    assert final.phase is Phase.PAUSED, "应显式暂停而非用弱模型顶替"
    assert toy.call_count(TaskKind.PLAN) == 0, "门控应在调用前拦截"
    err = next((e for e in final.emissions if e.event == "error"), None)
    assert err is not None and err.data["code"] == "INSUFFICIENT_CAPABILITY"


@pytest.mark.asyncio
async def test_fabricated_evidence_is_dropped(deps: Deps, fake: FakeProvider) -> None:
    """编造引用的洞察必须被丢弃（§6.4 第三层）。"""
    script_plans(fake, AskMode.BATCHED)
    script_questions(fake, TOTAL_QUESTIONS + 2)
    script_quality(fake, TOTAL_QUESTIONS + 2)

    # 第一段：一条真依据 + 一条编造依据
    fake.script(
        TaskKind.ANALYZE,
        Scripted(
            text=json.dumps(
                {
                    "surface_concern": "摇摆",
                    "underlying_needs": [
                        {
                            "statement": "真实洞察",
                            "evidence": [
                                {"question_index": 1, "quote": "我最怕失去现在稳定的生活"}
                            ],
                            "confidence": "medium",
                        },
                        {
                            "statement": "编造洞察",
                            "evidence": [
                                {"question_index": 1, "quote": "我其实一直想去环游世界"}
                            ],
                            "confidence": "high",
                        },
                    ],
                    "uncertainty_notes": [],
                },
                ensure_ascii=False,
            )
        ),
        Scripted(text=json.dumps({"inner_conflicts": [], "uncertainty_notes": []})),
        Scripted(
            text=json.dumps(
                {
                    "suggestions": [
                        {
                            "action": "排序担心",
                            "first_step": "今晚写下三件事",
                            "linked_insights": [0],
                        }
                    ],
                    "uncertainty_notes": [],
                },
                ensure_ascii=False,
            )
        ),
    )
    fake.script(TaskKind.CRITIQUE, Scripted(text=json.dumps({"approved": True, "issues": []})))

    graph = build_graph(deps)
    answers = [answer_text(i) for i in range(1, TOTAL_QUESTIONS + 1)]
    final = await drive(graph, make_state(), answers)

    assert final.report is not None
    statements = [i.statement for i in final.report.underlying_needs]
    assert "真实洞察" in statements
    assert "编造洞察" not in statements, "编造引用必须被程序化校验丢弃"
    assert final.report.degraded, "丢弃洞察是降级事件，必须可见（§9.5）"
    assert DegradeReason.EVIDENCE_DROPPED in final.report.degrade_reasons


@pytest.mark.asyncio
async def test_critic_rejection_triggers_redo(deps: Deps, fake: FakeProvider) -> None:
    """Critic 否决 → Analyst 重做；超上限仍出报告但标注不确定性（§4.3）。"""
    script_plans(fake, AskMode.BATCHED)
    script_questions(fake, TOTAL_QUESTIONS + 2)
    script_quality(fake, TOTAL_QUESTIONS + 2)

    # 三轮分析，前两轮被否决
    for _ in range(3):
        script_analysis(fake, approved=False)

    graph = build_graph(deps)
    answers = [answer_text(i) for i in range(1, TOTAL_QUESTIONS + 1)]
    final = await drive(graph, make_state(), answers)

    assert final.phase is Phase.DONE, "超过重做上限也必须出报告"
    assert final.report is not None
    assert final.report.degraded
    assert DegradeReason.CRITIC_UNRESOLVED in final.report.degrade_reasons
    assert final.report.uncertainty_notes, "应写出复核保留意见"
    assert fake.call_count(TaskKind.CRITIQUE) >= 2, "应发生重做"
