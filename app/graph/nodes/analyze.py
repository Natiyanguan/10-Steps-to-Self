"""ANALYZE / CRITIC / REPORT 节点（路线文档 §4.3 / §6.5 / §6.4）。

三个节点放在一个文件里，因为它们构成一个闭环：
`ANALYZE → CRITIC → ANALYZE`（最多重做 2 次）→ `REPORT`。

超过重做上限时**仍然出报告**，但带不确定性标注与 `degraded=true` ——
宁可承认依据不足，也不编造洞察（§4.3）。
"""

from __future__ import annotations

from typing import Any

from app.agents.critic import MAX_CRITIC_ROUNDS, feedback_for_retry
from app.graph.deps import Deps
from app.graph.state import GraphState
from app.llm.base import InsufficientCapabilityError, LLMError
from app.schemas.enums import DegradeReason, Phase


async def analyze(state: GraphState, deps: Deps) -> dict[str, Any]:
    try:
        result = await deps.analyst.analyze(
            state,
            critic_feedback=state.critic_feedback,
        )
    except InsufficientCapabilityError as exc:
        return {
            "phase": Phase.PAUSED,
            "turn_done": True,
            "emissions": [
                *state.emissions,
                state.emit(
                    "error",
                    code="INSUFFICIENT_CAPABILITY",
                    message="当前没有足以承担分析任务的模型。你的回答已保存，可稍后生成报告。",
                    detail=str(exc)[:500],
                ),
            ],
        }
    except LLMError as exc:
        return {
            "phase": Phase.PAUSED,
            "turn_done": True,
            "emissions": [
                *state.emissions,
                state.emit(
                    "error",
                    code="ANALYZE_FAILED",
                    message="生成分析暂时失败。你的回答已保存，可稍后重试。",
                    detail=str(exc)[:300],
                ),
            ],
        }

    degrade = list(state.degrade_reasons)
    for reason in result.report.degrade_reasons:
        if reason not in degrade:
            degrade.append(reason)

    return {
        "report": result.report,
        "phase": Phase.CRITIC,
        "analyze_rounds": state.analyze_rounds + 1,
        "degrade_reasons": degrade,
        "critic_feedback": [],
    }


async def critic(state: GraphState, deps: Deps) -> dict[str, Any]:
    if state.report is None:
        return {"phase": Phase.ANALYZE}

    outcome = await deps.critic.review(state.report, state)

    # 复核不可用：放行但标记降级（§9.5 —— 少了一道防线必须可见）
    if outcome.result is None:
        degrade = list(state.degrade_reasons)
        if DegradeReason.CRITIC_UNRESOLVED not in degrade:
            degrade.append(DegradeReason.CRITIC_UNRESOLVED)
        notes = [
            *state.report.uncertainty_notes,
            "本次报告未经过独立复核（复核环节暂时不可用）",
        ]
        return {
            "phase": Phase.REPORT,
            "report": state.report.model_copy(
                update={
                    "uncertainty_notes": notes,
                    "degraded": True,
                    "degrade_reasons": degrade,
                }
            ),
            "degrade_reasons": degrade,
        }

    if outcome.result.approved:
        return {"phase": Phase.REPORT}

    # 否决且还有重做机会 → 回到 ANALYZE，带上具体反馈
    if state.analyze_rounds < MAX_CRITIC_ROUNDS:
        return {
            "phase": Phase.ANALYZE,
            "critic_feedback": feedback_for_retry(outcome.result),
        }

    # 重做上限已到 → 仍然出报告，但标注不确定性（§4.3）
    degrade = list(state.degrade_reasons)
    if DegradeReason.CRITIC_UNRESOLVED not in degrade:
        degrade.append(DegradeReason.CRITIC_UNRESOLVED)
    notes = [
        *state.report.uncertainty_notes,
        *[f"复核仍有保留：{i.problem}" for i in outcome.result.issues[:3]],
    ]
    return {
        "phase": Phase.REPORT,
        "report": state.report.model_copy(
            update={
                "uncertainty_notes": notes,
                "degraded": True,
                "degrade_reasons": degrade,
            }
        ),
        "degrade_reasons": degrade,
    }


async def report(state: GraphState, deps: Deps) -> dict[str, Any]:
    """输出报告。免责声明在此出现（§3.1 决策 #10：报告显著位置）。"""
    if state.report is None:
        return {"phase": Phase.ANALYZE}

    emissions = [
        *state.emissions,
        state.emit(
            "report",
            payload=state.report.model_dump(mode="json"),
            disclaimer=deps.crisis.disclaimer,
            degraded=state.report.degraded,
        ),
        state.emit("done", phase=Phase.DONE.value),
    ]
    return {"phase": Phase.DONE, "emissions": emissions, "turn_done": True}


async def crisis(state: GraphState, deps: Deps) -> dict[str, Any]:
    """危机终止分支（§3.4）。

    这里**不生成任何文本** —— 危机资源全部来自配置（§16.2 硬约束）。
    模型编造错误的求助电话，后果不可接受。
    """
    return {
        "phase": Phase.CRISIS,
        "open_questions": [],
        "emissions": [
            *state.emissions,
            state.emit(
                "crisis_stopped",
                text=deps.crisis.render(deps.settings.default_region),
                resources_version=deps.crisis.version,
            ),
        ],
        "turn_done": True,
    }
