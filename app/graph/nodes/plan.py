"""PLAN 节点：规划维度与技术序列（路线文档 §4.3 / §1.4 / §1.5）。

BATCHED 模式下本节点会被多次进入：每批结束时对**未发出**的槽位重规划（§4.3）。
已问过的槽位绝不重规划 —— 那会让已发出的问题与规划不一致。
"""

from __future__ import annotations

from typing import Any

from app.agents.planner import PlanValidationError
from app.domain.dimension import TOTAL_QUESTIONS
from app.domain.mode import batch_bounds
from app.graph.deps import Deps
from app.graph.state import GraphState
from app.llm.base import InsufficientCapabilityError, LLMError
from app.schemas.enums import Phase, SafetyActionKind
from app.schemas.session import QuestionPlan


def _current_batch(state: GraphState) -> int:
    """当前需要规划的批次：已规划槽位数落在哪一批。"""
    planned = len(state.plan.slots) if state.plan else 0
    for i, (_lo, hi) in enumerate(batch_bounds(state.mode)):
        if planned < hi:
            return i
    return len(batch_bounds(state.mode)) - 1


async def plan(state: GraphState, deps: Deps) -> dict[str, Any]:
    existing = list(state.plan.slots) if state.plan else []
    if len(existing) >= TOTAL_QUESTIONS:
        return {"phase": Phase.ASK}

    action = state.safety_action
    crisis_mode = action.crisis_mode if action else False
    block_deep = action.block_deep_dimensions if action else state.is_minor

    try:
        result = await deps.planner.plan(
            concern=state.concern,
            mode=state.mode,
            sharpness=state.sharpness,
            batch_index=_current_batch(state),
            existing=existing,
            crisis_mode=crisis_mode,
            block_deep=block_deep,
        )
    except InsufficientCapabilityError as exc:
        # 能力门控拒绝 —— 显式失败而非降级顶替（§2.4.3）
        return {
            "phase": Phase.PAUSED,
            "turn_done": True,
            "emissions": [
                *state.emissions,
                state.emit(
                    "error",
                    code="INSUFFICIENT_CAPABILITY",
                    message="当前没有足以承担规划任务的模型，会话已保留，可稍后继续。",
                    detail=str(exc)[:500],
                ),
            ],
        }
    except (PlanValidationError, LLMError) as exc:
        return {
            "phase": Phase.PAUSED,
            "turn_done": True,
            "emissions": [
                *state.emissions,
                state.emit(
                    "error",
                    code="PLAN_FAILED",
                    message="规划暂时失败，会话已保留，可稍后继续。",
                    detail=str(exc)[:500],
                ),
            ],
        }

    # QuestionPlan 支持分批中间态：整体覆盖度在满 10 问时才校验，
    # 但逐问维度合法性与相邻不重复始终校验（见 QuestionPlan._check_plan）。
    new_plan = QuestionPlan(
        slots=existing + result.slots, mode=state.mode, rationale=result.rationale
    )

    update: dict[str, Any] = {"plan": new_plan, "phase": Phase.ASK}

    if action and action.kind is SafetyActionKind.SOFTEN:
        update["emissions"] = [
            *state.emissions,
            state.emit("crisis_resources", text=deps.crisis.render(deps.settings.default_region)),
        ]
    return update
