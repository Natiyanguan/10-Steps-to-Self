"""ASK 节点：渲染并发出问题（路线文档 §4.3 / §1.5）。

按模式决定一次发几问（§1.5）：
  ONE_SHOT 一次发全部、BATCHED 发完当前批、PROGRESSIVE 一次一问。

发出后停下等待用户回答 —— 这是本轮的自然终点。
"""

from __future__ import annotations

from typing import Any

from app.agents.interviewer import QuestionResult
from app.domain.dimension import TOTAL_QUESTIONS
from app.domain.mode import AskMode, batch_index_for, batch_slice
from app.graph.deps import Deps
from app.graph.state import GraphState
from app.llm.base import InsufficientCapabilityError, LLMError
from app.schemas.enums import Phase
from app.schemas.session import Question

RECENT_EXCHANGES = 3  # 最近 3 轮保留完整原文（§6.3）


def _recent(state: GraphState) -> list[dict[str, str]]:
    return [
        {"question": qa.question.text, "answer": qa.answer.text}
        for qa in state.qa_pairs[-RECENT_EXCHANGES:]
    ]


def _questions_this_turn(state: GraphState) -> int:
    """本轮要发几问。"""
    if state.mode is AskMode.PROGRESSIVE:
        return 1
    nxt = state.next_question_index
    if state.mode is AskMode.ONE_SHOT:
        return TOTAL_QUESTIONS - nxt + 1
    _, hi = batch_slice(state.mode, batch_index_for(state.mode, nxt))
    return hi - nxt + 1


async def ask(state: GraphState, deps: Deps) -> dict[str, Any]:
    if state.plan is None:
        return {"phase": Phase.PLAN}
    if state.is_complete:
        return {"phase": Phase.ANALYZE}

    action = state.safety_action
    crisis_mode = action.crisis_mode if action else False

    count = _questions_this_turn(state)
    start = state.next_question_index
    emissions = list(state.emissions)
    degrade = list(state.degrade_reasons)
    asked: list[Question] = []

    for offset in range(count):
        index = start + offset
        if index > TOTAL_QUESTIONS:
            break
        try:
            slot = state.plan.slot(index)
        except KeyError:
            break

        try:
            result: QuestionResult = await deps.interviewer.ask(
                slot=slot,
                concern=state.concern,
                sharpness=state.sharpness,
                recent_exchanges=_recent(state),
                crisis_mode=crisis_mode,
            )
        except InsufficientCapabilityError as exc:
            return {
                "phase": Phase.PAUSED,
                "turn_done": True,
                "emissions": [
                    *emissions,
                    state.emit(
                        "error",
                        code="INSUFFICIENT_CAPABILITY",
                        message="当前没有足以承担提问任务的模型，会话已保留。",
                        detail=str(exc)[:500],
                    ),
                ],
            }
        except LLMError as exc:
            if offset == 0:
                return {
                    "phase": Phase.PAUSED,
                    "turn_done": True,
                    "emissions": [
                        *emissions,
                        state.emit(
                            "error",
                            code="ASK_FAILED",
                            message="生成问题暂时失败，会话已保留，可稍后继续。",
                            detail=str(exc)[:300],
                        ),
                    ],
                }
            break  # 已发出部分问题，本轮就到这里

        for reason in result.degrade_reasons:
            if reason not in degrade:
                degrade.append(reason)
        if result.used_fallback:
            emissions.append(
                state.emit(
                    "degraded",
                    reason="question_guard_fallback",
                    impact="这一问使用了通用问法，深度不及正常生成",
                )
            )

        emissions.append(
            state.emit(
                "question",
                index=result.question.index,
                text=result.question.text,
                technique=result.question.technique.value,
                progress={"current": result.question.index, "total": TOTAL_QUESTIONS},
            )
        )
        asked.append(result.question)

    if not asked:
        return {"phase": Phase.PLAN}

    return {
        "phase": Phase.LISTEN,
        "open_questions": [*state.open_questions, *asked],
        "emissions": emissions,
        "degrade_reasons": degrade,
        "turn_done": True,
    }
