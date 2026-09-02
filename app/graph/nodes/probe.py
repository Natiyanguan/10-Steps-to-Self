"""PROBE 节点：生成追问（路线文档 §4.3）。

两条约束：
  1. 追问**不占用 10 问配额** —— 它是对同一问的深入，不是新的一问。
  2. 每问追问上限 `MAX_PROBES_PER_QUESTION`（默认 2），
     上限由 `listen` 在写入 `probe_counts` 时把关，这里只负责生成。

追问失败不是致命错误：直接回到 ASK 继续下一问，
用户不会察觉少了一次追问，但会察觉流程卡住。
"""

from __future__ import annotations

from typing import Any

from app.graph.deps import Deps
from app.graph.state import GraphState
from app.llm.base import LLMError
from app.schemas.enums import Phase


async def probe(state: GraphState, deps: Deps) -> dict[str, Any]:
    if state.plan is None or not state.qa_pairs:
        return {"phase": Phase.ASK, "probe_reason": ""}

    last = state.qa_pairs[-1]
    try:
        slot = state.plan.slot(last.question.index)
    except KeyError:
        return {"phase": Phase.ASK, "probe_reason": ""}

    action = state.safety_action
    crisis_mode = action.crisis_mode if action else False

    try:
        result = await deps.interviewer.ask(
            slot=slot,
            concern=state.concern,
            sharpness=state.sharpness,
            recent_exchanges=[
                {"question": qa.question.text, "answer": qa.answer.text}
                for qa in state.qa_pairs[-2:]
            ],
            crisis_mode=crisis_mode,
            is_probe=True,
            probe_reason=state.probe_reason,
        )
    except LLMError:
        # 追问失败 → 跳过，继续正常流程
        return {"phase": Phase.ASK, "probe_reason": ""}

    return {
        "phase": Phase.LISTEN,
        "open_questions": [*state.open_questions, result.question],
        "probe_reason": "",
        "emissions": [
            *state.emissions,
            state.emit(
                "question",
                index=result.question.index,
                text=result.question.text,
                technique=result.question.technique.value,
                is_probe=True,
            ),
        ],
        "turn_done": True,
    }
