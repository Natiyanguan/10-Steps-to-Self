"""API 层与 graph 的粘合（路线文档 §10 / §4.2 可恢复）。

核心是两个纯转换：
  `run_turn`：把 SessionState + inbox 送进 graph 跑一轮，取回更新后的 SessionState
             和本轮事件（emissions）。这是"会话可恢复"的 API 侧实现 ——
             每轮结束后把 SessionState 持久化，下次从库里读回来继续。
  `to_session_out`：把 SessionState + 事件转成给客户端的 JSON。

这里不 import 路由层，保持可独立单测。
"""

from __future__ import annotations

from typing import Any

from app.api.schemas import EventOut, SessionOut
from app.domain.dimension import TOTAL_QUESTIONS
from app.graph.state import Emission, GraphState
from app.schemas.session import SessionState
from app.storage.repo import extract_session_state


async def run_turn(
    graph: Any, session_state: SessionState, inbox: str | None
) -> tuple[SessionState, list[Emission]]:
    """跑一轮 graph。返回 (更新后的 SessionState, 本轮事件)。

    运行时字段（emissions / inbox / safety_action）不进 SessionState：
    它们是本轮临时的，持久化前会被 `extract_session_state` 剥掉。
    """
    gs = GraphState(**session_state.model_dump(), inbox=inbox, turn_done=False)
    result = await graph.ainvoke(gs)
    result_gs = GraphState.model_validate(result)
    new_session = extract_session_state(result_gs)
    return new_session, result_gs.emissions


def to_session_out(state: SessionState, emissions: list[Emission]) -> SessionOut:
    """把会话状态 + 事件转成客户端 JSON。"""
    open_questions = [
        {"index": q.index, "text": q.text, "is_probe": q.is_probe, "technique": q.technique.value}
        for q in state.open_questions
    ]

    consent_status = "none"
    if state.consent is not None:
        consent_status = "valid" if state.consent.is_active else "pending"

    return SessionOut(
        session_id=state.session_id,
        phase=state.phase.value,
        mode=state.mode.value,
        sharpness=int(state.sharpness),
        concern=state.concern,
        progress={"current": state.answered_count, "total": TOTAL_QUESTIONS},
        open_questions=open_questions,
        report=state.report.model_dump(mode="json") if state.report else None,
        degraded=bool(state.degrade_reasons),
        degrade_reasons=[r.value for r in state.degrade_reasons],
        events=[EventOut(event=e.event, data=e.data) for e in emissions],
        consent_status=consent_status,
        updated_at=state.paused_at,
    )
