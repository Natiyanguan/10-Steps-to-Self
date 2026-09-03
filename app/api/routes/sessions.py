"""会话路由（路线文档 §10.1）。

三条核心路径：
  POST /sessions            创建会话 → 跑第一轮（intake）→ 持久化 → 返回首个问题
  GET  /sessions/{id}       恢复会话状态（无副作用，不跑 graph）
  POST /sessions/{id}/answers 提交回答 → 恢复 → 跑一轮 → 乐观锁持久化 → 返回下一问/报告

并发（§9.1）：提交回答用 `update_with_revision` 乐观锁，冲突返回 409。
真正的分布式锁 + Redis 幂等留到 P5（HANDOFF §9.1）。
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Response

from app.api import errors
from app.api.app import Services
from app.api.deps import get_services
from app.api.schemas import CreateSessionRequest, SessionOut, SubmitAnswerRequest
from app.api.state import run_turn, to_session_out
from app.graph.state import GraphState
from app.llm.base import LLMError
from app.observability.metrics import METRICS
from app.schemas.enums import Phase
from app.schemas.session import Consent, SessionState
from app.storage.repo import AuditRepo, SessionConflictError, SessionNotFoundError, SessionRepo

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


def _record_turn_metrics(state: SessionState, duration_s: float) -> None:
    """记录单轮耗时，会话结束时记录结局与放弃点（§11.3）。"""
    METRICS.record_turn(phase=state.phase.value, duration_s=duration_s)
    if state.phase in (Phase.DONE, Phase.CRISIS):
        outcome = "completed" if state.phase is Phase.DONE else "crisis_stopped"
        METRICS.record_session_end(answered=state.answered_count, outcome=outcome)


def _user_id(x_user_id: str | None) -> str:
    return (x_user_id or "anonymous").strip()[:64]


async def _audit_if_needed(
    session: SessionRepo, audit: AuditRepo, before: SessionState, after: SessionState
) -> None:
    """本轮若新增了 medium 及以上的风险，写 append-only 审计（§3.4 / §5.2）。"""
    old_len = len(before.risk_history)
    for risk in after.risk_history[old_len:]:
        from app.schemas.enums import RiskLevel, risk_at_least

        if risk_at_least(risk.level, RiskLevel.MEDIUM):
            METRICS.record_risk(risk.level.value)
            await audit.append(
                session_id=after.session_id,
                user_id_hash=_user_id_hash(after.user_id),
                risk_level=risk.level,
                source=risk.source.value,
                matched_rules=risk.matched_rules,
                action_taken=("stop" if risk_at_least(risk.level, RiskLevel.HIGH) else "soften"),
            )


def _user_id_hash(user_id: str) -> str:
    from app.safety.injection import digest

    return digest(user_id)


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(
    body: CreateSessionRequest,
    response: Response,
    services: Annotated[Services, Depends(get_services)],
    x_user_id: str | None = Header(default=None),
) -> SessionOut:
    user_id = _user_id(x_user_id)
    session_id = uuid.uuid4().hex

    consent = None
    if body.consent is not None:
        from datetime import UTC, datetime

        consent = Consent(
            policy_version=body.consent.policy_version,
            granted_at=datetime.now(UTC),
            scopes=set(body.consent.scopes),
        )

    state = SessionState(
        session_id=session_id,
        user_id=user_id,
        mode=body.mode,
        sharpness=body.sharpness,
        is_minor=body.is_minor,
        concern="",
        consent=consent,
    )

    # 第一轮：inbox = concern，走 intake
    gs = GraphState(**state.model_dump(), inbox=body.concern, turn_done=False)
    started = time.perf_counter()
    try:
        result = await services.graph.ainvoke(gs)
    except LLMError as exc:
        raise errors.from_llm_error(exc) from exc
    result_gs = GraphState.model_validate(result)
    new_state = _extract(result_gs)
    _record_turn_metrics(new_state, time.perf_counter() - started)

    async with services.session_factory() as db:
        repo = SessionRepo(db)
        await repo.save(new_state)
        # 首次就可能触发危机，审计立即落库
        await _audit_if_needed(repo, AuditRepo(db), state, new_state)
        await db.commit()

    out = to_session_out(new_state, result_gs.emissions)
    if new_state.phase is Phase.CRISIS:
        # CRISIS_STOPPED 是正常路径（§10.3），用 200 而非错误码
        response.status_code = 200
    return out


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: str,
    services: Annotated[Services, Depends(get_services)],
) -> SessionOut:
    async with services.session_factory() as db:
        repo = SessionRepo(db)
        try:
            state = await repo.load_or_raise(session_id)
        except SessionNotFoundError as exc:
            raise errors.session_not_found(session_id) from exc
    return to_session_out(state, [])


@router.post("/{session_id}/answers", response_model=SessionOut)
async def submit_answer(
    session_id: str,
    body: SubmitAnswerRequest,
    services: Annotated[Services, Depends(get_services)],
) -> SessionOut:
    async with services.session_factory() as db:
        repo = SessionRepo(db)
        try:
            before = await repo.load_or_raise(session_id)
        except SessionNotFoundError as exc:
            raise errors.session_not_found(session_id) from exc

        if not before.open_questions:
            # 没有待答问题：要么已完成，要么还没到提问阶段
            if before.phase in (Phase.DONE, Phase.CRISIS):
                raise errors.invalid_request("会话已结束，无法继续提交回答")
            raise errors.invalid_request("当前没有待回答的问题")

        # 跑一轮：inbox = 回答
        started = time.perf_counter()
        try:
            new_state, emissions = await run_turn(services.graph, before, body.text)
        except LLMError as exc:
            raise errors.from_llm_error(exc) from exc
        _record_turn_metrics(new_state, time.perf_counter() - started)

        # 乐观锁持久化（§9.1）：并发提交会让 revision 冲突 → 409
        try:
            saved = await repo.update_with_revision(new_state)
        except SessionConflictError as exc:
            raise errors.session_conflict(session_id) from exc

        await _audit_if_needed(repo, AuditRepo(db), before, saved)
        await db.commit()

    return to_session_out(saved, emissions)


def _extract(gs: GraphState) -> SessionState:
    from app.storage.repo import extract_session_state

    return extract_session_state(gs)
