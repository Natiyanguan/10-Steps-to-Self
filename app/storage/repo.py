"""仓储层（路线文档 §5.2 / §9.1 乐观锁）。

核心职责：SessionState 的持久化与恢复、危机审计的 append-only 写入。

并发控制（§9.1）：
  `update_with_revision` 实现乐观锁 —— `UPDATE ... WHERE revision = :expected`，
  影响行数为 0 说明期间被别的请求改过，抛 `SessionConflictError`，
  由 API 层映射成 409。这是 Redis 分布式锁之外的兜底：锁抖动时版本号仍能防脏写。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.enums import RiskLevel
from app.schemas.session import Consent, SessionState
from app.storage.models import ConsentRecord, SafetyAuditRecord, SessionRecord


class SessionNotFoundError(LookupError):
    pass


class SessionConflictError(RuntimeError):
    """乐观锁冲突（§9.1）。调用方应返回 409，客户端重取后重试。"""


# --------------------------------------------------------------------------
# 序列化
# --------------------------------------------------------------------------
def serialize_state(state: SessionState) -> str:
    return json.dumps(state.model_dump(mode="json"), ensure_ascii=False)


def deserialize_state(raw: str) -> SessionState:
    return SessionState.model_validate(json.loads(raw))


def extract_session_state(graph_state: Any) -> SessionState:
    """从 GraphState 提取 SessionState（丢弃运行时字段 inbox/emissions/…）。

    持久化只存 SessionState：运行时字段（emissions、safety_action、inbox）
    是"本轮临时"的，恢复时不需要也不应该重放（见 `app/graph/state.py` 注释）。
    """
    data = {k: getattr(graph_state, k) for k in SessionState.model_fields}
    return SessionState(**data)


# --------------------------------------------------------------------------
# 会话
# --------------------------------------------------------------------------
class SessionRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def save(self, state: SessionState) -> None:
        """插入或覆盖。首次创建用这个；后续更新请用 `update_with_revision`。"""
        record = await self._s.get(SessionRecord, state.session_id)
        if record is None:
            self._s.add(
                SessionRecord(
                    session_id=state.session_id,
                    user_id=state.user_id,
                    status=_status_of(state),
                    phase=state.phase.value,
                    mode=state.mode.value,
                    sharpness=int(state.sharpness),
                    concern=state.concern,
                    state_json=serialize_state(state),
                    revision=state.revision,
                    paused_at=state.paused_at,
                )
            )
        else:
            record.state_json = serialize_state(state)
            record.status = _status_of(state)
            record.phase = state.phase.value
            record.mode = state.mode.value
            record.sharpness = int(state.sharpness)
            record.concern = state.concern
            record.revision = state.revision
            record.paused_at = state.paused_at
        await self._s.flush()

    async def load(self, session_id: str) -> SessionState | None:
        record = await self._s.get(SessionRecord, session_id)
        if record is None:
            return None
        return deserialize_state(record.state_json)

    async def load_or_raise(self, session_id: str) -> SessionState:
        state = await self.load(session_id)
        if state is None:
            raise SessionNotFoundError(f"会话不存在：{session_id}")
        return state

    async def update_with_revision(self, state: SessionState) -> SessionState:
        """乐观锁更新（§9.1）。

        用 `state.revision` 作为期望版本；成功则把版本 +1 并写回。
        影响行数为 0 → 抛 `SessionConflictError`。

        **关键**：序列化到 `state_json` 的必须是用 `revision + 1` 的 state，
        否则 `revision` 列和 `state_json` 里的 revision 字段不一致 ——
        下次 load 读回旧版本号，乐观锁形同虚设（这是测试抓到过的 bug）。
        """
        expected = state.revision
        bumped = state.model_copy(update={"revision": expected + 1})
        result = cast(
            CursorResult[Any],
            await self._s.execute(
                update(SessionRecord)
                .where(
                    SessionRecord.session_id == state.session_id,
                    SessionRecord.revision == expected,
                )
                .values(
                    state_json=serialize_state(bumped),
                    status=_status_of(bumped),
                    phase=bumped.phase.value,
                    mode=bumped.mode.value,
                    sharpness=int(bumped.sharpness),
                    concern=bumped.concern,
                    revision=expected + 1,
                    paused_at=bumped.paused_at,
                    ended_at=(
                        datetime.now(UTC)
                        if bumped.phase.value in ("done", "crisis")
                        else None
                    ),
                    updated_at=datetime.now(UTC),
                )
            ),
        )
        await self._s.flush()
        if result.rowcount == 0:
            raise SessionConflictError(
                f"会话 {state.session_id} 并发冲突（期望 revision={expected}）"
            )
        return bumped


def _status_of(state: SessionState) -> str:
    """把 Phase 映射到 sessions.status（§5.2）。"""
    phase = state.phase.value
    if phase in ("done", "crisis"):
        return "completed" if phase == "done" else "crisis_stopped"
    if phase == "paused":
        return "paused"
    return "active"


# --------------------------------------------------------------------------
# 危机审计（append-only，§5.2 / §3.4）
# --------------------------------------------------------------------------
class AuditRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def append(
        self,
        *,
        session_id: str | None,
        user_id_hash: str,
        risk_level: RiskLevel,
        source: str,
        matched_rules: list[str] | None,
        action_taken: str,
    ) -> None:
        self._s.add(
            SafetyAuditRecord(
                session_id=session_id,
                user_id_hash=user_id_hash,
                risk_level=risk_level.value,
                source=source,
                matched_rules=list(matched_rules or []),
                action_taken=action_taken,
            )
        )
        await self._s.flush()


# --------------------------------------------------------------------------
# 知情同意（§3.2）
# --------------------------------------------------------------------------
class ConsentRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def save(self, user_id: str, consent: Consent) -> None:
        record = await self._s.get(ConsentRecord, f"{user_id}:{consent.policy_version}")
        if record is None:
            self._s.add(
                ConsentRecord(
                    consent_id=f"{user_id}:{consent.policy_version}",
                    user_id=user_id,
                    policy_version=consent.policy_version,
                    scopes_json=json.dumps(sorted(s.value for s in consent.scopes)),
                    granted_at=consent.granted_at,
                    withdrawn_at=consent.withdrawn_at,
                )
            )
        else:
            record.scopes_json = json.dumps(sorted(s.value for s in consent.scopes))
            record.withdrawn_at = consent.withdrawn_at
        await self._s.flush()

    async def load(self, user_id: str, policy_version: str) -> Consent | None:
        record = await self._s.get(ConsentRecord, f"{user_id}:{policy_version}")
        if record is None:
            return None
        from app.schemas.enums import ConsentScope

        return Consent(
            policy_version=record.policy_version,
            granted_at=record.granted_at,
            scopes={ConsentScope(s) for s in json.loads(record.scopes_json)},
            withdrawn_at=record.withdrawn_at,
        )
