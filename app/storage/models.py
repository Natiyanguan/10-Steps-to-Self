"""ORM 模型（路线文档 §5.2 物理数据模型）。

设计取舍（P1）：
  - `sessions.state_json` 存**完整 SessionState 的 JSON 快照**，而不是把每条消息
    拆成单独一行。理由：SessionState 已经承载了全部领域状态（含 qa_pairs），
    逐字段建模会重复且容易漂移。§5.2 的 messages 表留到 P5 需要精细查询时再拆。
  - `revision` 乐观锁字段保留（§5.1 / §9.1），冲突检测在仓储层做。
  - `safety_audit` 是 append-only：业务代码只 insert，撤销 UPDATE/DELETE 权限。
  - 用 SQLite（开发）跑通；SQLAlchemy 方言屏蔽差异，P5 换 Postgres 只改连接串。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    """会话表。`state_json` 是权威，其余字段是冗余查询列。"""

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    phase: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(String(32), default="batched")
    sharpness: Mapped[int] = mapped_column(Integer, default=2)
    concern: Mapped[str] = mapped_column(Text, default="")
    state_json: Mapped[str] = mapped_column(Text)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    paused_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ConsentRecord(Base):
    """知情同意表（§3.2：版本化 + 分项授权）。"""

    __tablename__ = "consents"

    consent_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    policy_version: Mapped[str] = mapped_column(String(64))
    scopes_json: Mapped[str] = mapped_column(Text, default="[]")
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SafetyAuditRecord(Base):
    """安全审计表（§5.2 修正 #3：危机事件独立、append-only、user_id 哈希）。

    **只允许 insert。** 业务代码绝不 update/delete —— 审计痕迹不可篡改。
    """

    __tablename__ = "safety_audit"

    audit_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: user_id 的哈希，不是明文（§11.1 隐私红线）
    user_id_hash: Mapped[str] = mapped_column(String(64), index=True)
    risk_level: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(32))
    matched_rules: Mapped[list[Any]] = mapped_column(JSON, default=list)
    action_taken: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ReportRecord(Base):
    """报告表（§5.2）。payload 是 InsightReport 的 JSON 序列化。"""

    __tablename__ = "reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    payload_json: Mapped[str] = mapped_column(Text)
    degraded: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
