"""API 请求/响应模型（路线文档 §10.1）。

请求模型只接受**创建会话与提交回答所需的最小字段**，
不暴露内部结构（plan/qa_pairs 是系统内部状态，客户端不该直接构造）。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.mode import AskMode
from app.schemas.enums import ConsentScope, Sharpness


class ConsentIn(BaseModel):
    """客户端提交的知情同意。"""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    scopes: list[ConsentScope] = Field(default_factory=list)


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concern: str = Field(min_length=1, max_length=4000)
    mode: AskMode = AskMode.BATCHED
    sharpness: Sharpness = Sharpness.MODERATE
    is_minor: bool = False
    consent: ConsentIn | None = None

    @field_validator("sharpness", mode="before")
    @classmethod
    def _parse_sharpness(cls, v: object) -> object:
        """`Sharpness` 是 IntEnum，JSON 里传 "moderate" 会被拒。

        这里接受字符串名（客户端友好），转成 IntEnum。
        整数（1/2/3）也照常透传。
        """
        if isinstance(v, str):
            try:
                return Sharpness[v.upper()]
            except KeyError as exc:
                raise ValueError(
                    f"sharpness 必须是 gentle/moderate/sharp，收到 {v!r}"
                ) from exc
        return v


class SubmitAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)


class EventOut(BaseModel):
    """一个 SSE 事件的序列化形式（§10.2）。"""

    event: str
    data: dict[str, object] = Field(default_factory=dict)


class SessionOut(BaseModel):
    """会话状态响应。核心字段 + 进度 + 本轮事件。"""

    session_id: str
    phase: str
    mode: str
    sharpness: int
    concern: str
    progress: dict[str, object] = Field(default_factory=dict)
    open_questions: list[dict[str, object]] = Field(default_factory=list)
    report: dict[str, object] | None = None
    degraded: bool = False
    degrade_reasons: list[str] = Field(default_factory=list)
    events: list[EventOut] = Field(default_factory=list)
    consent_status: str = "none"  # none / pending / valid
    updated_at: datetime | None = None
