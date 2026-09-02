"""图状态（路线文档 §4.3 / §4.5）。

`GraphState` 在 `SessionState` 之上加了**仅编排期需要**的字段：
待处理的用户输入、本轮要发给前端的事件、安全处置结果。

为什么继承而不是塞进 SessionState：
持久化的会话状态不应包含"本轮的临时输入"，
否则 checkpoint 恢复时会重放一次已经处理过的输入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ConfigDict, Field

from app.safety.policy import SafetyAction
from app.schemas.enums import DegradeReason, Phase
from app.schemas.session import SessionState


@dataclass(slots=True)
class Emission:
    """一条要推给前端的事件（§10.2 SSE 事件类型）。"""

    event: str
    data: dict[str, Any] = field(default_factory=dict)


class GraphState(SessionState):
    """编排期状态。节点只返回增量更新字段（§4.5 生成指令）。"""

    #: 本轮待处理的用户输入（困惑描述或回答）。处理后必须清空。
    inbox: str | None = None
    #: 本轮产生的事件，由 API 层转成 SSE
    emissions: list[Emission] = Field(default_factory=list)
    #: 最近一次安全处置结果（由 listen/intake 写入，供路由与 ask 使用）
    safety_action: SafetyAction | None = None
    #: 本轮是否已终止（危机分支或等待用户输入）
    turn_done: bool = False
    #: 需要追问的原因（listen 写入，probe 读取后清空）
    probe_reason: str = ""
    #: Critic 否决后的反馈，供 Analyst 重做时参考（§4.3 最多 2 轮）
    critic_feedback: list[str] = Field(default_factory=list)
    #: Analyst 重做轮次计数
    analyze_rounds: int = 0

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def emit(self, event: str, **data: Any) -> Emission:
        return Emission(event=event, data=data)


def phase_update(phase: Phase, **extra: Any) -> dict[str, Any]:
    """构造一个阶段迁移的增量更新。"""
    return {"phase": phase, **extra}


def add_degradation(state: GraphState, reason: DegradeReason) -> list[DegradeReason]:
    """追加降级原因（去重）。降级必须可见（§9.5）。"""
    if reason in state.degrade_reasons:
        return list(state.degrade_reasons)
    return [*state.degrade_reasons, reason]
