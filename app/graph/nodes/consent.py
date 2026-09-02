"""CONSENT 节点：知情同意闸门（路线文档 §3.2 / §4.3）。

设计要点：同意**不由模型收集**，由 API/CLI 层显式提交后写入状态。
本节点只做闸门：没有有效同意就停在此处，绝不推进到提问。

分项授权被拒绝时**功能降级而非拒绝服务**（§3.2）：
拒绝 `cross_session_recall` 的用户仍能完整使用单次会话。
"""

from __future__ import annotations

from typing import Any

from app.graph.deps import Deps
from app.graph.state import GraphState
from app.schemas.enums import ConsentScope, Phase, Sharpness


async def consent_gate(state: GraphState, deps: Deps) -> dict[str, Any]:
    consent = state.consent

    if consent is None or not consent.is_active:
        # 停在 CONSENT 等待前端提交。附上免责声明（§3.1 决策 #10：会话开始处出现）
        return {
            "turn_done": True,
            "emissions": [
                *state.emissions,
                state.emit(
                    "consent_required",
                    policy_version=deps.settings.policy_version,
                    disclaimer=deps.crisis.disclaimer,
                    scopes=[s.value for s in ConsentScope],
                ),
            ],
        }

    # 条款版本过期需重新征求（§3.2：旧版同意不自动继承）。
    # 判定统一走 Consent.is_current —— 与 intake 节点共用同一入口，避免两处发散。
    expected = deps.settings.policy_version
    if not consent.is_current(expected):
        return {
            "turn_done": True,
            "emissions": [
                *state.emissions,
                state.emit(
                    "consent_required",
                    policy_version=expected,
                    reason=f"条款已更新（你同意的是 {consent.policy_version}）",
                    disclaimer=deps.crisis.disclaimer,
                    scopes=[s.value for s in ConsentScope],
                ),
            ],
        }

    update: dict[str, Any] = {"phase": Phase.PLAN}

    # 未成年人强制 GENTLE 且不可上调（§3.2 不可配置的硬约束）
    if state.is_minor and state.sharpness != Sharpness.GENTLE:
        update["sharpness"] = Sharpness.GENTLE
        update["sharpness_downgrades"] = [
            *state.sharpness_downgrades,
            "未成年人：强制温和模式（§3.2）",
        ]

    return update
