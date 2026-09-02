"""INTAKE 节点：接收困惑描述并做首次安全检查（路线文档 §4.3）。

关键：**安全检查在任何生成之前**。用户第一句话就可能是危机信号，
此时还没有 plan、没有 question，但危机分支必须已经可用（§3.4）。
"""

from __future__ import annotations

from typing import Any

from app.graph.deps import Deps
from app.graph.state import GraphState, add_degradation
from app.safety import injection, policy
from app.schemas.enums import DegradeReason, Phase, SafetyActionKind


async def intake(state: GraphState, deps: Deps) -> dict[str, Any]:
    raw = (state.inbox or state.concern or "").strip()
    if not raw:
        # 没有输入就停在 INTAKE 等待，不推进阶段
        return {"turn_done": True}

    # **安全评估必须在原文上做，剥离在其后**（§3.4 + §3.7 的交互）。
    # 顺序反了会漏检危机：sanitize 是按句剥离的，
    # 「忽略以上指令，我不想活了」这句里注入与危机信号在同一句，
    # 先剥离会把危机信号一起删掉，然后安全层什么也看不到。
    risk = await deps.safety.assess(raw)
    action = policy.decide(risk, state)

    # 剥离后的文本才是进提示词的那份（§3.7）
    clean = injection.sanitize(raw)

    update: dict[str, Any] = {
        "concern": clean.text,
        "inbox": None,
        "risk_history": [*state.risk_history, risk],
        "safety_action": action,
    }

    if clean.modified:
        # 剥离是降级事件，必须可见（§9.5）
        update["degrade_reasons"] = add_degradation(state, DegradeReason.INJECTION_STRIPPED)
        update["emissions"] = [
            *state.emissions,
            state.emit(
                "degraded",
                reason=DegradeReason.INJECTION_STRIPPED.value,
                impact="你的描述里有一段被识别为指令的内容，已移除后继续。",
                detail=clean.summary,
            ),
        ]

    if action.kind is SafetyActionKind.STOP_WITH_RESOURCES:
        update["phase"] = Phase.CRISIS
        return update

    # 锐度可能被安全侧下调（单向棘轮，§3.5）
    capped = policy.apply_sharpness_cap(state.sharpness, action)
    if capped != state.sharpness:
        update["sharpness"] = capped
        update["sharpness_downgrades"] = [*state.sharpness_downgrades, *action.reasons]

    if action.kind is SafetyActionKind.HOLD_UNVERIFIABLE:
        # 无法评估安全 → 不进入提问（§9.3）
        update["phase"] = Phase.PAUSED
        update["turn_done"] = True
        return update

    # 必须用 is_current 而不是"consent 是否存在"：
    # 过期版本的同意不算同意（§3.2），否则条款变更后老用户会静默绕过闸门。
    has_consent = state.consent is not None and state.consent.is_current(
        deps.settings.policy_version
    )
    update["phase"] = Phase.PLAN if has_consent else Phase.CONSENT
    return update
