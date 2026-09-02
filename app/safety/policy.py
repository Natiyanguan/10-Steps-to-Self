"""危机处置矩阵（路线文档 §3.4）。

设计约束：
  - **纯函数**，可完全离线单测，覆盖率要求 100%。
  - 处置由代码分支决定，**不由 LLM 自由决定**。
  - 危机资源信息从配置读取，**不由模型生成**（§16.2 硬约束）。
  - `HOLD_UNVERIFIABLE`：安全无法评估时的保守分支（§9.3）——
    绝不在无法评估安全的情况下继续犀利追问。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.enums import (
    RiskLevel,
    RiskSource,
    SafetyActionKind,
    Sharpness,
    risk_at_least,
)
from app.schemas.session import RiskAssessment, SessionState


@dataclass(frozen=True, slots=True)
class SafetyAction:
    kind: SafetyActionKind
    max_sharpness: Sharpness           # 本轮允许的最高锐度
    block_deep_dimensions: bool        # 是否禁用恐惧/身份等深度维度
    allow_further_questions: bool      # 是否还允许继续提问
    show_crisis_resources: bool        # 是否展示危机资源
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_stop(self) -> bool:
        return self.kind is SafetyActionKind.STOP_WITH_RESOURCES

    @property
    def crisis_mode(self) -> bool:
        """技术过滤用：非 PROCEED 即视为危机模式（§1.3 crisis_safe 过滤）。"""
        return self.kind is not SafetyActionKind.PROCEED


def _minor_cap(state: SessionState, cap: Sharpness, reasons: list[str]) -> Sharpness:
    """未成年人强制 GENTLE 且不可上调（§3.2 不可配置的硬约束）。"""
    if state.is_minor:
        reasons.append("未成年人：强制温和模式（§3.2）")
        return Sharpness.GENTLE
    return cap


def decide(risk: RiskAssessment, state: SessionState) -> SafetyAction:
    """根据风险判级与会话状态决定处置。纯函数。

    处置矩阵（§3.4）：
      none / low        → 正常提问
      medium            → 中断节奏、降锐度、提示资源，但会话继续
      high / imminent   → 立即终止提问，输出危机资源
      分类器不可用      → 保守暂停（HOLD），不进入深度提问
    """
    reasons: list[str] = []

    # 分类器不可用且规则层未命中：无法评估安全 → 保守处理（§9.3）
    if risk.source is RiskSource.UNAVAILABLE:
        reasons.append("安全分类器不可用，无法评估风险，采取保守策略（§9.3）")
        return SafetyAction(
            kind=SafetyActionKind.HOLD_UNVERIFIABLE,
            max_sharpness=Sharpness.GENTLE,
            block_deep_dimensions=True,
            allow_further_questions=False,
            show_crisis_resources=False,
            reasons=tuple(reasons),
        )

    if risk_at_least(risk.level, RiskLevel.HIGH):
        reasons.append(f"风险等级 {risk.level.value}：立即终止提问流程（§3.4）")
        if risk.matched_rules:
            reasons.append("命中规则：" + ",".join(risk.matched_rules))
        return SafetyAction(
            kind=SafetyActionKind.STOP_WITH_RESOURCES,
            max_sharpness=Sharpness.GENTLE,
            block_deep_dimensions=True,
            allow_further_questions=False,
            show_crisis_resources=True,
            reasons=tuple(reasons),
        )

    if risk.level is RiskLevel.MEDIUM:
        reasons.append("风险等级 medium：切换关怀性回应、锐度降级、提示专业资源（§3.4）")
        cap = _minor_cap(state, Sharpness.GENTLE, reasons)
        return SafetyAction(
            kind=SafetyActionKind.SOFTEN,
            max_sharpness=cap,
            block_deep_dimensions=True,
            allow_further_questions=True,
            show_crisis_resources=True,
            reasons=tuple(reasons),
        )

    # none / low：正常路径，但仍受未成年人与用户选择的锐度上限约束
    cap = _minor_cap(state, state.sharpness, reasons)
    return SafetyAction(
        kind=SafetyActionKind.PROCEED,
        max_sharpness=cap,
        block_deep_dimensions=state.is_minor,
        allow_further_questions=True,
        show_crisis_resources=False,
        reasons=tuple(reasons),
    )


# --------------------------------------------------------------------------
# 锐度自动降级（§3.5）
# --------------------------------------------------------------------------

EVASION_LENGTH_RATIO = 0.4  # 回答长度骤降到历史均值的此比例以下 → 回避信号
MIN_ANSWERS_FOR_TREND = 2


def detect_evasion(state: SessionState, latest_text: str) -> str | None:
    """回答长度骤降是回避信号（§3.5）。返回降级原因或 None。"""
    prior = [qa.answer.text for qa in state.qa_pairs]
    if len(prior) < MIN_ANSWERS_FOR_TREND:
        return None
    avg = sum(len(t) for t in prior) / len(prior)
    if avg <= 0:
        return None
    if len(latest_text) < avg * EVASION_LENGTH_RATIO:
        return f"回答长度骤降（{len(latest_text)} 字 vs 均值 {avg:.0f} 字），疑似回避"
    return None


def apply_sharpness_cap(current: Sharpness, action: SafetyAction) -> Sharpness:
    """锐度只能下调不能上调 —— 安全侧的单向棘轮。"""
    return Sharpness(min(int(current), int(action.max_sharpness)))
