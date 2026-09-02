"""Safety Agent —— 危机判级第二层（路线文档 §3.4 / §9.3）。

职责边界（§4.4）：**只判级，不修改流程**。处置交给 `app/safety/policy.py`。

关键设计：这一层可以失败，系统仍必须安全。
  - 规则层（`lexicon`）先跑，零 LLM 依赖，命中即定级。
  - 分类器不可用时返回 `source=UNAVAILABLE`，policy 会走保守分支（HOLD），
    **绝不因为"分类器挂了"就当作无风险继续犀利追问**（§9.3）。
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import Agent
from app.llm.base import LLMError, TaskKind
from app.safety import lexicon
from app.schemas.enums import RiskLevel, RiskSource, risk_at_least
from app.schemas.session import RiskAssessment


class SafetyVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: RiskLevel
    reason: str = Field(default="", max_length=200)
    self_referential: bool = True


class SafetyAgent(Agent):
    prompt_agent = "safety"
    task = TaskKind.SAFETY

    async def assess(self, text: str) -> RiskAssessment:
        """判级。规则层优先，必要时叠加分类器。"""
        now = datetime.now(UTC)

        hit = lexicon.scan(text)
        if hit is not None:
            # 规则层命中 HIGH/IMMINENT：直接采信，不再问模型。
            # 理由：这类命中的处置是"立即终止提问"，而让模型有机会把它降级
            # 等于给漏检开了一道门 —— 假阴性的代价不可逆（§3.4）。
            if risk_at_least(hit.level, RiskLevel.HIGH):
                return RiskAssessment(
                    level=hit.level,
                    source=RiskSource.LEXICON,
                    matched_rules=list(hit.rule_names),
                    evidence=hit.matched_text,
                    assessed_at=now,
                )
            # MEDIUM 命中：让分类器有机会区分"自身处境"与"转述他人"，
            # 但只允许它**升级或维持**，不允许降到 MEDIUM 以下。
            verdict = await self._classify(text, hint=",".join(hit.rule_names))
            if verdict is None:
                return RiskAssessment(
                    level=hit.level,
                    source=RiskSource.LEXICON,
                    matched_rules=list(hit.rule_names),
                    evidence=hit.matched_text,
                    assessed_at=now,
                )
            level = verdict.level if risk_at_least(verdict.level, hit.level) else hit.level
            return RiskAssessment(
                level=level,
                source=RiskSource.CLASSIFIER,
                matched_rules=list(hit.rule_names),
                evidence=verdict.reason or hit.matched_text,
                assessed_at=now,
            )

        # 规则层未命中且语义不可疑：无风险，省一次调用（§3.4 控制成本）
        if not lexicon.is_suspicious(text):
            return RiskAssessment(
                level=RiskLevel.NONE, source=RiskSource.LEXICON, assessed_at=now
            )

        verdict = await self._classify(text)
        if verdict is None:
            # 分类器不可用 → 保守分支（§9.3）。这不是"无风险"。
            return RiskAssessment(
                level=RiskLevel.LOW,
                source=RiskSource.UNAVAILABLE,
                evidence="安全分类器不可用，无法评估语义风险",
                assessed_at=now,
            )
        return RiskAssessment(
            level=verdict.level,
            source=RiskSource.CLASSIFIER,
            evidence=verdict.reason,
            assessed_at=now,
        )

    async def _classify(self, text: str, *, hint: str = "") -> SafetyVerdict | None:
        """调用分类器。任何失败都返回 None —— 由调用方决定保守处置。"""
        prompt = self.render(
            {"text": text, "lexicon_hint": hint},
            version="classify_v1",
        )
        try:
            verdict, _ = await self.call(
                prompt, SafetyVerdict, user_payload="请判级。", max_output_tokens=200
            )
        except LLMError:
            return None
        return verdict


def unavailable_assessment(reason: str = "LLM 全部不可用") -> RiskAssessment:
    """LLM 完全不可用时的判级结果。规则层仍已跑过（调用方保证）。"""
    return RiskAssessment(
        level=RiskLevel.LOW,
        source=RiskSource.UNAVAILABLE,
        evidence=reason,
        assessed_at=datetime.now(UTC),
    )
