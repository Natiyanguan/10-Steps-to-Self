"""回答质量评估 Agent（路线文档 §4.3 LISTEN → PROBE）。

职责边界：只评估"这个回答提供了多少可用材料"，**不决定是否追问**。
追问决策由图节点按 `probe_counts` 上限与阈值决定。

失败即降级为"不追问"：宁可少问一次，也不要因为评估器挂了就卡住流程。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import Agent
from app.llm.base import LLMError, TaskKind
from app.safety.policy import detect_evasion
from app.schemas.session import AnswerQuality, SessionState

# 低于此分视为信息不足，可考虑追问（§4.3）
PROBE_SCORE_THRESHOLD = 0.5
MIN_ANSWER_CHARS = 6


class QualityVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    is_evasive: bool = False
    is_too_brief: bool = False
    missing_aspects: list[str] = Field(default_factory=list)


class QualityAgent(Agent):
    prompt_agent = "quality"
    task = TaskKind.QUALITY

    async def evaluate(
        self, *, question: str, intent: str, answer: str, state: SessionState | None = None
    ) -> AnswerQuality:
        # 极短回答用规则判定即可，不必花一次调用
        stripped = answer.strip()
        if len(stripped) < MIN_ANSWER_CHARS:
            return AnswerQuality(
                score=0.1,
                is_evasive=False,
                is_too_brief=True,
                missing_aspects=["回答过短，几乎没有信息"],
            )

        prompt = self.render({"question": question, "intent": intent, "answer": stripped})
        try:
            verdict, _ = await self.call(
                prompt, QualityVerdict, user_payload="请评估。", max_output_tokens=250
            )
        except LLMError:
            # 评估不可用 → 视为质量足够，不追问（不阻塞流程）
            return AnswerQuality(score=0.7, missing_aspects=[])

        missing = list(verdict.missing_aspects)

        # 叠加规则层的回避信号（§3.5）：长度骤降是跨轮次信号，模型看不到
        evasion = detect_evasion(state, stripped) if state is not None else None
        if evasion:
            missing.append(evasion)

        return AnswerQuality(
            score=verdict.score,
            is_evasive=verdict.is_evasive or evasion is not None,
            is_too_brief=verdict.is_too_brief,
            missing_aspects=missing[:3],
        )


def needs_probe(quality: AnswerQuality, *, can_probe: bool) -> tuple[bool, str]:
    """是否应追问。纯函数，便于单测（§4.3）。"""
    if not can_probe:
        return False, ""
    if quality.score >= PROBE_SCORE_THRESHOLD and not quality.is_evasive:
        return False, ""
    reasons: list[str] = []
    if quality.is_too_brief:
        reasons.append("信息量不足")
    if quality.is_evasive:
        reasons.append("似乎绕开了核心")
    if quality.missing_aspects:
        reasons.append("；".join(quality.missing_aspects[:2]))
    return True, "，".join(reasons) or "信息不足"
