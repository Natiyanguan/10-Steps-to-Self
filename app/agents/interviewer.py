"""Interviewer Agent（路线文档 §4.4 / §3.7）。

职责边界：把一个规划槽位渲染成自然问句。**禁止给建议、禁止下判断**。

核心机制：输出必须过 `OutputGuard.check_question`。
命中违规则重生成；重生成仍失败降级到安全模板（§3.7）。

为什么不能只靠提示词：提示词里写"不要给建议"挡不住模型输出
"你有没有想过其实可以先辞职试试？" —— 那是伪装成问题的建议，
而它会直接违背苏格拉底方法的前提（§7.5-C 的"非引导性"维度）。
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import Agent
from app.domain.dimension import TOTAL_QUESTIONS
from app.domain.technique import spec
from app.llm.base import TaskKind
from app.llm.gateway import CallOutcome
from app.safety import output_guard
from app.schemas.enums import DegradeReason, Sharpness
from app.schemas.session import MAX_QUESTION_CHARS, PlanSlot, Question

MAX_GUARD_ATTEMPTS = 2  # 首次 + 一次重生成；仍失败则用安全模板

_SHARPNESS_LABEL: dict[Sharpness, str] = {
    Sharpness.GENTLE: "温和",
    Sharpness.MODERATE: "适度挑战",
    Sharpness.SHARP: "犀利直击",
}

_SHARPNESS_GUIDANCE: dict[Sharpness, str] = {
    Sharpness.GENTLE: "语气柔和，给足空间，不施加任何压力，允许对方回答得很浅。",
    Sharpness.MODERATE: "可以适度推进，触及对方可能一直在回避的角度，但保持尊重与余地。",
    Sharpness.SHARP: "直接切入要害，不绕弯。但仍然是提问，不是评判，不能带指责意味。",
}


class InterviewerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS * 2)


@dataclass(frozen=True, slots=True)
class QuestionResult:
    question: Question
    outcome: CallOutcome | None
    guard_retries: int = 0
    used_fallback: bool = False
    degrade_reasons: tuple[DegradeReason, ...] = ()


class Interviewer(Agent):
    prompt_agent = "interviewer"
    task = TaskKind.ASK

    async def ask(
        self,
        *,
        slot: PlanSlot,
        concern: str,
        sharpness: Sharpness,
        recent_exchanges: list[dict[str, str]] | None = None,
        summary: str = "",
        crisis_mode: bool = False,
        is_probe: bool = False,
        probe_reason: str = "",
    ) -> QuestionResult:
        tspec = spec(slot.technique)
        base_ctx: dict[str, object] = {
            "concern": concern,
            "index": slot.index,
            "total": TOTAL_QUESTIONS,
            "intent": slot.intent,
            "technique_label": tspec.label,
            "technique_pattern": tspec.pattern,
            "sharpness_label": _SHARPNESS_LABEL[sharpness],
            "sharpness_guidance": _SHARPNESS_GUIDANCE[sharpness],
            "max_chars": MAX_QUESTION_CHARS,
            "recent_exchanges": recent_exchanges or [],
            "summary": summary,
            "crisis_mode": crisis_mode,
            "is_probe": is_probe,
            "probe_reason": probe_reason or "信息不足",
        }

        last_outcome: CallOutcome | None = None
        guard_feedback = ""

        for attempt in range(1, MAX_GUARD_ATTEMPTS + 1):
            payload = "请提出这一问。"
            if guard_feedback:
                payload = (
                    f"上一次的问题不合格：{guard_feedback}\n"
                    "请重新提出一个纯粹的问句，不含任何建议或预设答案。"
                )

            prompt = self.render(base_ctx)
            out, outcome = await self.call(
                prompt, InterviewerOutput, user_payload=payload, max_output_tokens=400
            )
            last_outcome = outcome
            text = out.text.strip()

            # 超长截断到硬上限（§5.1 防认知过载）
            if len(text) > MAX_QUESTION_CHARS:
                text = text[:MAX_QUESTION_CHARS].rstrip()

            guard = output_guard.check_question(text)
            if guard.ok:
                return QuestionResult(
                    question=Question(
                        index=slot.index,
                        text=text,
                        technique=slot.technique,
                        sharpness=sharpness,
                        is_probe=is_probe,
                        prompt_version=prompt.ref,
                    ),
                    outcome=outcome,
                    guard_retries=attempt - 1,
                )
            guard_feedback = guard.summary

        # 重生成仍失败 → 安全模板（§3.7）。这是降级，必须可见（§9.5）。
        fallback = output_guard.safe_fallback()
        return QuestionResult(
            question=Question(
                index=slot.index,
                text=fallback,
                technique=slot.technique,
                sharpness=Sharpness.GENTLE,
                is_probe=is_probe,
                prompt_version="fallback",
            ),
            outcome=last_outcome,
            guard_retries=MAX_GUARD_ATTEMPTS,
            used_fallback=True,
            degrade_reasons=(DegradeReason.QUESTION_GUARD_FALLBACK,),
        )
