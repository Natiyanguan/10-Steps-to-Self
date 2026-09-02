"""Critic Agent —— 独立复核（路线文档 §4.4 铁律 2 / §6.4 第四层）。

两条职责边界：
  1. **只否决不改写**。改写权属于 Analyst。若两边都能改，会出现互相覆盖的振荡，
     最终谁都说不清结论是怎么来的。
  2. **不接收 Analyst 的推理过程**，只看结论 + 原始问答 —— 防 anchoring。
     被上游思路带偏的复核等于没有复核。

它负责程序校验查不出来的那一类问题：**引用真实但推论跳跃**。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agents.analyst import _qa_context
from app.agents.base import Agent
from app.llm.base import LLMError, TaskKind
from app.schemas.session import CritiqueResult, InsightReport, SessionState

MAX_CRITIC_ROUNDS = 2  # §4.3：自检不通过最多重做 2 次


@dataclass(slots=True)
class CritiqueOutcome:
    result: CritiqueResult | None
    error: str = ""

    @property
    def approved(self) -> bool:
        """复核不可用时视为通过 —— 但调用方必须记降级（§9.5）。

        理由：Critic 是第四层防线，前三层（提示词约束、结构化强制挂依据、
        Evidence 程序化校验）已经跑过。因它不可用就整份报告失败，
        代价大于收益；但必须让用户知道少了一道复核。
        """
        return self.result.approved if self.result is not None else True


class Critic(Agent):
    prompt_agent = "critic"
    task = TaskKind.CRITIQUE

    async def review(self, report: InsightReport, state: SessionState) -> CritiqueOutcome:
        items = self._flatten(report)
        if not items:
            return CritiqueOutcome(result=CritiqueResult(approved=True))

        prompt = self.render(
            {
                "concern": state.concern,
                "qa_pairs": _qa_context(state),
                "items": items,
            }
        )
        try:
            result, _ = await self.call(
                prompt, CritiqueResult, user_payload="请复核。", max_output_tokens=800
            )
        except LLMError as exc:
            return CritiqueOutcome(result=None, error=str(exc)[:300])
        return CritiqueOutcome(result=result)

    @staticmethod
    def _flatten(report: InsightReport) -> list[dict[str, object]]:
        """摊平成待复核条目。

        注意只传 statement / evidence / confidence ——
        不传 Analyst 的 rationale 或推理链（防 anchoring）。
        """
        items: list[dict[str, object]] = []
        for group, insights in (
            ("underlying_needs", report.underlying_needs),
            ("inner_conflicts", report.inner_conflicts),
        ):
            for i, ins in enumerate(insights):
                items.append(
                    {
                        "target": f"{group}[{i}]",
                        "statement": ins.statement,
                        "confidence": ins.confidence.value,
                        "evidence": [
                            {"question_index": e.question_index, "quote": e.quote}
                            for e in ins.evidence
                        ],
                    }
                )
        return items


def feedback_for_retry(result: CritiqueResult) -> list[str]:
    """把否决意见转成 Analyst 可用的具体反馈（§4.4：反馈必须可操作）。"""
    return [f"{i.target}：{i.problem}" for i in result.issues]
