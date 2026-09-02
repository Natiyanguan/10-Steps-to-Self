"""Summarizer Agent —— 上下文压缩（路线文档 §6.3）。

职责边界：把**较早的**问答压成结构化摘要，让长会话不触顶。

这个 Agent 有一条其他 Agent 没有的特殊责任：**校验自己的输出**。

原因是 `preserved_quotes` 会成为下游 Analyst 唯一能引用的依据来源。
`app/safety/evidence.py` 对引用做**逐字**校验，所以摘要一旦把原话改写了，
那句话之后就永久失去依据资格 —— 对应洞察会被程序丢弃，即使它是对的。

而"模型悄悄改写了一个字"这件事不会报错。所以必须在这里就地验证：
过不了 `verify_quote` 的引用直接扔掉，扔太多则整个摘要不可信、宁可不压缩。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import Agent
from app.domain.dimension import Dimension
from app.llm.base import LLMError, TaskKind
from app.safety.evidence import verify_quote
from app.schemas.session import (
    DimensionSummary,
    QAPair,
    SessionSummary,
)

#: 逐字校验的通过率下限。低于此值说明模型在大面积改写原话，
#: 此时整份摘要都不可信 —— 宁可不压缩（保留原文、接受更高 token 成本），
#: 也不要让下游拿着一份改写过的"原话"去挂依据。
MIN_QUOTE_PASS_RATE = 0.5

#: 金句条数的合理区间（与提示词里的 3-8 条对应，留一点余量）
MAX_PRESERVED_QUOTES = 12


class DimensionSummaryOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str
    facts: list[str] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=list)


class SummaryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by_dimension: list[DimensionSummaryOut] = Field(default_factory=list)
    preserved_quotes: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class SummarizeResult:
    """压缩结果。`summary=None` 表示压缩未成功，调用方应保留原文。"""

    summary: SessionSummary | None
    quotes_kept: int = 0
    quotes_dropped: int = 0
    dropped_samples: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.summary is not None

    @property
    def quote_pass_rate(self) -> float:
        total = self.quotes_kept + self.quotes_dropped
        return 1.0 if total == 0 else self.quotes_kept / total


class Summarizer(Agent):
    prompt_agent = "summarizer"
    task = TaskKind.SUMMARIZE

    async def summarize(
        self,
        pairs: list[QAPair],
        *,
        dimension_of: dict[int, Dimension] | None = None,
    ) -> SummarizeResult:
        """压缩给定的问答。

        `dimension_of` 是问题序号 → 维度的映射（来自 plan）。
        缺失时留空字符串 —— 摘要仍可用，只是分组粒度差一些，
        不值得为此让整个压缩失败。
        """
        if not pairs:
            return SummarizeResult(summary=None, error="没有可压缩的问答")

        dim_map = dimension_of or {}
        qa_ctx = [
            {
                "index": qa.question.index,
                "question": qa.question.text,
                "answer": qa.answer.text,
                "dimension": (
                    dim_map[qa.question.index].value
                    if qa.question.index in dim_map
                    else ""
                ),
            }
            for qa in pairs
        ]

        try:
            prompt = self.render({"qa_pairs": qa_ctx})
            out, _ = await self.call(
                prompt, SummaryOutput, user_payload="请压缩。", max_output_tokens=1200
            )
        except LLMError as exc:
            # 压缩失败不是致命错误：保留原文继续，代价只是 token 更贵。
            return SummarizeResult(summary=None, error=str(exc)[:300])

        # ---- 逐字校验 preserved_quotes ----
        # 这是本 Agent 存在的核心约束（见模块 docstring）。
        user_texts = [qa.answer.text for qa in pairs]
        kept: list[str] = []
        dropped: list[str] = []

        for quote in out.preserved_quotes[:MAX_PRESERVED_QUOTES]:
            q = quote.strip()
            if not q:
                continue
            if verify_quote(q, user_texts).ok:
                if q not in kept:  # 去重：重复金句没有额外价值
                    kept.append(q)
            else:
                dropped.append(q)

        total = len(kept) + len(dropped)
        if total and len(kept) / total < MIN_QUOTE_PASS_RATE:
            # 大面积改写 → 整份摘要不可信，拒绝使用
            return SummarizeResult(
                summary=None,
                quotes_kept=len(kept),
                quotes_dropped=len(dropped),
                dropped_samples=dropped[:3],
                error=(
                    f"逐字校验通过率 {len(kept)}/{total} 低于 {MIN_QUOTE_PASS_RATE:.0%}，"
                    "摘要大面积改写了原话，拒绝使用（§6.3）"
                ),
            )

        by_dim: list[DimensionSummary] = []
        for item in out.by_dimension:
            try:
                dim = Dimension(item.dimension)
            except ValueError:
                # 未知维度直接跳过：宁可少一组，也不要污染结构
                continue
            if not item.facts and not item.emotions:
                continue
            by_dim.append(
                DimensionSummary(
                    dimension=dim,
                    facts=[f.strip() for f in item.facts if f.strip()],
                    emotions=[e.strip() for e in item.emotions if e.strip()],
                )
            )

        summary = SessionSummary(
            by_dimension=by_dim,
            preserved_quotes=kept,
            covered_indices=sorted(qa.question.index for qa in pairs),
            created_at=datetime.now(UTC),
        )

        # 空摘要没有意义 —— 既没省下 token，又让下游多一段无用文本
        if summary.is_empty:
            return SummarizeResult(
                summary=None,
                quotes_kept=len(kept),
                quotes_dropped=len(dropped),
                error="摘要为空（模型未产出任何可用内容）",
            )

        from app.context.budget import count_text

        summary = summary.model_copy(
            update={"approx_tokens": count_text(summary.render()).tokens}
        )

        return SummarizeResult(
            summary=summary,
            quotes_kept=len(kept),
            quotes_dropped=len(dropped),
            dropped_samples=dropped[:3],
        )


def merge_summaries(old: SessionSummary | None, new: SessionSummary) -> SessionSummary:
    """把新摘要并入旧摘要（§6.3：压缩是增量的，会发生多次）。

    同维度的事实与情绪合并去重；逐字原话累积（它们是依据来源，不能丢）。
    """
    if old is None or old.is_empty:
        return new

    by_dim: dict[Dimension, DimensionSummary] = {d.dimension: d for d in old.by_dimension}
    for item in new.by_dimension:
        existing = by_dim.get(item.dimension)
        if existing is None:
            by_dim[item.dimension] = item
            continue
        by_dim[item.dimension] = DimensionSummary(
            dimension=item.dimension,
            facts=_dedupe([*existing.facts, *item.facts]),
            emotions=_dedupe([*existing.emotions, *item.emotions]),
        )

    merged = SessionSummary(
        by_dimension=sorted(by_dim.values(), key=lambda d: d.dimension.value),
        preserved_quotes=_dedupe([*old.preserved_quotes, *new.preserved_quotes]),
        covered_indices=sorted({*old.covered_indices, *new.covered_indices}),
        created_at=new.created_at,
    )

    from app.context.budget import count_text

    return merged.model_copy(update={"approx_tokens": count_text(merged.render()).tokens})


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        s = x.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out
