"""压缩触发决策（路线文档 §6.3）。

**纯函数**：决定"要不要压缩、压哪几轮"，不发起任何调用。
这样才能对边界条件做穷尽单测 —— 压缩策略错了的表现是长会话突然触顶，
那种故障在生产上极难复现。

与 `budget.py` 的分工：
  - `budget.py` 回答"额度够不够、超了怎么裁"
  - 本模块回答"什么时候该压缩、压缩哪些"

两者刻意分开：裁剪（丢弃）是最后的兜底手段，压缩（摘要）是主动手段。
只有裁剪没有压缩，长会话会静默丢掉早期信息；
只有压缩没有裁剪，压缩失败时就没有退路了。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.context.budget import count_text
from app.schemas.session import QAPair, SessionSummary

#: 最近多少轮保留原文。与 `interviewer.RECENT_EXCHANGES` 对应 ——
#: 提问的连贯性依赖近期原文，压缩它们会让问题显得答非所问。
KEEP_RECENT_PAIRS = 3

#: 触发压缩的水位：历史占用超过可用额度的这个比例就压缩。
#: 不等到 100% 才动手 —— 那时已经来不及，
#: 因为压缩本身也要一次 LLM 调用，而调用可能失败。
COMPRESS_WATERMARK = 0.6

#: 少于这么多轮不压缩：压缩本身要花一次调用，
#: 轮数太少时省下的 token 抵不上调用成本。
MIN_PAIRS_TO_COMPRESS = 2


@dataclass(slots=True)
class CompressionPlan:
    """压缩计划。`should_compress=False` 时其余字段无意义。"""

    should_compress: bool
    to_compress: list[QAPair] = field(default_factory=list)
    keep_verbatim: list[QAPair] = field(default_factory=list)
    history_tokens: int = 0
    available_tokens: int = 0
    reason: str = ""

    @property
    def pressure(self) -> float:
        """当前压力比。> 1 表示已经超出额度。"""
        if self.available_tokens <= 0:
            return 0.0
        return self.history_tokens / self.available_tokens


def history_tokens(
    pairs: list[QAPair],
    summary: SessionSummary | None = None,
    *,
    tokenizer_hint: str = "cl100k_base",
) -> int:
    """估算历史占用。含摘要自身 —— 摘要也占额度，忽略它会低估压力。"""
    total = 0
    for qa in pairs:
        total += count_text(qa.question.text, tokenizer_hint=tokenizer_hint).tokens
        total += count_text(qa.answer.text, tokenizer_hint=tokenizer_hint).tokens
    if summary is not None and not summary.is_empty:
        total += summary.approx_tokens or count_text(
            summary.render(), tokenizer_hint=tokenizer_hint
        ).tokens
    return total


def plan_compression(
    pairs: list[QAPair],
    *,
    available_tokens: int,
    summary: SessionSummary | None = None,
    keep_recent: int = KEEP_RECENT_PAIRS,
    watermark: float = COMPRESS_WATERMARK,
    tokenizer_hint: str = "cl100k_base",
) -> CompressionPlan:
    """决定是否压缩，以及压缩哪几轮。

    只压缩**已被摘要覆盖之外**的早期问答：
    重复压缩同一批问答毫无收益，还会让金句在多轮摘要间反复改写 ——
    每次改写都增加一次逐字校验失败的机会。
    """
    covered = set(summary.covered_indices) if summary is not None else set()
    uncovered = [qa for qa in pairs if qa.question.index not in covered]

    used = history_tokens(pairs, summary, tokenizer_hint=tokenizer_hint)
    plan = CompressionPlan(
        should_compress=False,
        history_tokens=used,
        available_tokens=available_tokens,
    )

    if available_tokens <= 0:
        plan.reason = "无可用额度信息，跳过压缩决策"
        return plan

    if used < available_tokens * watermark:
        plan.reason = f"历史占用 {used} 未达水位（{available_tokens} × {watermark:.0%}）"
        return plan

    # 保留最近 keep_recent 轮原文，其余候选压缩
    candidates = uncovered[:-keep_recent] if keep_recent > 0 else uncovered
    keep = uncovered[-keep_recent:] if keep_recent > 0 else []

    if len(candidates) < MIN_PAIRS_TO_COMPRESS:
        plan.reason = (
            f"可压缩轮数 {len(candidates)} 少于 {MIN_PAIRS_TO_COMPRESS}，"
            "压缩调用的成本大于收益"
        )
        return plan

    plan.should_compress = True
    plan.to_compress = candidates
    plan.keep_verbatim = keep
    plan.reason = (
        f"历史占用 {used} 达到水位（{available_tokens} × {watermark:.0%}），"
        f"压缩最早 {len(candidates)} 轮，保留最近 {len(keep)} 轮原文"
    )
    return plan
