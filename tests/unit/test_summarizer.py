"""Summarizer 单测（路线文档 §6.3）。

两条核心要验证的：
1. `merge_summaries` 的纯函数行为（增量合并、去重、逐字原话不丢）。
2. `Summarizer.summarize` 的**逐字校验拒绝逻辑** —— 这是这个 Agent 存在的原因：
   `preserved_quotes` 一旦改写，下游 Evidence 校验会静默丢弃对应洞察，
   所以必须在生成处就挡住改写。

用 FakeProvider 注入 SUMMARIZE 响应，不需要真实 LLM。
"""

from __future__ import annotations

import json

import pytest

from app.agents.summarizer import (
    MIN_QUOTE_PASS_RATE,
    SummarizeResult,
    merge_summaries,
)
from app.domain.dimension import Dimension
from app.domain.technique import Technique
from app.llm.base import TaskKind
from app.llm.providers.fake import Scripted
from app.schemas.enums import Sharpness
from app.schemas.session import (
    Answer,
    DimensionSummary,
    QAPair,
    Question,
    SessionSummary,
)

# 逐字引用必须从这里复制 —— 用 conftest 风格的长句，含真实可引用的子串。
ANSWER_A = "我最怕失去现在稳定的生活，但又不甘心一直这样耗下去。"
ANSWER_B = "我真正想要的是能自己掌控节奏的工作。"
QUOTE_A = "我最怕失去现在稳定的生活"
QUOTE_B = "能自己掌控节奏的工作"


def _pair(index: int, text: str) -> QAPair:
    return QAPair(
        question=Question(
            index=index,
            text=f"第{index}个问题？",
            technique=Technique.SCALING,
            sharpness=Sharpness.MODERATE,
        ),
        answer=Answer(question_index=index, text=text),
    )


def _summary(
    *,
    quotes: list[str] | None = None,
    covered: list[int] | None = None,
    by_dim: list[DimensionSummary] | None = None,
) -> SessionSummary:
    return SessionSummary(
        preserved_quotes=quotes or [],
        covered_indices=covered or [],
        by_dimension=by_dim or [],
    )


# --------------------------------------------------------------------------
# merge_summaries（纯函数）
# --------------------------------------------------------------------------
class TestMergeSummaries:
    def test_old_none_returns_new(self) -> None:
        new = _summary(quotes=[QUOTE_A], covered=[1])
        assert merge_summaries(None, new) is new

    def test_old_empty_returns_new_content(self) -> None:
        """空 old 合并后内容等价于 new（不要求同一对象引用）。"""
        empty = SessionSummary()
        new = _summary(quotes=[QUOTE_A], covered=[1])
        merged = merge_summaries(empty, new)
        assert merged.preserved_quotes == new.preserved_quotes
        assert merged.covered_indices == new.covered_indices
        assert merged.by_dimension == new.by_dimension

    def test_quotes_accumulate_and_dedupe(self) -> None:
        old = _summary(quotes=[QUOTE_A], covered=[1])
        new = _summary(quotes=[QUOTE_A, QUOTE_B], covered=[2])
        merged = merge_summaries(old, new)
        assert merged.preserved_quotes == [QUOTE_A, QUOTE_B]

    def test_covered_indices_union(self) -> None:
        old = _summary(covered=[1, 2])
        new = _summary(covered=[3, 4])
        merged = merge_summaries(old, new)
        assert merged.covered_indices == [1, 2, 3, 4]

    def test_same_dimension_merges_and_dedupes(self) -> None:
        old = _summary(
            by_dim=[DimensionSummary(dimension=Dimension.VALUE, facts=["重视稳定"])]
        )
        new = _summary(
            by_dim=[DimensionSummary(dimension=Dimension.VALUE, facts=["重视稳定", "渴望成长"])]
        )
        merged = merge_summaries(old, new)
        value_items = [d for d in merged.by_dimension if d.dimension is Dimension.VALUE]
        assert len(value_items) == 1
        assert value_items[0].facts == ["重视稳定", "渴望成长"]

    def test_distinct_dimensions_kept_separate(self) -> None:
        old = _summary(by_dim=[DimensionSummary(dimension=Dimension.VALUE, facts=["a"])])
        new = _summary(by_dim=[DimensionSummary(dimension=Dimension.FEAR, facts=["b"])])
        merged = merge_summaries(old, new)
        assert {d.dimension for d in merged.by_dimension} == {Dimension.VALUE, Dimension.FEAR}

    def test_emotions_merge_dedupe(self) -> None:
        old = _summary(
            by_dim=[DimensionSummary(dimension=Dimension.EMOTION, emotions=["焦虑"])]
        )
        new = _summary(
            by_dim=[DimensionSummary(dimension=Dimension.EMOTION, emotions=["焦虑", "矛盾"])]
        )
        merged = merge_summaries(old, new)
        emo = next(d for d in merged.by_dimension if d.dimension is Dimension.EMOTION)
        assert emo.emotions == ["焦虑", "矛盾"]


# --------------------------------------------------------------------------
# Summarizer.summarize（逐字校验）
# --------------------------------------------------------------------------
class TestSummarize:
    def _script(self, fake, by_dimension, preserved_quotes):
        fake.script(
            TaskKind.SUMMARIZE,
            Scripted(
                text=json.dumps(
                    {"by_dimension": by_dimension, "preserved_quotes": preserved_quotes},
                    ensure_ascii=False,
                )
            ),
        )

    @pytest.mark.asyncio
    async def test_empty_pairs_returns_error(self, deps) -> None:
        result = await deps.summarizer.summarize([])
        assert not result.ok
        assert result.summary is None
        assert result.error

    @pytest.mark.asyncio
    async def test_verbatim_quotes_kept(self, deps, fake) -> None:
        self._script(
            fake,
            by_dimension=[{"dimension": "value", "facts": ["重视稳定"], "emotions": []}],
            preserved_quotes=[QUOTE_A, QUOTE_B],
        )
        pairs = [_pair(1, ANSWER_A), _pair(2, ANSWER_B)]
        result = await deps.summarizer.summarize(pairs)

        assert result.ok
        assert result.quotes_kept == 2
        assert result.quotes_dropped == 0
        assert set(result.summary.preserved_quotes) == {QUOTE_A, QUOTE_B}
        assert result.summary.covered_indices == [1, 2]

    @pytest.mark.asyncio
    async def test_fabricated_quotes_dropped(self, deps, fake) -> None:
        """改写/编造的原话必须被丢弃（§6.3 逐字校验）。"""
        self._script(
            fake,
            by_dimension=[],
            preserved_quotes=[QUOTE_A, "我其实想去环游世界"],  # 第二条是编造
        )
        pairs = [_pair(1, ANSWER_A), _pair(2, ANSWER_B)]
        result = await deps.summarizer.summarize(pairs)

        assert result.ok  # 通过率 1/2 >= 0.5，仍可用
        assert result.quotes_kept == 1
        assert result.quotes_dropped == 1
        assert result.summary.preserved_quotes == [QUOTE_A]

    @pytest.mark.asyncio
    async def test_mass_rewrite_rejects_whole_summary(self, deps, fake) -> None:
        """大面积改写 → 通过率低于 MIN_QUOTE_PASS_RATE，整份摘要拒绝（§6.3）。"""
        fabricated = [f"编造的第{i}句话，不存在于原文" for i in range(5)]
        self._script(fake, by_dimension=[{"dimension": "value", "facts": ["x"], "emotions": []}],
                     preserved_quotes=fabricated)
        pairs = [_pair(1, ANSWER_A), _pair(2, ANSWER_B)]
        result = await deps.summarizer.summarize(pairs)

        assert not result.ok
        assert result.summary is None
        assert result.quotes_dropped == 5
        assert "通过率" in result.error

    @pytest.mark.asyncio
    async def test_llm_error_returns_error_not_exception(self, deps, fake) -> None:
        """压缩失败不抛异常 —— 调用方据此保留原文继续（§6.3）。"""
        fake.script_timeout(TaskKind.SUMMARIZE)
        pairs = [_pair(1, ANSWER_A)]
        result = await deps.summarizer.summarize(pairs)
        assert not result.ok
        assert result.summary is None
        assert result.error

    @pytest.mark.asyncio
    async def test_unknown_dimension_skipped(self, deps, fake) -> None:
        """未知维度跳过，不污染结构（宁可少一组）。"""
        self._script(
            fake,
            by_dimension=[
                {"dimension": "not_a_real_dimension", "facts": ["x"], "emotions": []},
                {"dimension": "value", "facts": ["重视稳定"], "emotions": []},
            ],
            preserved_quotes=[QUOTE_A],
        )
        pairs = [_pair(1, ANSWER_A)]
        result = await deps.summarizer.summarize(pairs)
        assert result.ok
        assert {d.dimension for d in result.summary.by_dimension} == {Dimension.VALUE}

    @pytest.mark.asyncio
    async def test_empty_dimension_entries_skipped(self, deps, fake) -> None:
        """facts 和 emotions 都空的维度条目不保留。"""
        self._script(
            fake,
            by_dimension=[{"dimension": "value", "facts": [], "emotions": []}],
            preserved_quotes=[QUOTE_A],
        )
        pairs = [_pair(1, ANSWER_A)]
        result = await deps.summarizer.summarize(pairs)
        assert result.ok
        assert result.summary.by_dimension == []

    @pytest.mark.asyncio
    async def test_summary_records_approx_tokens(self, deps, fake) -> None:
        self._script(
            fake,
            by_dimension=[{"dimension": "value", "facts": ["重视稳定"], "emotions": []}],
            preserved_quotes=[QUOTE_A],
        )
        pairs = [_pair(1, ANSWER_A)]
        result = await deps.summarizer.summarize(pairs)
        assert result.ok
        assert result.summary.approx_tokens > 0

    @pytest.mark.asyncio
    async def test_all_empty_output_rejected(self, deps, fake) -> None:
        """模型没产出任何可用内容 → 摘要为空，拒绝。"""
        self._script(fake, by_dimension=[], preserved_quotes=[])
        pairs = [_pair(1, ANSWER_A)]
        result = await deps.summarizer.summarize(pairs)
        assert not result.ok
        assert "摘要为空" in result.error


class TestSummarizeResult:
    def test_pass_rate(self) -> None:
        r = SummarizeResult(summary=None, quotes_kept=1, quotes_dropped=1)
        assert r.quote_pass_rate == 0.5

    def test_pass_rate_zero_total(self) -> None:
        r = SummarizeResult(summary=None, quotes_kept=0, quotes_dropped=0)
        assert r.quote_pass_rate == 1.0

    def test_min_pass_rate_is_half(self) -> None:
        """记录阈值约定：低于 50% 拒绝整份摘要。改这个值要同步改 summarize 里的判断。"""
        assert MIN_QUOTE_PASS_RATE == 0.5
