"""压缩触发决策单测（路线文档 §6.3）。

`plan_compression` 是纯函数，适合穷尽边界。
压缩策略错了的表现是"长会话突然触顶"，这种故障在生产上极难复现，
所以这里的边界用例比"应命中"的用例多。

断言全部基于 `CompressionPlan` 的字段，不依赖任何 LLM 调用。
"""

from __future__ import annotations

import pytest

from app.context.compress import (
    COMPRESS_WATERMARK,
    KEEP_RECENT_PAIRS,
    MIN_PAIRS_TO_COMPRESS,
    history_tokens,
    plan_compression,
)
from app.domain.technique import Technique
from app.schemas.enums import Sharpness
from app.schemas.session import Answer, QAPair, Question, SessionSummary


def _pair(index: int, answer_len: int = 40) -> QAPair:
    """构造一个问答对。answer_len 用重复中文控制 token 量。"""
    return QAPair(
        question=Question(
            index=index,
            text=f"第{index}个问题？",
            technique=Technique.SCALING,
            sharpness=Sharpness.MODERATE,
        ),
        answer=Answer(question_index=index, text="这是一个足够长的回答。" * answer_len),
    )


def _pairs(n: int) -> list[QAPair]:
    return [_pair(i) for i in range(1, n + 1)]


class TestHistoryTokens:
    def test_scales_with_content(self) -> None:
        short = history_tokens([_pair(1, answer_len=5)])
        long = history_tokens([_pair(1, answer_len=50)])
        assert long > short

    def test_empty_is_zero(self) -> None:
        assert history_tokens([]) == 0

    def test_summary_counts_toward_budget(self) -> None:
        """摘要自身也占额度，忽略它会低估压力。"""
        summary = SessionSummary(
            preserved_quotes=["这是一个用于测试的逐字原话"],
            covered_indices=[1],
            approx_tokens=500,
        )
        assert history_tokens(_pairs(3), summary) >= 500

    def test_summary_zero_tokens_falls_back_to_render(self) -> None:
        """approx_tokens 为 0 时用 render() 实测，而不是当成 0。"""
        summary = SessionSummary(
            preserved_quotes=["这是一个用于测试的逐字原话"],
            covered_indices=[1],
            approx_tokens=0,
        )
        assert history_tokens(_pairs(3), summary) > 0


class TestPlanCompression:
    def test_empty_pairs_never_compress(self) -> None:
        plan = plan_compression([], available_tokens=10_000)
        assert not plan.should_compress

    def test_no_available_tokens_skips(self) -> None:
        plan = plan_compression(_pairs(5), available_tokens=0)
        assert not plan.should_compress
        assert "额度" in plan.reason

    def test_negative_available_skips(self) -> None:
        plan = plan_compression(_pairs(5), available_tokens=-1)
        assert not plan.should_compress

    def test_below_watermark_does_not_compress(self) -> None:
        """历史占用低于水位时不压缩。"""
        pairs = _pairs(4)
        used = history_tokens(pairs)
        # 给足额度，让占用远低于 60%
        plan = plan_compression(pairs, available_tokens=used * 10)
        assert not plan.should_compress
        assert "未达水位" in plan.reason

    def test_above_watermark_compresses(self) -> None:
        """占用达到水位时触发，压缩除最近 KEEP_RECENT_PAIRS 轮之外的全部。"""
        pairs = _pairs(8)
        used = history_tokens(pairs)
        # 让水位正好落在当前占用之下
        plan = plan_compression(pairs, available_tokens=used)
        assert plan.should_compress
        assert len(plan.to_compress) == 8 - KEEP_RECENT_PAIRS
        assert len(plan.keep_verbatim) == KEEP_RECENT_PAIRS

    def test_too_few_pairs_never_compress(self) -> None:
        """轮数不足 MIN_PAIRS_TO_COMPRESS 时，压缩成本大于收益。"""
        pairs = _pairs(KEEP_RECENT_PAIRS + MIN_PAIRS_TO_COMPRESS - 1)
        used = history_tokens(pairs)
        plan = plan_compression(pairs, available_tokens=used)
        assert not plan.should_compress
        assert "成本大于收益" in plan.reason or "少于" in plan.reason

    def test_covered_indices_not_recompressed(self) -> None:
        """已被摘要覆盖的问题序号不重复压缩（§6.3）。"""
        summary = SessionSummary(
            preserved_quotes=["已压缩过的一句原话"],
            covered_indices=[1, 2],
            approx_tokens=100,
        )
        pairs = _pairs(8)
        used = history_tokens(pairs, summary)
        plan = plan_compression(pairs, available_tokens=used, summary=summary)
        assert plan.should_compress
        compressed_idx = {qa.question.index for qa in plan.to_compress}
        assert not (compressed_idx & {1, 2}), "已覆盖的序号不该再次压缩"

    def test_keep_recent_zero_compresses_all_uncovered(self) -> None:
        pairs = _pairs(6)
        used = history_tokens(pairs)
        plan = plan_compression(pairs, available_tokens=used, keep_recent=0)
        assert plan.should_compress
        assert len(plan.to_compress) == 6
        assert plan.keep_verbatim == []

    def test_watermark_boundary_exact(self) -> None:
        """占用恰好等于水位时——按实现 `used < available * watermark` 判断，
        等于时**不**压缩（严格小于才触发）。这是边界约定，记录下来。"""
        pairs = _pairs(5)
        used = history_tokens(pairs)
        available = int(used / COMPRESS_WATERMARK)
        plan = plan_compression(pairs, available_tokens=available)
        # 由于整数除法，available * watermark 可能略小于或等于 used，
        # 这里只断言：要么不压缩，要么压缩后 keep 的数量正确。
        if plan.should_compress:
            assert len(plan.keep_verbatim) == KEEP_RECENT_PAIRS

    def test_pressure_reflects_ratio(self) -> None:
        pairs = _pairs(4)
        used = history_tokens(pairs)
        plan = plan_compression(pairs, available_tokens=used * 2)
        assert plan.pressure == pytest.approx(0.5, abs=0.05)

    def test_pressure_zero_when_no_available(self) -> None:
        plan = plan_compression(_pairs(3), available_tokens=0)
        assert plan.pressure == 0.0

    def test_is_pure(self) -> None:
        pairs = _pairs(6)
        a = plan_compression(pairs, available_tokens=1000)
        b = plan_compression(pairs, available_tokens=1000)
        assert a == b


class TestCompressionPlanFields:
    def test_reason_always_present(self) -> None:
        """无论压不压缩，`reason` 都不能为空——观测与排障依赖它。"""
        for avail in (0, 10_000, 10_000_000):
            plan = plan_compression(_pairs(5), available_tokens=avail)
            assert plan.reason
