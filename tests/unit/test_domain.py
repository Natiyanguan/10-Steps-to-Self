"""domain 层单测（路线文档 §12：零 LLM 依赖，100% 覆盖）。

这些是纯函数，测试必须离线、确定、瞬时完成。
"""

from __future__ import annotations

import pytest

from app.domain.dimension import (
    DEEP_DIMENSIONS,
    TOTAL_QUESTIONS,
    Dimension,
    allowed_dimensions,
    band_name,
    coverage_ratio,
    filter_for_safety,
    validate_coverage,
)
from app.domain.mode import (
    AskMode,
    allows_dependencies,
    allows_replan,
    batch_bounds,
    batch_index_for,
    batch_sizes,
    batch_slice,
    is_batch_end,
)
from app.domain.technique import (
    Technique,
    all_specs,
    allowed_techniques,
    is_compatible,
    spec,
    technique_balance,
    techniques_for_dimension,
)
from tests.conftest import GENTLE_PLAN, VALID_PLAN


class TestDimension:
    def test_valid_plan_passes(self) -> None:
        assert validate_coverage([d for d, _ in VALID_PLAN]) == []

    def test_gentle_plan_passes(self) -> None:
        assert validate_coverage([d for d, _ in GENTLE_PLAN]) == []

    def test_wrong_length_rejected(self) -> None:
        errors = validate_coverage([Dimension.FACT] * 5)
        assert errors and "必须规划 10 问" in errors[0]

    def test_partial_plan_allowed_when_incomplete(self) -> None:
        """分批规划中间态：长度不足不算错（§1.5 BATCHED）。"""
        dims = [d for d, _ in VALID_PLAN][:3]
        assert validate_coverage(dims, complete=False) == []

    def test_partial_plan_still_checks_adjacency(self) -> None:
        """中间态仍须校验相邻不重复 —— 错误越早暴露越好。"""
        errors = validate_coverage(
            [Dimension.FACT, Dimension.FACT, Dimension.RELATION], complete=False
        )
        assert any("重复" in e for e in errors)

    def test_consecutive_repeat_rejected(self) -> None:
        dims = [d for d, _ in VALID_PLAN]
        dims[4] = dims[3]  # 第 4、5 问同维度
        errors = validate_coverage(dims)
        assert any("重复" in e for e in errors)

    def test_wrong_band_rejected(self) -> None:
        dims = [d for d, _ in VALID_PLAN]
        dims[0] = Dimension.FEAR  # 第 1 问不该是恐惧层
        errors = validate_coverage(dims)
        assert any("不合法" in e for e in errors)

    def test_too_few_distinct_rejected(self) -> None:
        """段约束 + 相邻不重复下的理论最小覆盖是 4 个维度，应被拒绝。

        opening 集合与 core 集合互不相交，各自至少贡献 2 个维度；
        closing 可以完全复用 core 的维度。所以 4 是能构造出的下限。
        """
        dims = [
            Dimension.FACT, Dimension.EMOTION, Dimension.FACT,
            Dimension.VALUE, Dimension.IDENTITY, Dimension.VALUE, Dimension.IDENTITY,
            Dimension.VALUE, Dimension.IDENTITY, Dimension.VALUE,
        ]
        assert len(set(dims)) == 4
        errors = validate_coverage(dims)
        assert any("至少需" in e for e in errors)

    def test_over_length_rejected(self) -> None:
        errors = validate_coverage([Dimension.FACT] * 11)
        assert errors and "最多规划" in errors[0]

    @pytest.mark.parametrize(
        ("index", "band"),
        [(1, "opening"), (3, "opening"), (4, "core"), (7, "core"), (8, "closing"), (10, "closing")],
    )
    def test_band_boundaries(self, index: int, band: str) -> None:
        assert band_name(index) == band

    @pytest.mark.parametrize("index", [0, 11, -1, 99])
    def test_out_of_range_raises(self, index: int) -> None:
        with pytest.raises(ValueError, match="序号"):
            allowed_dimensions(index)
        with pytest.raises(ValueError):
            band_name(index)

    def test_coverage_ratio(self) -> None:
        assert coverage_ratio([]) == 0.0
        assert coverage_ratio([d for d, _ in VALID_PLAN]) == 1.0
        assert coverage_ratio([Dimension.FACT] * 4) == 1 / len(Dimension)

    def test_filter_for_safety_removes_deep(self) -> None:
        """未成年人/高风险时禁用深度维度（§3.2 / §3.4）。"""
        core = allowed_dimensions(5)
        filtered = filter_for_safety(core, block_deep=True)
        assert not (filtered & DEEP_DIMENSIONS)
        assert filter_for_safety(core, block_deep=False) == core

    def test_filter_never_returns_empty(self) -> None:
        """全被过滤时必须有兜底 —— 否则无维度可问，流程会卡死。"""
        only_deep = frozenset({Dimension.FEAR, Dimension.IDENTITY})
        assert filter_for_safety(only_deep, block_deep=True)


class TestMode:
    @pytest.mark.parametrize("mode", list(AskMode))
    def test_batch_sizes_sum_to_total(self, mode: AskMode) -> None:
        assert sum(batch_sizes(mode)) == TOTAL_QUESTIONS

    def test_batched_is_3_4_3(self) -> None:
        assert batch_sizes(AskMode.BATCHED) == (3, 4, 3)
        assert batch_bounds(AskMode.BATCHED) == ((1, 3), (4, 7), (8, 10))

    def test_one_shot_single_batch(self) -> None:
        assert batch_bounds(AskMode.ONE_SHOT) == ((1, 10),)

    def test_progressive_ten_batches(self) -> None:
        assert len(batch_bounds(AskMode.PROGRESSIVE)) == TOTAL_QUESTIONS

    @pytest.mark.parametrize(
        ("index", "expected"), [(1, 0), (3, 0), (4, 1), (7, 1), (8, 2), (10, 2)]
    )
    def test_batch_index_for(self, index: int, expected: int) -> None:
        assert batch_index_for(AskMode.BATCHED, index) == expected

    def test_batch_index_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="越界"):
            batch_index_for(AskMode.BATCHED, 11)

    def test_batch_slice_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="越界"):
            batch_slice(AskMode.BATCHED, 9)

    @pytest.mark.parametrize(
        ("index", "expected"), [(3, True), (7, True), (10, True), (1, False), (5, False)]
    )
    def test_is_batch_end(self, index: int, expected: bool) -> None:
        assert is_batch_end(AskMode.BATCHED, index) is expected

    def test_one_shot_forbids_dependencies(self) -> None:
        """ONE_SHOT 无法基于回答调整（§1.5）。"""
        assert allows_dependencies(AskMode.ONE_SHOT) is False
        assert allows_dependencies(AskMode.BATCHED) is True
        assert allows_dependencies(AskMode.PROGRESSIVE) is True

    def test_one_shot_forbids_replan(self) -> None:
        assert allows_replan(AskMode.ONE_SHOT) is False
        assert allows_replan(AskMode.BATCHED) is True


class TestTechnique:
    def test_all_specs_registered(self) -> None:
        assert len(all_specs()) == len(Technique)
        for t in Technique:
            assert spec(t).technique is t

    def test_crisis_unsafe_filtered_in_crisis_mode(self) -> None:
        """危机模式下 crisis_safe=False 的技术必须被硬过滤（§1.3 / §3.4）。"""
        unsafe = [s.technique for s in all_specs() if not s.crisis_safe]
        assert unsafe, "应存在危机不安全的技术，否则这条约束没有意义"

        for t in unsafe:
            s = spec(t)
            dim = next(iter(s.dimensions))
            allowed = allowed_techniques(dimension=dim, sharpness=3, crisis_mode=True)
            assert t not in allowed, f"{t.value} 在危机模式下不应可用"

    def test_sharpness_gate(self) -> None:
        """锐度不足时高锐度技术不可用（§3.5）。"""
        s = spec(Technique.TIME_SHIFT)
        assert s.min_sharpness == 3
        dim = next(iter(s.dimensions))
        assert Technique.TIME_SHIFT not in allowed_techniques(dimension=dim, sharpness=2)
        assert Technique.TIME_SHIFT in allowed_techniques(dimension=dim, sharpness=3)

    def test_unsafe_techniques_have_contraindication(self) -> None:
        """危机不安全的技术必须写明禁用原因 —— 供 Planner 反馈与知识库对齐（§7.7.3）。"""
        for s in all_specs():
            if not s.crisis_safe:
                assert s.contraindication, f"{s.technique.value} 缺少 contraindication"

    def test_techniques_for_dimension_non_empty(self) -> None:
        for d in Dimension:
            assert techniques_for_dimension(d), f"维度 {d.value} 没有任何可用技术"

    def test_every_dimension_has_gentle_safe_technique(self) -> None:
        """每个维度在 GENTLE + 危机模式下都要有可用技术，否则规划会无解。"""
        for d in Dimension:
            if d in DEEP_DIMENSIONS:
                continue  # 深度维度在危机时本就被禁用
            assert allowed_techniques(dimension=d, sharpness=1, crisis_mode=True), (
                f"维度 {d.value} 在温和+危机模式下无可用技术"
            )

    def test_is_compatible(self) -> None:
        assert is_compatible(Technique.SCALING, Dimension.EMOTION, sharpness=1)
        assert not is_compatible(Technique.SCALING, Dimension.FEAR, sharpness=3)

    def test_technique_balance(self) -> None:
        assert technique_balance([]) == 0.0
        assert technique_balance([Technique.MIRACLE] * 4) == pytest.approx(0.25)
        varied = [Technique.MIRACLE, Technique.SCALING, Technique.EXCEPTION, Technique.CIRCULAR]
        assert technique_balance(varied) == 1.0

    def test_conftest_plans_are_internally_consistent(self) -> None:
        """守卫测试夹具本身：夹具规划必须满足全部约束。

        夹具一旦失效，集成测试会测出"校验器在拒绝"而不是"流程能跑通"，
        那种失败很容易被误读成产品 bug。
        """
        for label, table, sharpness, crisis in (
            ("VALID_PLAN", VALID_PLAN, 2, False),
            ("GENTLE_PLAN", GENTLE_PLAN, 1, True),
        ):
            assert validate_coverage([d for d, _ in table]) == [], f"{label} 维度不合法"
            for i, (d, t) in enumerate(table, start=1):
                assert d in allowed_dimensions(i), f"{label} 第 {i} 问维度越界"
                assert t in allowed_techniques(
                    dimension=d, sharpness=sharpness, crisis_mode=crisis
                ), f"{label} 第 {i} 问技术 {t.value} 与维度 {d.value} 不匹配"
