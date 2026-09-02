"""提问模式（路线文档 §1.5）。

三种模式都必须满足 §1.4 的维度覆盖约束，模式只影响调用节奏。
"""

from __future__ import annotations

from enum import StrEnum

from app.domain.dimension import TOTAL_QUESTIONS


class AskMode(StrEnum):
    ONE_SHOT = "one_shot"        # 一次性生成全部 10 问
    BATCHED = "batched"          # 3 → 4 → 3 分三批（默认）
    PROGRESSIVE = "progressive"  # 每次 1 问，完全动态


DEFAULT_MODE = AskMode.BATCHED

_BATCH_SIZES: dict[AskMode, tuple[int, ...]] = {
    AskMode.ONE_SHOT: (TOTAL_QUESTIONS,),
    AskMode.BATCHED: (3, 4, 3),
    AskMode.PROGRESSIVE: (1,) * TOTAL_QUESTIONS,
}


def batch_sizes(mode: AskMode) -> tuple[int, ...]:
    sizes = _BATCH_SIZES[mode]
    if sum(sizes) != TOTAL_QUESTIONS:  # pragma: no cover - 配置错误的防御
        raise AssertionError(f"{mode} 的分批之和必须等于 {TOTAL_QUESTIONS}")
    return sizes


def batch_bounds(mode: AskMode) -> tuple[tuple[int, int], ...]:
    """每批的 (起始问号, 结束问号)，闭区间、从 1 开始。"""
    bounds: list[tuple[int, int]] = []
    cursor = 1
    for size in batch_sizes(mode):
        bounds.append((cursor, cursor + size - 1))
        cursor += size
    return tuple(bounds)


def batch_index_for(mode: AskMode, question_index: int) -> int:
    for i, (lo, hi) in enumerate(batch_bounds(mode)):
        if lo <= question_index <= hi:
            return i
    raise ValueError(f"问题序号越界：{question_index}")


def batch_slice(mode: AskMode, batch_index: int) -> tuple[int, int]:
    bounds = batch_bounds(mode)
    if not 0 <= batch_index < len(bounds):
        raise ValueError(f"批次序号越界：{batch_index}")
    return bounds[batch_index]


def is_batch_end(mode: AskMode, question_index: int) -> bool:
    """该问是否为某批最后一问 —— BATCHED 下触发剩余槽位重规划（§4.3）。"""
    return any(hi == question_index for _, hi in batch_bounds(mode))


def allows_dependencies(mode: AskMode) -> bool:
    """ONE_SHOT 无法基于回答调整，故 PlanSlot.depends_on 必须为空（§1.5）。"""
    return mode is not AskMode.ONE_SHOT


def allows_replan(mode: AskMode) -> bool:
    """ONE_SHOT 已全部发出，不可重规划；其余模式可对未发出的槽位重规划。"""
    return mode is not AskMode.ONE_SHOT
