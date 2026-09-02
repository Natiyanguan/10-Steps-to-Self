"""提问维度矩阵（路线文档 §1.4）。

纯数据 + 纯函数，零 LLM 依赖、零 IO。所有校验必须可离线单测。
"""

from __future__ import annotations

from enum import StrEnum

TOTAL_QUESTIONS = 10


class Dimension(StrEnum):
    """10 问在这些维度上做覆盖规划，而非随机生成。"""

    FACT = "fact"          # 事实层：发生了什么
    EMOTION = "emotion"    # 情绪层：什么感受
    VALUE = "value"        # 价值层：在意什么
    FEAR = "fear"          # 恐惧层：怕失去什么
    COST = "cost"          # 代价层：愿意付什么
    RESOURCE = "resource"  # 资源层：手上有什么
    RELATION = "relation"  # 关系层：谁在其中
    IDENTITY = "identity"  # 身份层：想成为谁


# §1.4 三段式：建立安全感 → 核心挖掘 → 收敛到行动。
# 用"允许集合"而非固定序列，给 Planner 留出适配具体困惑的空间。
_OPENING = frozenset({Dimension.FACT, Dimension.EMOTION, Dimension.RELATION})
_CORE = frozenset({Dimension.VALUE, Dimension.FEAR, Dimension.COST, Dimension.IDENTITY})
_CLOSING = frozenset({Dimension.RESOURCE, Dimension.VALUE, Dimension.IDENTITY})

OPENING_RANGE = range(1, 4)   # 第 1-3 问
CORE_RANGE = range(4, 8)      # 第 4-7 问
CLOSING_RANGE = range(8, 11)  # 第 8-10 问

# 深度维度：未成年人或高风险时必须关闭（§3.2 / §3.4）
DEEP_DIMENSIONS = frozenset({Dimension.FEAR, Dimension.IDENTITY})

MIN_DISTINCT_DIMENSIONS = 5


def allowed_dimensions(index: int) -> frozenset[Dimension]:
    """返回第 index 问允许的维度集合。index 从 1 开始。"""
    if index in OPENING_RANGE:
        return _OPENING
    if index in CORE_RANGE:
        return _CORE
    if index in CLOSING_RANGE:
        return _CLOSING
    raise ValueError(f"问题序号必须在 1..{TOTAL_QUESTIONS}，收到 {index}")


def band_name(index: int) -> str:
    if index in OPENING_RANGE:
        return "opening"
    if index in CORE_RANGE:
        return "core"
    if index in CLOSING_RANGE:
        return "closing"
    raise ValueError(f"问题序号越界：{index}")


def validate_coverage(dimensions: list[Dimension], *, complete: bool = True) -> list[str]:
    """校验维度序列，返回错误列表（空 = 通过）。

    规则（§1.4）：
      1. 长度必须是 TOTAL_QUESTIONS（仅 `complete=True` 时）
      2. 每问维度须落在该段允许集合内
      3. 任一维度不得连续两问重复（重复即规划失败，触发重规划）
      4. 至少覆盖 MIN_DISTINCT_DIMENSIONS 个不同维度（仅 `complete=True` 时）

    `complete=False` 用于 BATCHED / PROGRESSIVE 模式的**分批规划中间态**：
    此时只规划了前几问，长度与总覆盖度都还不该被要求满足，
    但**逐问维度合法性与相邻不重复必须立即校验** ——
    否则错误会累积到最后一批才暴露，而那时前面的问题已经发给用户了。
    """
    errors: list[str] = []

    if len(dimensions) > TOTAL_QUESTIONS:
        return [f"最多规划 {TOTAL_QUESTIONS} 问，实际 {len(dimensions)} 问"]

    if complete and len(dimensions) != TOTAL_QUESTIONS:
        errors.append(f"必须规划 {TOTAL_QUESTIONS} 问，实际 {len(dimensions)} 问")
        return errors  # 长度不对，后续校验无意义

    for i, dim in enumerate(dimensions, start=1):
        allowed = allowed_dimensions(i)
        if dim not in allowed:
            names = "/".join(sorted(d.value for d in allowed))
            errors.append(f"第 {i} 问（{band_name(i)}）维度 {dim.value} 不合法，应属 {names}")

    for i in range(1, len(dimensions)):
        if dimensions[i] == dimensions[i - 1]:
            errors.append(
                f"第 {i} 与第 {i + 1} 问维度重复（{dimensions[i].value}），规划失败需重规划"
            )

    if complete:
        distinct = len(set(dimensions))
        if distinct < MIN_DISTINCT_DIMENSIONS:
            errors.append(
                f"仅覆盖 {distinct} 个维度，至少需 {MIN_DISTINCT_DIMENSIONS} 个（§1.2 盲区要求）"
            )

    return errors


def coverage_ratio(dimensions: list[Dimension]) -> float:
    """维度覆盖率：零成本可程序化计算的质量指标（§7.5-C / §11.3）。"""
    if not dimensions:
        return 0.0
    return len(set(dimensions)) / len(Dimension)


def filter_for_safety(
    candidates: frozenset[Dimension], *, block_deep: bool
) -> frozenset[Dimension]:
    """高风险或未成年人时移除深度维度（§3.2 / §3.4 的硬约束）。"""
    if not block_deep:
        return candidates
    remaining = candidates - DEEP_DIMENSIONS
    # 全被过滤掉时退回情绪层：宁可问得浅，也不能无维度可问。
    return remaining or frozenset({Dimension.EMOTION})
