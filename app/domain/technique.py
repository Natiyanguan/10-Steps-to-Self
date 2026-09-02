"""提问技术库（路线文档 §1.3）。

把焦点解决短期治疗等技术代码化为枚举 + 规格 + 纯函数，
而不是散落在提示词里的自然语言。

关键设计：`crisis_safe` 与 `min_sharpness` 让"什么时候不能用这个技术"
成为可程序化过滤的约束（§3.4 处置矩阵在提问侧的延伸），
而不是依赖提示词里的一句叮嘱。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.domain.dimension import Dimension


class Technique(StrEnum):
    MIRACLE = "miracle"            # 奇迹问句
    EXCEPTION = "exception"        # 例外问句
    SCALING = "scaling"            # 量尺问句
    CIRCULAR = "circular"          # 循环问句
    COST_OF_LOSS = "cost_of_loss"  # 代价问句
    REFRAME = "reframe"            # 重构问句
    VALUE_PROBE = "value_probe"    # 价值观追问
    TIME_SHIFT = "time_shift"      # 时间线问句


@dataclass(frozen=True, slots=True)
class TechniqueSpec:
    technique: Technique
    label: str
    pattern: str          # 提问范式（给 Interviewer 的参考，不是最终文本）
    intent_hint: str      # 这个技术意在挖什么
    dimensions: frozenset[Dimension]
    crisis_safe: bool     # risk_level >= medium 时是否仍可使用
    min_sharpness: int    # 需要的最低锐度档（1=GENTLE 2=MODERATE 3=SHARP）
    contraindication: str | None = None


_SPECS: tuple[TechniqueSpec, ...] = (
    TechniqueSpec(
        technique=Technique.MIRACLE,
        label="奇迹问句",
        pattern="假设明天醒来这个困扰已经解决了，你最先注意到的不同是什么？",
        intent_hint="绕过问题本身，让来访者描述真正想要的状态",
        dimensions=frozenset({Dimension.VALUE, Dimension.RESOURCE, Dimension.IDENTITY}),
        crisis_safe=True,
        min_sharpness=1,
    ),
    TechniqueSpec(
        technique=Technique.EXCEPTION,
        label="例外问句",
        pattern="有没有哪段时间这个困扰没那么强烈？那时候有什么不一样？",
        intent_hint="找出已经存在的资源与有效经验",
        dimensions=frozenset({Dimension.FACT, Dimension.RESOURCE}),
        crisis_safe=True,
        min_sharpness=1,
    ),
    TechniqueSpec(
        technique=Technique.SCALING,
        label="量尺问句",
        pattern="0 到 10 分，现在大概几分？往上挪 1 分会发生什么？",
        intent_hint="把模糊感受变成可讨论的刻度，并找出最小可行的一步",
        # 含 COST：量尺是唯一能温和触及"代价"的技术。
        # 「满分10分是你能接受的最大代价，这个决定大概要几分？」
        # 只要求校准，不迫使对方正面设想丧失 —— 因此 crisis_safe。
        dimensions=frozenset(
            {Dimension.EMOTION, Dimension.RESOURCE, Dimension.FACT, Dimension.COST}
        ),
        crisis_safe=True,
        min_sharpness=1,
    ),
    TechniqueSpec(
        technique=Technique.CIRCULAR,
        label="循环问句",
        pattern="如果最了解你的人来回答这个问题，他会怎么说？",
        intent_hint="借他人视角触及乔哈里视窗的盲区",
        dimensions=frozenset({Dimension.RELATION, Dimension.IDENTITY}),
        crisis_safe=True,
        # 1 而非 2：借第三人视角是 SFBT 对脆弱来访者的常规做法，
        # 它把压力移到"别人会怎么说"，反而比直接提问更温和。
        min_sharpness=1,
    ),
    TechniqueSpec(
        technique=Technique.COST_OF_LOSS,
        label="代价问句",
        pattern="做这个决定，你最怕失去的是什么？",
        intent_hint="直击恐惧与真实优先级",
        dimensions=frozenset({Dimension.FEAR, Dimension.COST}),
        crisis_safe=False,
        min_sharpness=2,
        contraindication="危机或高情绪强度时不宜使用：会放大丧失感",
    ),
    TechniqueSpec(
        technique=Technique.REFRAME,
        label="重构问句",
        pattern="换个角度看，现在的犹豫可能在保护你什么？",
        intent_hint="让来访者看见症状的功能性意义",
        dimensions=frozenset({Dimension.VALUE, Dimension.FEAR, Dimension.IDENTITY}),
        crisis_safe=True,
        # 1 而非 2：重构是"你的犹豫在保护你"这类肯定性移动，
        # 是全表最温和的技术之一。设为 2 会让 FEAR 维度在温和档无技术可用，
        # 导致规划无解 —— 见 test_every_dimension_has_technique_at_gentle。
        min_sharpness=1,
    ),
    TechniqueSpec(
        technique=Technique.VALUE_PROBE,
        label="价值观追问",
        pattern="这件事之所以让你放不下，是因为你在意什么？",
        intent_hint="从表层困惑往下挖到价值层",
        dimensions=frozenset({Dimension.VALUE, Dimension.IDENTITY}),
        crisis_safe=True,
        min_sharpness=1,
    ),
    TechniqueSpec(
        technique=Technique.TIME_SHIFT,
        label="时间线问句",
        pattern="五年后的你回看现在这个犹豫，会怎么评价？",
        intent_hint="拉开时间距离，降低当下情绪的支配力",
        dimensions=frozenset({Dimension.VALUE, Dimension.COST, Dimension.IDENTITY}),
        crisis_safe=False,
        min_sharpness=3,
        contraindication="危机状态下谈长远未来可能加重无望感",
    ),
)

SPEC_BY_TECHNIQUE: dict[Technique, TechniqueSpec] = {s.technique: s for s in _SPECS}


def spec(technique: Technique) -> TechniqueSpec:
    return SPEC_BY_TECHNIQUE[technique]


def all_specs() -> tuple[TechniqueSpec, ...]:
    return _SPECS


def techniques_for_dimension(dimension: Dimension) -> tuple[Technique, ...]:
    return tuple(s.technique for s in _SPECS if dimension in s.dimensions)


def allowed_techniques(
    *,
    dimension: Dimension,
    sharpness: int,
    crisis_mode: bool = False,
) -> tuple[Technique, ...]:
    """给定维度与锐度，返回可用技术。

    `crisis_mode=True`（risk_level >= medium）时过滤 `crisis_safe=False` 的技术。
    硬过滤，不经 LLM —— 安全约束不能交给模型自觉。
    """
    out: list[Technique] = []
    for s in _SPECS:
        if dimension not in s.dimensions:
            continue
        if crisis_mode and not s.crisis_safe:
            continue
        if s.min_sharpness > sharpness:
            continue
        out.append(s.technique)
    return tuple(out)


def is_compatible(
    technique: Technique,
    dimension: Dimension,
    *,
    sharpness: int,
    crisis_mode: bool = False,
) -> bool:
    return technique in allowed_techniques(
        dimension=dimension, sharpness=sharpness, crisis_mode=crisis_mode
    )


def technique_balance(used: list[Technique]) -> float:
    """技术分布均衡度（0-1，越高越均衡）。

    §11.3 质量指标：防止 10 问全用同一种技术。
    单一技术占满 → 0.0；完全不重复 → 1.0。
    """
    if not used:
        return 0.0
    largest = max(used.count(t) for t in set(used))
    return 1.0 - (largest - 1) / len(used)
