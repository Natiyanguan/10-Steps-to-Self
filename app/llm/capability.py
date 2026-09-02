"""模型能力分级与门控（路线文档 §2.4 —— v3.0 核心新增）。

存在理由（§2.4.3 的注解）：
  §9.5 反复强调"静默降级最危险"，而模型可切换恰恰是静默降级的最大入口 ——
  换个 endpoint 就能跑通，但输出从"有依据的洞察"退化成"通顺的空话"，
  接口依然返回 200。能力门控是把这种退化变成显式失败的唯一手段。

三条不可绕过的规则：
  1. 档位不足 → 拒绝路由（抛 InsufficientCapabilityError），不是"先试试看"。
  2. TOY 档永远不允许承担 STRONG 任务 —— **无条件断言，不读配置**。
  3. 要求 json_mode 而 provider 不支持 → 同样视为能力不足。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum

from app.llm.base import InsufficientCapabilityError, TaskKind


class ModelTier(IntEnum):
    """档位。用 IntEnum 以支持 `>=` 比较（§2.4.3 第 1 条）。"""

    TOY = 1     # 仅开发冒烟与 CI：能返回合法响应，内容质量不作保证
    FAST = 2    # 中等能力、低延迟低成本：Safety/Quality/Summary/改写/终审
    STRONG = 3  # 长上下文 + 强指令遵循 + 稳定 JSON：Planner/Interviewer/Analyst/Critic

    @classmethod
    def parse(cls, value: str | int | ModelTier) -> ModelTier:
        if isinstance(value, ModelTier):
            return value
        if isinstance(value, int):
            return cls(value)
        key = value.strip().upper()
        try:
            return cls[key]
        except KeyError as exc:
            raise ValueError(f"未知档位 {value!r}，应为 toy/fast/strong") from exc


class ZhQuality(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# §2.4.1 任务对模型的真实要求
@dataclass(frozen=True, slots=True)
class TaskRequirement:
    task: TaskKind
    min_tier: ModelTier
    needs_json: bool
    min_context: int          # 该任务所需的最小上下文窗口
    min_output: int           # 所需的最小输出额度
    note: str = ""


_REQUIREMENTS: tuple[TaskRequirement, ...] = (
    TaskRequirement(TaskKind.PLAN, ModelTier.STRONG, True, 8_000, 1_200,
                    "多约束同时满足 + 嵌套 JSON"),
    TaskRequirement(TaskKind.ASK, ModelTier.STRONG, True, 4_000, 400,
                    "问题措辞就是产品本体，中文表达要求极高"),
    TaskRequirement(TaskKind.PROBE, ModelTier.STRONG, True, 4_000, 400),
    TaskRequirement(TaskKind.ANALYZE, ModelTier.STRONG, True, 16_000, 1_500,
                    "全量问答 + 原文引用；分段生成后单段额度已降低（§6.5）"),
    TaskRequirement(TaskKind.CRITIQUE, ModelTier.STRONG, True, 8_000, 800),
    TaskRequirement(TaskKind.SAFETY, ModelTier.FAST, True, 2_000, 200,
                    "低难度但绝不允许 TOY 承担（§2.5）"),
    TaskRequirement(TaskKind.QUALITY, ModelTier.FAST, True, 2_000, 200),
    TaskRequirement(TaskKind.SUMMARIZE, ModelTier.FAST, True, 8_000, 800,
                    "须含 preserved_quotes 原文回填"),
    TaskRequirement(TaskKind.QUERY_REWRITE, ModelTier.FAST, False, 1_000, 200),
    TaskRequirement(TaskKind.RERANK_JUDGE, ModelTier.FAST, True, 2_000, 100),
)

REQUIREMENT_BY_TASK: dict[TaskKind, TaskRequirement] = {r.task: r for r in _REQUIREMENTS}

# 需要 STRONG 的任务集合 —— 用于第 2 条硬断言
STRONG_TASKS: frozenset[TaskKind] = frozenset(
    r.task for r in _REQUIREMENTS if r.min_tier is ModelTier.STRONG
)

# 即使 provider 声明 FAST，这些任务也不允许交给 TOY（§2.5 表格）
SAFETY_CRITICAL_TASKS: frozenset[TaskKind] = frozenset({TaskKind.SAFETY})


@dataclass(frozen=True, slots=True)
class ProviderCaps:
    """provider 的实测能力（§2.4.2）。

    `probed_at` 为 None 表示**未经探测**，此时所有 unknown 项按最保守值处理
    （§17.4：绝不乐观假设 —— 乐观假设会在生产以解析失败的形式暴露）。
    """

    tier: ModelTier
    context_window: int
    max_output_tokens: int
    supports_json_mode: bool = False
    supports_function_calling: bool = False
    supports_temperature: bool = True
    supports_streaming: bool = True
    zh_quality: ZhQuality = ZhQuality.MEDIUM
    tokenizer_hint: str = "cl100k_base"
    cost_per_1k_in: float | None = None
    cost_per_1k_out: float | None = None
    probed_at: datetime | None = None

    @property
    def is_probed(self) -> bool:
        return self.probed_at is not None


@dataclass(frozen=True, slots=True)
class GateResult:
    ok: bool
    reason: str = ""

    def raise_if_blocked(self, *, task: TaskKind, provider: str) -> None:
        if not self.ok:
            raise InsufficientCapabilityError(
                f"能力门控拒绝路由：task={task.value} provider={provider} —— {self.reason}"
            )


def check(task: TaskKind, caps: ProviderCaps, *, provider_name: str = "?") -> GateResult:
    """校验 provider 能否承担该任务。纯函数，可完全离线单测。"""
    req = REQUIREMENT_BY_TASK.get(task)
    if req is None:  # pragma: no cover - 新增任务未登记时的防御
        return GateResult(False, f"任务 {task.value} 未登记能力要求，拒绝路由")

    # 规则 2：无条件断言，不读配置、不可覆盖（§2.4.3 第 3 条）
    if caps.tier is ModelTier.TOY and task in STRONG_TASKS:
        return GateResult(
            False,
            f"TOY 档不得承担 STRONG 任务（{task.value}）："
            "小模型会产出格式完整但内容失效的结果，用户无法分辨",
        )
    if caps.tier is ModelTier.TOY and task in SAFETY_CRITICAL_TASKS:
        return GateResult(
            False,
            "TOY 档不得承担安全判级：安全不能建立在能力不足的模型上（§2.5）",
        )

    # 规则 1：档位比较
    if caps.tier < req.min_tier:
        return GateResult(
            False,
            f"档位不足：需要 {req.min_tier.name}，provider={provider_name} 为 {caps.tier.name}",
        )

    # 规则 4：JSON 模式要求
    if req.needs_json and not caps.supports_json_mode:
        return GateResult(
            False,
            f"任务 {task.value} 需要结构化输出，但 provider={provider_name} "
            "未声明支持 json_mode（未探测时按不支持处理，§17.4）",
        )

    # 上下文与输出额度
    if caps.context_window < req.min_context:
        return GateResult(
            False,
            f"上下文窗口不足：需要 ≥{req.min_context}，实际 {caps.context_window}",
        )
    if caps.max_output_tokens < req.min_output:
        return GateResult(
            False,
            f"输出额度不足：需要 ≥{req.min_output}，实际 {caps.max_output_tokens}"
            "（可考虑进一步分段，§6.5）",
        )

    # 中文表达要求：提问类任务的措辞就是产品本体
    if task in (TaskKind.ASK, TaskKind.PROBE) and caps.zh_quality is ZhQuality.LOW:
        return GateResult(
            False,
            "提问类任务要求中文表达质量 ≥ medium：措辞就是产品本体（§2.4.1）",
        )

    return GateResult(True)


def assert_never_toy_for_strong(tier: ModelTier, task: TaskKind) -> None:
    """独立断言入口，供 provider 注册时提前拦截配置错误。"""
    if tier is ModelTier.TOY and task in STRONG_TASKS:
        raise InsufficientCapabilityError(
            f"配置错误：不允许把 TOY 档 provider 注册给 STRONG 任务 {task.value}（§2.4.3）"
        )


def describe_requirements() -> str:
    """人类可读的能力要求表，供 CLI `tss doctor` 与探测报告使用。"""
    lines = [f"{'任务':<14}{'最低档':<8}{'JSON':<6}{'上下文':>8}{'输出':>7}  说明"]
    for r in _REQUIREMENTS:
        lines.append(
            f"{r.task.value:<14}{r.min_tier.name:<8}"
            f"{'必需' if r.needs_json else '可选':<6}"
            f"{r.min_context:>8}{r.min_output:>7}  {r.note}"
        )
    return "\n".join(lines)
