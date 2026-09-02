"""Planner Agent（路线文档 §4.4 / §1.4）。

职责边界（铁律 1）：**只规划维度与技术序列，不生成问题文本**。
产物是骨架，渲染交给 Interviewer。

关键设计：LLM 的规划输出必须过 `app/domain` 的纯函数校验。
维度重复、技术与维度不匹配、越出该段允许集合 —— 一律拒绝并重规划。
把方法论约束交给代码而不是提示词，是这套系统能保证"提问有结构"的原因。
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import Agent
from app.domain import dimension as dim_mod
from app.domain.dimension import TOTAL_QUESTIONS, Dimension, allowed_dimensions, band_name
from app.domain.mode import AskMode, allows_dependencies, batch_slice
from app.domain.technique import Technique, all_specs, allowed_techniques, spec
from app.llm.base import TaskKind
from app.llm.gateway import CallOutcome
from app.schemas.enums import Sharpness
from app.schemas.session import PlanSlot, QuestionPlan

MAX_PLAN_ATTEMPTS = 3

_DIMENSION_DESC: dict[Dimension, str] = {
    Dimension.FACT: "发生了什么 —— 客观情况、时间线、已经做过的尝试",
    Dimension.EMOTION: "什么感受 —— 情绪的性质与强度",
    Dimension.VALUE: "在意什么 —— 排序、优先级、什么值得",
    Dimension.FEAR: "怕失去什么 —— 恐惧与最坏设想",
    Dimension.COST: "愿意付什么 —— 可接受的代价与底线",
    Dimension.RESOURCE: "手上有什么 —— 能力、支持系统、已有的例外经验",
    Dimension.RELATION: "谁在其中 —— 关键他人及其影响",
    Dimension.IDENTITY: "想成为谁 —— 自我认同与长期方向",
}


class PlannerSlotOut(BaseModel):
    """LLM 输出的单个槽位。字段名与提示词严格对应。"""

    model_config = ConfigDict(extra="forbid")

    index: int
    dimension: str
    technique: str
    intent: str = Field(min_length=1, max_length=200)
    depends_on: list[int] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slots: list[PlannerSlotOut]
    rationale: str = Field(default="", max_length=600)


@dataclass(frozen=True, slots=True)
class PlanResult:
    slots: list[PlanSlot]
    rationale: str
    attempts: int
    #: 无需调用 LLM 时（该批次已规划完）为 None
    outcome: CallOutcome | None = None


class PlanValidationError(RuntimeError):
    """连续重规划仍不合法 —— 显式失败，不接受不合规的规划。"""


class Planner(Agent):
    prompt_agent = "planner"
    task = TaskKind.PLAN

    async def plan(
        self,
        *,
        concern: str,
        mode: AskMode,
        sharpness: Sharpness,
        batch_index: int = 0,
        existing: list[PlanSlot] | None = None,
        crisis_mode: bool = False,
        block_deep: bool = False,
        profile_notes: list[str] | None = None,
        references: list[str] | None = None,
        references_ambiguous: bool = False,
    ) -> PlanResult:
        """规划一个批次的槽位。

        `existing` 是已规划（或已问过）的槽位，用于校验相邻维度不重复。
        BATCHED 模式下本方法会被调用多次，每次只规划当前批次。
        """
        existing = list(existing or [])
        lo, hi = batch_slice(mode, batch_index)
        # 已规划过的部分不重复规划
        lo = max(lo, len(existing) + 1)
        if lo > hi:
            return PlanResult(slots=[], rationale="", attempts=0, outcome=None)

        context = self._build_context(
            concern=concern,
            mode=mode,
            sharpness=sharpness,
            lo=lo,
            hi=hi,
            crisis_mode=crisis_mode,
            block_deep=block_deep,
            profile_notes=profile_notes or [],
            references=references or [],
            references_ambiguous=references_ambiguous,
        )

        feedback: str | None = None
        last_errors: list[str] = []

        for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
            ctx = dict(context)
            if feedback:
                ctx["retry_feedback"] = feedback
            prompt = self.render(ctx)

            payload = (
                "请开始规划。"
                if not feedback
                else f"上一次规划不合法：{feedback}\n请重新规划。"
            )
            out, outcome = await self.call(prompt, PlannerOutput, user_payload=payload)

            slots, errors = self._coerce_and_validate(
                out,
                lo=lo,
                hi=hi,
                mode=mode,
                sharpness=sharpness,
                existing=existing,
                crisis_mode=crisis_mode,
                block_deep=block_deep,
            )
            if not errors:
                return PlanResult(
                    slots=slots, rationale=out.rationale, attempts=attempt, outcome=outcome
                )

            last_errors = errors
            feedback = "；".join(errors[:5])

        raise PlanValidationError(
            f"连续 {MAX_PLAN_ATTEMPTS} 次规划均不合法：{'；'.join(last_errors)}。\n"
            "  不接受不合规的规划：维度重复或技术错配会让 10 问退化成随机提问（§1.4）。"
        )

    # ---- 提示词上下文 ----

    def _build_context(
        self,
        *,
        concern: str,
        mode: AskMode,
        sharpness: Sharpness,
        lo: int,
        hi: int,
        crisis_mode: bool,
        block_deep: bool,
        profile_notes: list[str],
        references: list[str],
        references_ambiguous: bool,
    ) -> dict[str, object]:
        # 关键：必须给出**维度 → 可用技术**的对应关系，而不是两份独立列表。
        # 只给"允许的维度"+"可用的技术"会让模型组合出非法配对
        # （如 emotion + exception），然后被校验拒绝、触发重规划 ——
        # 每次重规划都是一次真实的 token 开销，且可能连续失败导致会话中断。
        usable_by_dim: dict[Dimension, tuple[Technique, ...]] = {}
        slot_constraints: list[dict[str, object]] = []

        for i in range(lo, hi + 1):
            allowed = dim_mod.filter_for_safety(
                allowed_dimensions(i), block_deep=block_deep or crisis_mode
            )
            viable: list[Dimension] = []
            for d in sorted(allowed, key=lambda x: x.value):
                techs = allowed_techniques(
                    dimension=d, sharpness=int(sharpness), crisis_mode=crisis_mode
                )
                if techs:
                    viable.append(d)
                    usable_by_dim[d] = techs

            if not viable:
                # 该槽位在当前锐度/危机状态下无解。这是配置或技术表的问题，
                # 不该让模型去试 —— 显式失败比反复重规划更容易定位。
                raise PlanValidationError(
                    f"第 {i} 问在锐度={int(sharpness)}、危机模式={crisis_mode} 下"
                    "没有任何可用的维度+技术组合，请检查技术表的 min_sharpness/crisis_safe"
                )

            slot_constraints.append(
                {"index": i, "band": band_name(i), "allowed": [d.value for d in viable]}
            )

        dimension_guide = [
            {
                "dimension": d.value,
                "desc": _DIMENSION_DESC[d],
                "techniques": [t.value for t in techs],
            }
            for d, techs in sorted(usable_by_dim.items(), key=lambda kv: kv[0].value)
        ]

        used_techniques = {t for techs in usable_by_dim.values() for t in techs}
        technique_glossary = [
            {"key": s.technique.value, "label": s.label, "intent": s.intent_hint}
            for s in all_specs()
            if s.technique in used_techniques
        ]

        return {
            "concern": concern,
            "slot_range_start": lo,
            "slot_range_end": hi,
            "slot_count": hi - lo + 1,
            "slot_constraints": slot_constraints,
            "dimension_guide": dimension_guide,
            "technique_glossary": technique_glossary,
            "allow_dependencies": allows_dependencies(mode),
            "profile_notes": profile_notes,
            "references": references,
            "references_ambiguous": references_ambiguous,
            # 必须给默认值：PromptRegistry 用 StrictUndefined，
            # 模板引用未传的变量会直接报错（这是刻意的 —— 见 app/prompts.py）。
            "retry_feedback": "",
        }

    # ---- 校验：LLM 输出必须过 domain 纯函数 ----

    def _coerce_and_validate(
        self,
        out: PlannerOutput,
        *,
        lo: int,
        hi: int,
        mode: AskMode,
        sharpness: Sharpness,
        existing: list[PlanSlot],
        crisis_mode: bool,
        block_deep: bool,
    ) -> tuple[list[PlanSlot], list[str]]:
        errors: list[str] = []
        expected = list(range(lo, hi + 1))
        got = [s.index for s in out.slots]

        if sorted(got) != expected:
            return [], [f"槽位序号应为 {expected}，实际 {got}"]

        slots: list[PlanSlot] = []
        for raw in sorted(out.slots, key=lambda s: s.index):
            try:
                d = Dimension(raw.dimension)
            except ValueError:
                errors.append(f"第 {raw.index} 问：未知维度 {raw.dimension!r}")
                continue
            try:
                t = Technique(raw.technique)
            except ValueError:
                errors.append(f"第 {raw.index} 问：未知技术 {raw.technique!r}")
                continue

            allowed = dim_mod.filter_for_safety(
                allowed_dimensions(raw.index), block_deep=block_deep or crisis_mode
            )
            if d not in allowed:
                errors.append(
                    f"第 {raw.index} 问（{band_name(raw.index)} 段）不允许维度 {d.value}，"
                    f"应属 {sorted(x.value for x in allowed)}"
                )
                continue

            usable = allowed_techniques(
                dimension=d, sharpness=int(sharpness), crisis_mode=crisis_mode
            )
            if t not in usable:
                reason = spec(t).contraindication or "与该维度或当前锐度不匹配"
                errors.append(
                    f"第 {raw.index} 问：技术 {t.value} 不可用于维度 {d.value}（{reason}），"
                    f"可用：{[x.value for x in usable]}"
                )
                continue

            if not allows_dependencies(mode) and raw.depends_on:
                errors.append(f"第 {raw.index} 问：{mode.value} 模式下 depends_on 必须为空")
                continue

            deps = [x for x in raw.depends_on if 1 <= x < raw.index]
            slots.append(
                PlanSlot(
                    index=raw.index,
                    dimension=d,
                    technique=t,
                    intent=raw.intent.strip(),
                    depends_on=deps,
                )
            )

        if errors:
            return [], errors

        # 相邻维度不重复：必须在 existing + 新槽位的**合并序列**上校验
        merged = existing + slots
        for i in range(1, len(merged)):
            if merged[i].dimension == merged[i - 1].dimension:
                errors.append(
                    f"第 {merged[i - 1].index} 与第 {merged[i].index} 问维度重复"
                    f"（{merged[i].dimension.value}）"
                )

        # 规划完成时做整体覆盖校验
        if len(merged) == TOTAL_QUESTIONS:
            errors.extend(dim_mod.validate_coverage([s.dimension for s in merged]))

        return (slots, errors) if not errors else ([], errors)


def build_full_plan(slots: list[PlanSlot], mode: AskMode, rationale: str = "") -> QuestionPlan:
    """全部槽位就绪后构造 QuestionPlan（会再过一次 schema 层校验）。"""
    return QuestionPlan(slots=slots, mode=mode, rationale=rationale)
