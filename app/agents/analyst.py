"""Analyst Agent —— 分段生成报告（路线文档 §6.5 / §6.4 / §4.4）。

**为什么必须分段**（§6.5）：
`InsightReport` 是深层嵌套结构，中文原话引用尤其占 token。
`LLM_MAX_TOKENS=4096` 下一次生成整份报告会触顶，
而截断发生在 JSON 中间是**稳定失败**而非偶发失败 —— 重试链救不回来。

分三段：诉求 → 冲突 → 建议，各自独立 schema 与 Evidence 校验，代码组装。
好处是某段失败可局部重试；最终仍失败时报告照样出，
该段留空并写进 `uncertainty_notes` + `degraded=true`（§9.5 降级可见）。

**每段生成后立刻做 Evidence 校验**（§6.4 第三层）：
校验不通过的洞察直接丢弃，不带进下一段 ——
否则建议段会引用一条已被判定为编造的洞察。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base import Agent
from app.llm.base import LLMError, TaskKind
from app.safety import evidence as ev_mod
from app.schemas.enums import ConfidenceLevel, DegradeReason
from app.schemas.session import (
    Evidence,
    Insight,
    InsightReport,
    SessionState,
    Suggestion,
)


# --------------------------------------------------------------------------
# 各段的输出 schema
# --------------------------------------------------------------------------
class EvidenceOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_index: int
    quote: str


class InsightOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1)
    evidence: list[EvidenceOut] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


class NeedsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface_concern: str = ""
    underlying_needs: list[InsightOut] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)


class ConflictsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inner_conflicts: list[InsightOut] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)


class SuggestionOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1)
    first_step: str = Field(min_length=1)
    linked_insights: list[int] = Field(default_factory=list)


class SuggestionsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suggestions: list[SuggestionOut] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
@dataclass(slots=True)
class SectionTrace:
    """每段的生成痕迹，供观测与质量指标使用（§11.3）。"""

    name: str
    ok: bool
    attempts: int = 0
    kept: int = 0
    dropped: int = 0
    error: str = ""


@dataclass(slots=True)
class AnalysisResult:
    report: InsightReport
    traces: list[SectionTrace] = field(default_factory=list)

    @property
    def evidence_pass_rate(self) -> float:
        kept = sum(t.kept for t in self.traces)
        dropped = sum(t.dropped for t in self.traces)
        total = kept + dropped
        return 1.0 if total == 0 else kept / total


def _qa_context(state: SessionState) -> list[dict[str, object]]:
    """构造问答上下文。追问也要带上 —— 它们常含最关键的材料。"""
    out: list[dict[str, object]] = []
    for qa in state.qa_pairs:
        slot = None
        if state.plan is not None:
            try:
                slot = state.plan.slot(qa.question.index)
            except KeyError:
                slot = None
        out.append(
            {
                "index": qa.question.index,
                "question": qa.question.text,
                "answer": qa.answer.text,
                "dimension": slot.dimension.value if slot else "",
                "technique": qa.question.technique.value,
            }
        )
    return out


def _to_insights(raw: list[InsightOut]) -> list[Insight]:
    """转换成领域模型。无依据的条目在此丢弃 —— schema 要求 evidence 非空。"""
    out: list[Insight] = []
    for item in raw:
        if not item.evidence:
            continue
        out.append(
            Insight(
                statement=item.statement.strip(),
                evidence=[
                    Evidence(question_index=e.question_index, quote=e.quote)
                    for e in item.evidence
                ],
                confidence=item.confidence,
            )
        )
    return out


class Analyst(Agent):
    prompt_agent = "analyst"
    task = TaskKind.ANALYZE

    async def analyze(
        self,
        state: SessionState,
        *,
        memory_conflicts: list[dict[str, str]] | None = None,
        resources_mentioned: list[str] | None = None,
        critic_feedback: list[str] | None = None,
    ) -> AnalysisResult:
        qa_pairs = _qa_context(state)
        user_texts = state.user_texts()
        answered = {qa.question.index for qa in state.qa_pairs}
        traces: list[SectionTrace] = []
        notes: list[str] = []
        degrade: list[DegradeReason] = []

        # ---- 第 1 段：表层困惑 + 深层诉求 ----
        needs_section, t1 = await self._section(
            "needs_v1",
            NeedsSection,
            {
                "concern": state.concern,
                "qa_pairs": qa_pairs,
                "critic_feedback": critic_feedback or [],
            },
            max_output_tokens=1400,
        )
        traces.append(t1)

        surface = ""
        needs: list[Insight] = []
        if needs_section is not None:
            surface = needs_section.surface_concern.strip()
            notes.extend(needs_section.uncertainty_notes)
            outcome = ev_mod.verify_insights(
                _to_insights(needs_section.underlying_needs),
                user_texts=user_texts,
                answered_indices=answered,
            )
            needs = outcome.kept
            t1.kept, t1.dropped = len(outcome.kept), len(outcome.dropped)
            if outcome.dropped:
                degrade.append(DegradeReason.EVIDENCE_DROPPED)
                notes.append(
                    f"有 {len(outcome.dropped)} 条深层诉求因引用无法在你的原话中核实而被移除"
                )
        else:
            degrade.append(DegradeReason.REPORT_SECTION_FAILED)
            notes.append("深层诉求部分生成失败，本次报告缺少这一节")

        # ---- 第 2 段：内在冲突 ----
        conflicts_section, t2 = await self._section(
            "conflicts_v1",
            ConflictsSection,
            {
                "concern": state.concern,
                "qa_pairs": qa_pairs,
                "known_needs": [i.statement for i in needs],
                "memory_conflicts": memory_conflicts or [],
            },
            max_output_tokens=1200,
        )
        traces.append(t2)

        conflicts: list[Insight] = []
        if conflicts_section is not None:
            notes.extend(conflicts_section.uncertainty_notes)
            outcome = ev_mod.verify_insights(
                _to_insights(conflicts_section.inner_conflicts),
                user_texts=user_texts,
                answered_indices=answered,
            )
            conflicts = outcome.kept
            t2.kept, t2.dropped = len(outcome.kept), len(outcome.dropped)
            if outcome.dropped:
                degrade.append(DegradeReason.EVIDENCE_DROPPED)
        else:
            degrade.append(DegradeReason.REPORT_SECTION_FAILED)
            notes.append("内在冲突部分生成失败")

        # ---- 第 3 段：建议 ----
        # 只把**已通过校验**的洞察喂进去，编号即列表下标。
        available = needs + conflicts
        suggestions: list[Suggestion] = []
        if available:
            sug_section, t3 = await self._section(
                "suggestions_v1",
                SuggestionsSection,
                {
                    "concern": state.concern,
                    "needs": [i.statement for i in needs],
                    "conflicts": [i.statement for i in conflicts],
                    "resources_mentioned": resources_mentioned or [],
                    "max_index": len(available) - 1,
                },
                max_output_tokens=1200,
            )
            traces.append(t3)
            if sug_section is not None:
                notes.extend(sug_section.uncertainty_notes)
                for s in sug_section.suggestions:
                    # 模型常引用"它以为存在"的洞察编号，这里代码校验（§6.5）
                    links = ev_mod.validate_suggestion_links(
                        s.linked_insights, len(available)
                    )
                    suggestions.append(
                        Suggestion(
                            action=s.action.strip(),
                            first_step=s.first_step.strip(),
                            linked_insights=links,
                        )
                    )
                t3.kept = len(suggestions)
            else:
                degrade.append(DegradeReason.REPORT_SECTION_FAILED)
                notes.append("建议部分生成失败")
        else:
            traces.append(
                SectionTrace(
                    name="suggestions_v1",
                    ok=False,
                    error="无通过校验的洞察，跳过建议生成",
                )
            )
            notes.append("由于没有可核实的洞察，本次不给出建议")
            degrade.append(DegradeReason.REPORT_SECTION_FAILED)

        report = InsightReport(
            surface_concern=surface or state.concern[:200],
            underlying_needs=needs,
            inner_conflicts=conflicts,
            suggestions=suggestions,
            uncertainty_notes=_dedupe(notes),
            degraded=bool(degrade),
            degrade_reasons=_dedupe_enum(degrade),
        )
        return AnalysisResult(report=report, traces=traces)

    async def _section[S: BaseModel](
        self,
        version: str,
        model: type[S],
        context: dict[str, object],
        *,
        max_output_tokens: int,
    ) -> tuple[S | None, SectionTrace]:
        """生成一段。失败不抛异常 —— 让报告能带着缺口出来（§6.5）。"""
        trace = SectionTrace(name=version, ok=False)
        try:
            prompt = self.render(context, version=version)
            out, outcome = await self.call(
                prompt,
                model,
                user_payload="请开始。",
                max_output_tokens=max_output_tokens,
            )
        except LLMError as exc:
            trace.error = str(exc)[:300]
            return None, trace
        trace.ok = True
        trace.attempts = outcome.attempts
        return out, trace


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        s = x.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _dedupe_enum(items: list[DegradeReason]) -> list[DegradeReason]:
    seen: set[DegradeReason] = set()
    out: list[DegradeReason] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
