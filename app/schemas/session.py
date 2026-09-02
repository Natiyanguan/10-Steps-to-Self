"""领域模型 —— 全系统的契约（路线文档 §5.1）。

四个关键设计点：
  - `Evidence.quote` 必须是用户原话子串 —— 把"不许编造"从提示词约束
    升级为可程序化验证的约束（§6.4 第三层，全系统最重要的反幻觉设计）。
  - `Suggestion.first_step` 强制 24h 内可启动，逼出具体动作而非"调整心态"。
  - `uncertainty_notes` 是允许模型说不知道的出口。没有它，模型只会编。
  - `revision` 用于乐观锁并发控制（§9.1）。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.dimension import TOTAL_QUESTIONS, Dimension, validate_coverage
from app.domain.mode import AskMode, allows_dependencies
from app.domain.technique import Technique
from app.schemas.enums import (
    ConfidenceLevel,
    ConsentScope,
    DegradeReason,
    Phase,
    RiskLevel,
    RiskSource,
    Sharpness,
)

MAX_QUESTION_CHARS = 300  # §5.1 硬约束：防认知过载
MAX_PROBES_PER_QUESTION = 2  # §4.3 追问上限


class StrictModel(BaseModel):
    """禁止未声明字段：LLM 多吐字段时应显式失败，而非静默接受。"""

    model_config = ConfigDict(extra="forbid", frozen=False)


# --------------------------------------------------------------------------
# 同意与授权
# --------------------------------------------------------------------------
class Consent(StrictModel):
    """知情同意（§3.2）。条款会变，因此必须版本化 + 分项授权。"""

    policy_version: str
    granted_at: datetime
    scopes: set[ConsentScope] = Field(default_factory=set)
    withdrawn_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.withdrawn_at is None

    def allows(self, scope: ConsentScope) -> bool:
        return self.is_active and scope in self.scopes

    def is_current(self, policy_version: str) -> bool:
        """是否仍然有效：未撤回**且**条款版本匹配（§3.2）。

        单一判定入口。之前 intake 与 consent 节点各自判断，
        intake 只看"有没有 consent 对象"，导致过期版本的同意能绕过闸门 ——
        这类重复逻辑必然发散，所以判定只留这一处。
        """
        if not self.is_active:
            return False
        if policy_version and policy_version != "unset":
            return self.policy_version == policy_version
        return True


# --------------------------------------------------------------------------
# 规划与提问
# --------------------------------------------------------------------------
class PlanSlot(StrictModel):
    index: int = Field(ge=1, le=TOTAL_QUESTIONS)
    dimension: Dimension
    technique: Technique
    intent: str = Field(min_length=1, max_length=200)  # 内部说明，不展示给用户
    depends_on: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_self_dependency(self) -> PlanSlot:
        if self.index in self.depends_on:
            raise ValueError(f"第 {self.index} 问不能依赖自己")
        if any(d >= self.index for d in self.depends_on):
            raise ValueError(f"第 {self.index} 问只能依赖更早的问题")
        return self


class QuestionPlan(StrictModel):
    """10 问的维度与技术序列。注意：这里不含问题文本（§4.4 Planner 禁止生成文本）。

    **允许承载分批规划的中间态**：BATCHED（3→4→3）与 PROGRESSIVE 模式下，
    规划是逐批产生的，`slots` 会先只有 3 个再逐步补齐。
    因此整体覆盖度只在槽位满 `TOTAL_QUESTIONS` 时校验，
    而逐问维度合法性与相邻不重复**始终校验**（错误越早暴露越好）。
    """

    slots: list[PlanSlot]
    mode: AskMode
    rationale: str = Field(default="", max_length=1000)

    @property
    def is_complete(self) -> bool:
        return len(self.slots) == TOTAL_QUESTIONS

    @model_validator(mode="after")
    def _check_plan(self) -> QuestionPlan:
        indices = [s.index for s in self.slots]
        if indices != list(range(1, len(self.slots) + 1)):
            raise ValueError(f"槽位序号必须是 1..N 连续，收到 {indices}")

        errors = validate_coverage(
            [s.dimension for s in self.slots], complete=self.is_complete
        )
        if errors:
            raise ValueError("维度规划不合法：" + "；".join(errors))

        if not allows_dependencies(self.mode):
            offenders = [s.index for s in self.slots if s.depends_on]
            if offenders:
                raise ValueError(
                    f"{self.mode} 模式下 depends_on 必须为空，违规槽位：{offenders}"
                )
        return self

    def slot(self, index: int) -> PlanSlot:
        for s in self.slots:
            if s.index == index:
                return s
        raise KeyError(f"槽位 {index} 不存在")


class Question(StrictModel):
    index: int = Field(ge=1, le=TOTAL_QUESTIONS)
    text: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    technique: Technique
    sharpness: Sharpness
    is_probe: bool = False  # 追问不占用 10 问配额（§4.3）
    prompt_version: str | None = None  # 便于问题复现时定位版本（§6.1）


class AnswerQuality(StrictModel):
    score: float = Field(ge=0.0, le=1.0)
    is_evasive: bool = False
    is_too_brief: bool = False
    missing_aspects: list[str] = Field(default_factory=list)  # 供 PROBE 生成追问


class Answer(StrictModel):
    question_index: int = Field(ge=1, le=TOTAL_QUESTIONS)
    text: str
    submitted_at: datetime | None = None
    quality: AnswerQuality | None = None


class QAPair(StrictModel):
    """问答对。用显式模型而非 tuple，便于序列化与按索引检索。"""

    question: Question
    answer: Answer


# --------------------------------------------------------------------------
# 安全
# --------------------------------------------------------------------------
class RiskAssessment(StrictModel):
    level: RiskLevel
    source: RiskSource
    matched_rules: list[str] = Field(default_factory=list)
    evidence: str | None = None  # 命中片段；日志中需脱敏（§11.1）
    assessed_at: datetime | None = None


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
class Evidence(StrictModel):
    """每条洞察必须挂依据。`quote` 必须能在用户原话中找到（§6.4 第三层）。"""

    question_index: int = Field(ge=1, le=TOTAL_QUESTIONS)
    quote: str = Field(min_length=1)


class Insight(StrictModel):
    statement: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)  # 无依据不许成条
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


class Suggestion(StrictModel):
    action: str = Field(min_length=1)
    first_step: str = Field(min_length=1)  # 必须 24h 内可启动
    linked_insights: list[int] = Field(default_factory=list)


class InsightReport(StrictModel):
    surface_concern: str = ""
    underlying_needs: list[Insight] = Field(default_factory=list)
    inner_conflicts: list[Insight] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    degraded: bool = False
    degrade_reasons: list[DegradeReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def _degraded_needs_reason(self) -> InsightReport:
        # §9.5：降级必须可见且可归因，不允许只置标志位不给原因。
        if self.degraded and not self.degrade_reasons:
            raise ValueError("degraded=True 时必须给出 degrade_reasons")
        return self


class DimensionSummary(StrictModel):
    """某一维度上的压缩结果（§6.3）。"""

    dimension: Dimension
    facts: list[str] = Field(default_factory=list)
    emotions: list[str] = Field(default_factory=list)


class SessionSummary(StrictModel):
    """早期问答的压缩产物（§6.3）。

    `preserved_quotes` 是这个模型存在的**核心理由**：
    报告里每条洞察都要挂原话依据，而 `app/safety/evidence.py` 会**逐字**校验引用。
    一旦摘要把原话改写了，那句话就再也无法作为依据 —— 对应洞察会被程序丢弃，
    即使它是对的。所以摘要必须显式携带一份逐字原话清单。

    `covered_indices` 记录被压缩掉的问题序号，用于：
      - 判断某条引用该在摘要里找还是在完整问答里找
      - 观测压缩比（§11.3）
    """

    by_dimension: list[DimensionSummary] = Field(default_factory=list)
    preserved_quotes: list[str] = Field(default_factory=list)
    covered_indices: list[int] = Field(default_factory=list)
    #: 摘要自身的 token 估算，避免重复计算
    approx_tokens: int = 0
    created_at: datetime | None = None

    @property
    def is_empty(self) -> bool:
        return not self.by_dimension and not self.preserved_quotes

    def render(self) -> str:
        """渲染成可放入提示词的文本。

        逐字原话单独成段并明确标注 —— 下游 Analyst 需要知道
        哪些句子可以直接引用，哪些是概括（概括过不能当依据）。
        """
        lines: list[str] = []
        for item in self.by_dimension:
            parts: list[str] = []
            if item.facts:
                parts.append("事实：" + "；".join(item.facts))
            if item.emotions:
                parts.append("情绪：" + "；".join(item.emotions))
            if parts:
                lines.append(f"- {item.dimension.value}：" + " / ".join(parts))

        out: list[str] = []
        if lines:
            out.append("【早期问答摘要（已概括，不可作为引用依据）】")
            out.extend(lines)
        if self.preserved_quotes:
            out.append("【保留的逐字原话（可作为引用依据）】")
            out.extend(f"- 「{q}」" for q in self.preserved_quotes)
        return "\n".join(out)


class CritiqueIssue(StrictModel):
    target: str  # 如 "underlying_needs[2]"
    problem: str
    severity: ConfidenceLevel = ConfidenceLevel.MEDIUM


class CritiqueResult(StrictModel):
    """Critic 只否决不改写（§4.4 铁律 2）—— 因此这里没有修改后的报告字段。"""

    approved: bool
    issues: list[CritiqueIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rejection_needs_issues(self) -> CritiqueResult:
        if not self.approved and not self.issues:
            raise ValueError("否决时必须给出具体 issues，否则 Analyst 无法针对性重做")
        return self


# --------------------------------------------------------------------------
# 会话状态
# --------------------------------------------------------------------------
class SessionState(StrictModel):
    session_id: str
    user_id: str
    phase: Phase = Phase.INTAKE
    mode: AskMode = AskMode.BATCHED
    sharpness: Sharpness = Sharpness.MODERATE
    concern: str = ""

    plan: QuestionPlan | None = None
    qa_pairs: list[QAPair] = Field(default_factory=list)
    #: 已发出但未回答的问题。
    #: 用列表而非单个字段：ONE_SHOT/BATCHED 会一次发出多问，
    #: 且用户可以不按顺序回答 —— 单个 pending 字段表达不了这两种情况。
    open_questions: list[Question] = Field(default_factory=list)
    probe_counts: dict[int, int] = Field(default_factory=dict)
    #: 早期问答的压缩摘要（§6.3）。
    #: **原始 qa_pairs 不删除** —— 摘要只影响放进提示词的内容，
    #: 不影响持久化与审计。删原文会让"引用逐字校验"彻底失去基准。
    summary: SessionSummary | None = None

    risk_history: list[RiskAssessment] = Field(default_factory=list)
    sharpness_downgrades: list[str] = Field(default_factory=list)  # 降级原因（§3.5）

    report: InsightReport | None = None
    consent: Consent | None = None
    is_minor: bool = False  # 未成年人：强制 GENTLE 且不可上调（§3.2）

    created_at: datetime | None = None
    paused_at: datetime | None = None
    revision: int = 0  # 乐观锁（§9.1）

    degrade_reasons: list[DegradeReason] = Field(default_factory=list)

    # ---- 派生查询（纯函数，便于图节点与测试使用）----

    @property
    def answered_count(self) -> int:
        return sum(1 for qa in self.qa_pairs if not qa.question.is_probe)

    @property
    def next_question_index(self) -> int:
        return self.answered_count + 1

    @property
    def is_complete(self) -> bool:
        return self.answered_count >= TOTAL_QUESTIONS

    @property
    def latest_risk(self) -> RiskAssessment | None:
        return self.risk_history[-1] if self.risk_history else None

    def answers_for(self, index: int) -> list[Answer]:
        return [qa.answer for qa in self.qa_pairs if qa.answer.question_index == index]

    def user_texts(self) -> list[str]:
        """全部用户原话 —— Evidence 校验的比对基准（§6.4）。"""
        texts = [self.concern] if self.concern else []
        texts.extend(qa.answer.text for qa in self.qa_pairs)
        return texts

    def probe_count(self, index: int) -> int:
        return self.probe_counts.get(index, 0)

    def can_probe(self, index: int) -> bool:
        return self.probe_count(index) < MAX_PROBES_PER_QUESTION

    def open_question(self, index: int | None = None) -> Question | None:
        """取一个未回答的问题。

        `index=None` 时返回最早发出的那一个（渐进模式的自然语义）。
        指定 index 时用于 ONE_SHOT：用户可以挑着回答。
        """
        if not self.open_questions:
            return None
        if index is None:
            return self.open_questions[0]
        for q in self.open_questions:
            if q.index == index:
                return q
        return None

    def without_open_question(self, question: Question) -> list[Question]:
        """回答后把该问从待答列表移除（追问与正题各自独立成条）。"""
        return [
            q
            for q in self.open_questions
            if not (q.index == question.index and q.is_probe == question.is_probe)
        ]
