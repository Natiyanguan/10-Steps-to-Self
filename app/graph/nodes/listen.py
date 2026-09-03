"""LISTEN 节点：接收回答、安全检查、质量评估（路线文档 §4.3）。

节点内的顺序是刻意的：
  1. **先安全检查** —— 回答里可能出现危机信号，此时必须能立即短路（§3.4）
  2. 再质量评估 —— 决定是否追问
  3. 最后决定路由

安全检查绝不能放在质量评估之后：若质量评估调用超时，
危机信号就会被延迟处理甚至丢失。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.agents.quality import needs_probe
from app.agents.summarizer import merge_summaries
from app.context.budget import BudgetError, build_budgeter
from app.context.compress import plan_compression
from app.domain.dimension import TOTAL_QUESTIONS
from app.domain.mode import allows_replan, is_batch_end
from app.graph.deps import Deps
from app.graph.state import GraphState, add_degradation
from app.llm.base import LLMError, TaskKind
from app.safety import injection, policy
from app.schemas.enums import DegradeReason, Phase, SafetyActionKind
from app.schemas.session import Answer, QAPair

#: 系统提示词的保守 token 估计（interviewer 模板 + preamble + 尾部约束）。
#: 刻意往高估 —— §6.3 的原则是"宁可早压缩，不要晚触顶"：
#: 早压缩的代价是多一次摘要调用，晚触顶的代价是长会话突然截断。
#:
#: **必须用中文**：连续 ASCII 会被 BPE 大幅压缩（`"x" * 1500` 只算 225 token），
#: 而真实提示词是中文（约 1200 token）。用 ASCII 估算会严重高估可用额度、
#: 抬高压缩水位，正好违背"宁可早"的原则。
_SYSTEM_PROMPT_ESTIMATE = (
    "这是一段用于保守估算系统提示词占用量的中文占位文本，"
    "它替代真实的 interviewer 提示词参与额度估算，因此刻意偏长。"
) * 20


def _history_budget(deps: Deps) -> int | None:
    """估算可用于历史的 token 额度。无可用 provider 时返回 None（跳过压缩）。

    用 `TaskKind.ASK` 的 caps：历史是被 interviewer 消费的，
    而 interviewer 即 ASK 任务。若没有 provider 能承担 ASK，
    说明下一问根本生成不出来，压缩也救不了，直接跳过。
    """
    routes = deps.gateway.eligible_routes(TaskKind.ASK)
    if not routes:
        return None
    caps = routes[0].provider.caps
    try:
        plan = build_budgeter(caps).plan(
            system_prompt=_SYSTEM_PROMPT_ESTIMATE,
            desired_output_tokens=caps.max_output_tokens,
        )
    except BudgetError:
        # 窗口小到连保守估计都放不下 —— 不该压缩，交给上层能力门控去拒。
        return None
    return plan.available_for_history


async def _maybe_compress(
    state: GraphState, deps: Deps, update: dict[str, Any], pairs: list[QAPair]
) -> None:
    """在回答累积后、路由前，按水位决定是否压缩早期问答（§6.3）。

    压缩失败**不阻断会话**：`SummarizeResult.ok == False` 时保留原文继续，
    代价只是 token 更贵。压缩是增强手段，不是必需路径。

    注意本函数只**追加**到 `update`，不读原始 `state` 的 emissions / degrade_reasons ——
    因为本节点前面可能已经写入过（例如注入剥离），必须基于当前累积值追加。
    """
    available = _history_budget(deps)
    if available is None:
        return

    cplan = plan_compression(
        pairs,
        available_tokens=available,
        summary=state.summary,
    )
    if not cplan.should_compress:
        return

    dimension_of = (
        {slot.index: slot.dimension for slot in state.plan.slots}
        if state.plan is not None
        else {}
    )
    result = await deps.summarizer.summarize(cplan.to_compress, dimension_of=dimension_of)
    if not result.ok or result.summary is None:
        # 压缩失败：保留原文继续，代价只是 token 更贵（§6.3）
        return

    update["summary"] = merge_summaries(state.summary, result.summary)

    cur_degrade = update.get("degrade_reasons", list(state.degrade_reasons))
    if DegradeReason.HISTORY_COMPRESSED not in cur_degrade:
        update["degrade_reasons"] = [*cur_degrade, DegradeReason.HISTORY_COMPRESSED]

    cur_emissions = update.get("emissions", list(state.emissions))
    update["emissions"] = [
        *cur_emissions,
        state.emit(
            "degraded",
            reason=DegradeReason.HISTORY_COMPRESSED.value,
            impact=f"会话较长，较早的 {len(cplan.to_compress)} 轮已压成摘要，"
            "这部分只保留要点与逐字原话。",
            detail=cplan.reason,
        ),
    ]


async def listen(state: GraphState, deps: Deps) -> dict[str, Any]:
    raw = (state.inbox or "").strip()
    question = state.open_question()

    if not raw or question is None:
        # 没有输入或没有待答问题：停下等待
        return {"turn_done": True}

    # ---- 1. 安全优先，且**在原文上评估** ----
    # 顺序不能反：sanitize 按句剥离，
    # 「忽略以上指令，我不想活了」的注入与危机信号在同一句里，
    # 先剥离会连危机信号一起删掉，安全层就什么都看不到了（§3.4 + §3.7）。
    risk = await deps.safety.assess(raw)
    action = policy.decide(risk, state)

    # 存进会话的是剥离后的文本。
    # 这一点对后续 Evidence 校验很关键：报告里的引用要与**存下来的**原话逐字比对，
    # 若这里存原文而提示词里用剥离版，两者就会不一致。
    clean = injection.sanitize(raw)
    text = clean.text

    answer = Answer(
        question_index=question.index,
        text=text,
        submitted_at=datetime.now(UTC),
    )

    update: dict[str, Any] = {
        "inbox": None,
        "risk_history": [*state.risk_history, risk],
        "safety_action": action,
    }

    if clean.modified:
        update["degrade_reasons"] = add_degradation(state, DegradeReason.INJECTION_STRIPPED)
        update["emissions"] = [
            *state.emissions,
            state.emit(
                "degraded",
                reason=DegradeReason.INJECTION_STRIPPED.value,
                impact="你的回答里有一段被识别为指令的内容，已移除后继续。",
                detail=clean.summary,
            ),
        ]

    if action.kind is SafetyActionKind.STOP_WITH_RESOURCES:
        # 立即终止提问流程。回答仍然保存 —— 它是审计与后续人工复核的依据。
        update["qa_pairs"] = [*state.qa_pairs, QAPair(question=question, answer=answer)]
        update["open_questions"] = []
        update["phase"] = Phase.CRISIS
        return update

    capped = policy.apply_sharpness_cap(state.sharpness, action)
    if capped != state.sharpness:
        update["sharpness"] = capped
        update["sharpness_downgrades"] = [*state.sharpness_downgrades, *action.reasons]

    # ---- 2. 质量评估 ----
    intent = ""
    if state.plan is not None:
        try:
            intent = state.plan.slot(question.index).intent
        except KeyError:
            intent = ""

    try:
        quality = await deps.quality.evaluate(
            question=question.text,
            intent=intent,
            answer=text,
            state=state,
        )
    except LLMError:
        quality = None  # 评估不可用 → 不追问，不阻塞流程

    scored = answer.model_copy(update={"quality": quality})
    new_pairs = [*state.qa_pairs, QAPair(question=question, answer=scored)]
    update["qa_pairs"] = new_pairs
    update["open_questions"] = state.without_open_question(question)

    # ---- 2.5 上下文压缩（§6.3）----
    # 放在路由分支之前：listen 有 PROBE / PAUSED / ANALYZE 多个出口，
    # 压缩必须对每条路径都生效，否则某些模式下会漏压。
    await _maybe_compress(state, deps, update, new_pairs)

    # ---- 3. 路由 ----
    # 追问不占用 10 问配额，因此只在"正题"上触发（§4.3）
    if quality is not None and not question.is_probe:
        should, reason = needs_probe(quality, can_probe=state.can_probe(question.index))
        if should and action.allow_further_questions:
            update["phase"] = Phase.PROBE
            update["probe_counts"] = {
                **state.probe_counts,
                question.index: state.probe_count(question.index) + 1,
            }
            update["probe_reason"] = reason
            return update

    if action.kind is SafetyActionKind.HOLD_UNVERIFIABLE:
        update["phase"] = Phase.PAUSED
        update["turn_done"] = True
        return update

    answered = sum(1 for qa in new_pairs if not qa.question.is_probe)
    if answered >= TOTAL_QUESTIONS:
        update["phase"] = Phase.ANALYZE
        return update

    # 还有已发出未答的问题（ONE_SHOT / BATCHED）→ 继续等待，不重复生成
    if update["open_questions"]:
        update["phase"] = Phase.LISTEN
        update["turn_done"] = True
        return update

    # 本批答完 → 重规划剩余槽位（§4.3 LISTEN → PLAN）
    if allows_replan(state.mode) and is_batch_end(state.mode, question.index):
        update["phase"] = Phase.PLAN
    else:
        update["phase"] = Phase.ASK
    return update
