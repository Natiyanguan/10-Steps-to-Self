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
from app.domain.dimension import TOTAL_QUESTIONS
from app.domain.mode import allows_replan, is_batch_end
from app.graph.deps import Deps
from app.graph.state import GraphState, add_degradation
from app.llm.base import LLMError
from app.safety import injection, policy
from app.schemas.enums import DegradeReason, Phase, SafetyActionKind
from app.schemas.session import Answer, QAPair


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
