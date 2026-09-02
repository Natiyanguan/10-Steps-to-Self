"""上下文预算与 token 计数（路线文档 §6.3 / §2.2 / §6.5）。

两个容易踩的坑：

1. **tiktoken 对 DeepSeek 不准**（§6.3）。tiktoken 只覆盖 OpenAI 系词表；
   对 DeepSeek 与本地模型只能估算。因此估算模式必须**在日志里标记**，
   否则一旦发生意外截断，根本无法归因到"预算算错了"还是"输入真的太长"。
   中文的 token/字 比值随内容波动，安全系数不是保守，是必需。

2. **输出预留必须按绝对值算，不能按百分比**（§6.5）。
   `LLM_MAX_TOKENS=4096` 是硬上限；按"20% 留给输出"这种算法，
   在长上下文模型上会预留过多、在短上下文模型上会预留不足。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from typing import Any

from app.llm.base import Message, Role

# 估算安全系数（§6.3）。中文实测 token/字 通常在 0.6-1.0 之间波动，
# 词表不匹配时会偏低估，因此宁可高估。
ESTIMATE_SAFETY_FACTOR = 1.20

# 每条消息的协议开销（role、分隔符等），各家略有差异，取保守值
PER_MESSAGE_OVERHEAD = 4


class CountMode(StrEnum):
    EXACT = "exact"          # 使用了目标模型的官方 tokenizer
    ESTIMATED = "estimated"  # 用 cl100k_base 近似 + 安全系数（必须标记）


@dataclass(frozen=True, slots=True)
class TokenCount:
    tokens: int
    mode: CountMode

    @property
    def is_estimated(self) -> bool:
        return self.mode is CountMode.ESTIMATED


@lru_cache(maxsize=8)
def _encoding(name: str) -> Any:
    import tiktoken

    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_text(text: str, *, tokenizer_hint: str = "cl100k_base") -> TokenCount:
    """计数单段文本。

    `tokenizer_hint` 来自 `ProviderCaps.tokenizer_hint`（§17 探测写入）。
    只有明确标注为官方词表时才算 EXACT。
    """
    if not text:
        return TokenCount(0, CountMode.EXACT)

    # 本地模型或未知词表：无法精确计数
    if tokenizer_hint in ("local", "unknown", ""):
        approx = _encoding("cl100k_base").encode(text, disallowed_special=())
        return TokenCount(int(len(approx) * ESTIMATE_SAFETY_FACTOR), CountMode.ESTIMATED)

    enc = _encoding(tokenizer_hint)
    raw = len(enc.encode(text, disallowed_special=()))

    # cl100k_base 用在非 OpenAI 模型上就是估算 —— 这里是 §6.3 的核心提醒
    if tokenizer_hint == "cl100k_base":
        return TokenCount(int(raw * ESTIMATE_SAFETY_FACTOR), CountMode.ESTIMATED)
    return TokenCount(raw, CountMode.EXACT)


def count_messages(
    messages: list[Message], *, tokenizer_hint: str = "cl100k_base"
) -> TokenCount:
    total = 0
    mode = CountMode.EXACT
    for m in messages:
        c = count_text(m.content, tokenizer_hint=tokenizer_hint)
        total += c.tokens + PER_MESSAGE_OVERHEAD
        if c.is_estimated:
            mode = CountMode.ESTIMATED
    return TokenCount(total, mode)


class BudgetError(RuntimeError):
    """预算无法满足 —— 显式失败，不静默截断到不可用的程度。"""


@dataclass(slots=True)
class BudgetPlan:
    """一次调用的预算分配结果。"""

    context_window: int
    reserved_output: int
    system_tokens: int
    available_for_history: int
    mode: CountMode
    notes: list[str] = field(default_factory=list)

    @property
    def is_estimated(self) -> bool:
        return self.mode is CountMode.ESTIMATED


class ContextBudgeter:
    """按 provider 能力分配上下文额度。

    分配顺序（§6.3，优先级从高到低）：
      1. system prompt + 安全指令（**绝不压缩**）
      2. 输出预留（绝对值）
      3. 当前问题与最近若干轮原文
      4. 更早历史的摘要
    """

    def __init__(
        self,
        *,
        context_window: int,
        max_output_tokens: int,
        tokenizer_hint: str = "cl100k_base",
        headroom_ratio: float = 0.05,
    ) -> None:
        self._window = context_window
        self._max_output = max_output_tokens
        self._hint = tokenizer_hint
        self._headroom = headroom_ratio

    def plan(
        self,
        *,
        system_prompt: str,
        desired_output_tokens: int,
    ) -> BudgetPlan:
        notes: list[str] = []

        sys_count = count_text(system_prompt, tokenizer_hint=self._hint)

        # 输出预留：绝对值取 min(期望, provider 上限)（§6.5）
        reserved = min(desired_output_tokens, self._max_output)
        if reserved < desired_output_tokens:
            notes.append(
                f"输出预留被 provider 上限压到 {reserved}（期望 {desired_output_tokens}）："
                "报告类任务应进一步分段（§6.5）"
            )

        headroom = int(self._window * self._headroom)
        available = self._window - sys_count.tokens - reserved - headroom

        if available <= 0:
            raise BudgetError(
                f"上下文窗口 {self._window} 不足：system {sys_count.tokens} + "
                f"输出预留 {reserved} + 余量 {headroom} 已超出。\n"
                "  这通常意味着 provider 档位与任务不匹配，应由能力门控提前拒绝（§2.4.3）。"
            )

        if sys_count.is_estimated:
            notes.append(
                "token 计数为**估算**（非目标模型官方 tokenizer，"
                f"已乘安全系数 {ESTIMATE_SAFETY_FACTOR}）。"
                "若出现意外截断，先怀疑此处（§6.3）。"
            )

        return BudgetPlan(
            context_window=self._window,
            reserved_output=reserved,
            system_tokens=sys_count.tokens,
            available_for_history=available,
            mode=sys_count.mode,
            notes=notes,
        )

    def fit_messages(
        self,
        *,
        system_prompt: str,
        history: list[Message],
        desired_output_tokens: int,
        keep_recent: int = 4,
    ) -> tuple[list[Message], BudgetPlan, bool]:
        """把历史裁剪进预算。

        返回 (可用消息列表, 预算方案, 是否发生截断)。
        截断是降级事件，调用方必须记 `DegradeReason.CONTEXT_TRUNCATED`（§9.5）。

        裁剪策略：保留最近 `keep_recent` 轮原文，从**最早**的开始丢 ——
        近期上下文对提问连贯性更关键。
        """
        plan = self.plan(
            system_prompt=system_prompt, desired_output_tokens=desired_output_tokens
        )

        counted = [
            (m, count_text(m.content, tokenizer_hint=self._hint).tokens + PER_MESSAGE_OVERHEAD)
            for m in history
        ]
        total = sum(t for _, t in counted)

        if total <= plan.available_for_history:
            return list(history), plan, False

        # 必须保留的尾部
        tail = counted[-keep_recent:] if keep_recent > 0 else []
        tail_tokens = sum(t for _, t in tail)

        if tail_tokens > plan.available_for_history:
            # 连最近几轮都放不下：只保留最后一条，并明确标记严重截断
            plan.notes.append(
                f"最近 {keep_recent} 轮（{tail_tokens} tokens）已超出可用额度 "
                f"{plan.available_for_history}，仅保留最后一轮。"
                "该会话应考虑提前进入分析阶段。"
            )
            return [counted[-1][0]], plan, True

        kept: list[Message] = [m for m, _ in tail]
        budget_left = plan.available_for_history - tail_tokens

        # 从紧邻 tail 的位置往前补，直到额度用尽
        for m, t in reversed(counted[: len(counted) - len(tail)]):
            if t > budget_left:
                continue
            kept.insert(0, m)
            budget_left -= t

        dropped = len(history) - len(kept)
        plan.notes.append(f"上下文超限，已丢弃最早 {dropped} 条历史（§6.3）")
        return kept, plan, True


def build_budgeter(caps: object) -> ContextBudgeter:
    """从 `ProviderCaps` 构造 budgeter（避免上层手抄字段）。"""
    return ContextBudgeter(
        context_window=getattr(caps, "context_window", 8_000),
        max_output_tokens=getattr(caps, "max_output_tokens", 2_048),
        tokenizer_hint=getattr(caps, "tokenizer_hint", "cl100k_base"),
    )


def system_message(content: str) -> Message:
    return Message(Role.SYSTEM, content)
