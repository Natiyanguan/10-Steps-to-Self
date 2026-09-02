"""Evidence 程序化校验 —— 全系统最重要的反幻觉设计（路线文档 §6.4 第三层）。

为什么这层最硬：它**不依赖任何模型判断**，纯字符串校验。
前两层（提示词约束、结构化输出强制挂依据）都可能被模型绕过；
Critic（第四层）是模型审模型，仍可能同时出错。只有字符串匹配不会。

规则：`Evidence.quote` 必须能在用户原话中找到。
**校验不通过的 Insight 直接丢弃**，不保留待人工判断（§6.4）。

中文特化（§2.2 第 4 条）：必须归一化全角/半角、中英标点、空白、繁简，
否则用户原话校验会因标点差异误判失败 —— 那会把正确的洞察也丢掉。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

import zhconv

from app.schemas.session import Evidence, Insight

DEFAULT_THRESHOLD = 0.9
MIN_QUOTE_CHARS = 4  # 过短的引用无法证明任何东西

# 标点与空白：归一化时一并剥离（中英标点混用是中文文本的常态）
_PUNCT = re.compile(
    r"[\s\.,;:!?\"'`~@#\$%\^&\*\(\)\[\]\{\}<>/\\|\-_=\+"
    r"。，、；：！？…—～·「」『』【】《》（）〈〉“”‘’]+"
)

def normalize_for_match(text: str) -> str:
    """归一化到可比对形式：NFKC → 繁转简 → 去标点空白 → 小写。

    繁简转换用 `zhconv` 而非手写映射表：手写表一定会漏字
    （最初漏了「穩」），而漏字会让**合法的洞察被误判为编造并丢弃** ——
    校验器的假阴性比假阳性更糟，它会静默删掉正确内容。
    """
    s = unicodedata.normalize("NFKC", text)
    s = zhconv.convert(s, "zh-cn")
    s = _PUNCT.sub("", s)
    return s.lower()


@dataclass(frozen=True, slots=True)
class QuoteCheck:
    ok: bool
    ratio: float
    reason: str = ""


def _best_window_ratio(needle: str, haystack: str) -> float:
    """在 haystack 中滑窗寻找与 needle 最相似的片段，返回最高相似度。

    用于容忍模型对原话的轻微改写（多一个字、少一个字），
    但不容忍语义级的编造 —— 阈值 0.9 把这条线划得很紧。
    """
    n, h = len(needle), len(haystack)
    if n == 0 or h == 0:
        return 0.0
    if n > h:
        return SequenceMatcher(None, needle, haystack).ratio()

    best = 0.0
    # 窗口略宽于 needle，给"多一两个字"留余量；步长 1 保证不漏。
    width = min(h, n + max(2, n // 10))
    for start in range(0, h - n + 1):
        window = haystack[start : start + width]
        ratio = SequenceMatcher(None, needle, window).ratio()
        if ratio > best:
            best = ratio
            if best >= 0.999:
                break
    return best


def verify_quote(
    quote: str,
    user_texts: list[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> QuoteCheck:
    """校验单条引用是否出自用户原话。纯函数。"""
    if len(quote.strip()) < MIN_QUOTE_CHARS:
        return QuoteCheck(False, 0.0, f"引用过短（<{MIN_QUOTE_CHARS} 字），无法作为依据")

    needle = normalize_for_match(quote)
    if not needle:
        return QuoteCheck(False, 0.0, "引用归一化后为空（可能只含标点）")

    best = 0.0
    for text in user_texts:
        hay = normalize_for_match(text)
        if not hay:
            continue
        if needle in hay:  # 快路径：精确子串
            return QuoteCheck(True, 1.0)
        ratio = _best_window_ratio(needle, hay)
        best = max(best, ratio)
        if best >= threshold:
            return QuoteCheck(True, best)

    return QuoteCheck(
        False,
        best,
        f"引用未在用户原话中找到（最高相似度 {best:.2f} < {threshold}），判定为编造",
    )


@dataclass(frozen=True, slots=True)
class DroppedInsight:
    statement: str
    reason: str


@dataclass(frozen=True, slots=True)
class VerifyOutcome:
    kept: list[Insight]
    dropped: list[DroppedInsight]

    @property
    def pass_rate(self) -> float:
        """Evidence 校验通过率 —— §11.3 的核心质量指标，直接反映幻觉水平。"""
        total = len(self.kept) + len(self.dropped)
        return 1.0 if total == 0 else len(self.kept) / total


def _evidence_index_valid(ev: Evidence, answered_indices: set[int]) -> bool:
    return ev.question_index in answered_indices


def verify_insights(
    insights: list[Insight],
    *,
    user_texts: list[str],
    answered_indices: set[int] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> VerifyOutcome:
    """逐条校验洞察的依据。

    一条 Insight 的所有 Evidence 中：
      - 至少一条通过 → 保留该 Insight，但只保留通过的 Evidence
      - 全部失败     → **丢弃该 Insight**（§6.4：不保留待人工判断）
    """
    kept: list[Insight] = []
    dropped: list[DroppedInsight] = []

    for ins in insights:
        good: list[Evidence] = []
        reasons: list[str] = []

        for ev in ins.evidence:
            if answered_indices is not None and not _evidence_index_valid(ev, answered_indices):
                reasons.append(f"引用了不存在的问题序号 {ev.question_index}")
                continue
            check = verify_quote(ev.quote, user_texts, threshold=threshold)
            if check.ok:
                good.append(ev)
            else:
                reasons.append(check.reason)

        if good:
            kept.append(ins.model_copy(update={"evidence": good}))
        else:
            dropped.append(
                DroppedInsight(
                    statement=ins.statement,
                    reason="；".join(reasons) or "无有效依据",
                )
            )

    return VerifyOutcome(kept=kept, dropped=dropped)


def validate_suggestion_links(
    linked: list[int], available_count: int
) -> list[int]:
    """建议引用的洞察索引必须真实存在（§6.5 分段生成后的代码校验）。

    模型在分段生成时容易引用一个"它以为存在"的洞察编号。
    这里做代码校验而非依赖模型自觉。
    """
    return [i for i in linked if 0 <= i < available_count]
