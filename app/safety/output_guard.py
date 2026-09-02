"""输出侧兜底（路线文档 §3.7 / §3.1）。

拦截四类违规：
  1. **医疗断言**（诊断、确诊、疗程、处方）—— §3.1 产品定位
  2. **绝对化承诺**（"一定会好"）
  3. **越界建议**（用药、停药、法律、投资）
  4. **给建议而非提问** —— 提问阶段的核心违规（§4.4 Interviewer 禁止项）

第 4 类最重要也最容易漏：一个暗含预设答案的"问题"不是提问，
是伪装成问题的建议，直接违背苏格拉底方法（§7.5-C 的"非引导性"维度）。

命中则重生成；重生成仍失败降级到安全模板回复（§3.7）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class ViolationKind(StrEnum):
    MEDICAL_CLAIM = "medical_claim"
    ABSOLUTE_PROMISE = "absolute_promise"
    OUT_OF_SCOPE_ADVICE = "out_of_scope_advice"
    ADVICE_NOT_QUESTION = "advice_not_question"
    LEADING_QUESTION = "leading_question"
    IMPERSONATION = "impersonation"


@dataclass(frozen=True, slots=True)
class GuardRule:
    kind: ViolationKind
    pattern: re.Pattern[str]
    explain: str


def _p(expr: str) -> re.Pattern[str]:
    return re.compile(expr)


_RULES: tuple[GuardRule, ...] = (
    # 1. 医疗断言
    GuardRule(
        ViolationKind.MEDICAL_CLAIM,
        _p(r"(你(患有|得了|有)|确诊|诊断为|属于)(抑郁症|焦虑症|双相|人格障碍|精神)"),
        "对用户下医疗诊断",
    ),
    GuardRule(
        ViolationKind.MEDICAL_CLAIM,
        _p(r"(疗程|处方|开药|剂量|服药方案)"),
        "涉及医疗处置",
    ),
    GuardRule(
        ViolationKind.MEDICAL_CLAIM,
        _p(r"(这是|属于)典型的[^。？！]{0,10}(症|障碍|病)"),
        "以症状学口吻给用户归类",
    ),
    # 2. 绝对化承诺
    GuardRule(
        ViolationKind.ABSOLUTE_PROMISE,
        _p(r"(一定|肯定|必然|保证|绝对)(会)?(好起来|没问题|成功|痊愈|解决)"),
        "给出绝对化承诺",
    ),
    # 3. 越界建议
    GuardRule(
        ViolationKind.OUT_OF_SCOPE_ADVICE,
        _p(r"(建议你|你应该|可以)(去)?(停药|加药|减药|换药|吃.{0,6}药)"),
        "给用药建议",
    ),
    GuardRule(
        ViolationKind.OUT_OF_SCOPE_ADVICE,
        _p(r"(建议你|你应该)(去)?(起诉|打官司|报警处理|签这份|投资|买入|梭哈)"),
        "给法律或投资建议",
    ),
    # 4. 冒充身份
    GuardRule(
        ViolationKind.IMPERSONATION,
        _p(r"(我是|作为)(一名|一位)?(执业|持证|注册)?(心理医生|精神科医生|心理咨询师|治疗师)"),
        "声称具有执业资质",
    ),
)

# 提问阶段专用：这些句式表明模型在给建议而不是提问
_ADVICE_MARKERS: tuple[re.Pattern[str], ...] = (
    _p(r"(我建议|建议你|你应该|你需要|你最好|不如你|试试看|可以考虑|我认为你)"),
    _p(r"(最好的办法是|正确的做法是|你要做的是)"),
)

# 引导性问句：暗含预设答案，是"伪装成问题的建议"
_LEADING_MARKERS: tuple[re.Pattern[str], ...] = (
    _p(r"(难道|不觉得|是不是应该|你不认为|其实你早就知道)"),
    _p(r"[^。？！]{0,30}(才是|就是)(更好|最好|正确)的(选择|做法)[^。？！]{0,10}[？?]"),
    _p(r"(既然|明明).{0,20}(为什么不|何不)"),
)


@dataclass(frozen=True, slots=True)
class Violation:
    kind: ViolationKind
    explain: str
    matched: str


@dataclass(frozen=True, slots=True)
class GuardResult:
    ok: bool
    violations: tuple[Violation, ...] = ()

    @property
    def summary(self) -> str:
        return "；".join(f"{v.kind.value}（{v.explain}：{v.matched}）" for v in self.violations)


def check_common(text: str) -> tuple[Violation, ...]:
    out: list[Violation] = []
    for rule in _RULES:
        m = rule.pattern.search(text)
        if m:
            out.append(Violation(rule.kind, rule.explain, m.group(0)))
    return tuple(out)


def check_question(text: str) -> GuardResult:
    """校验一个提问。提问阶段的约束比其他阶段更严。"""
    violations = list(check_common(text))

    for pat in _ADVICE_MARKERS:
        m = pat.search(text)
        if m:
            violations.append(
                Violation(
                    ViolationKind.ADVICE_NOT_QUESTION,
                    "提问阶段给出建议（§4.4 Interviewer 禁止项）",
                    m.group(0),
                )
            )
            break

    for pat in _LEADING_MARKERS:
        m = pat.search(text)
        if m:
            violations.append(
                Violation(
                    ViolationKind.LEADING_QUESTION,
                    "引导性问句：暗含预设答案，是伪装成问题的建议",
                    m.group(0),
                )
            )
            break

    # 提问必须真的是问句
    if "？" not in text and "?" not in text:
        violations.append(
            Violation(
                ViolationKind.ADVICE_NOT_QUESTION,
                "输出不含问号，不是一个提问",
                text[:30],
            )
        )

    return GuardResult(ok=not violations, violations=tuple(violations))


def check_report_text(text: str) -> GuardResult:
    """校验报告类文本。允许给建议（那是报告的职责），但仍禁止医疗断言与越界。"""
    violations = check_common(text)
    return GuardResult(ok=not violations, violations=violations)


SAFE_FALLBACK_QUESTION = "关于这件事，现在最占据你注意力的是哪一部分？"


def safe_fallback(dimension_hint: str = "") -> str:
    """重生成仍失败时的安全模板回复（§3.7）。

    刻意保持开放、无预设、无锐度 —— 它的职责是不出错，不是出彩。
    """
    if dimension_hint:
        return f"说到{dimension_hint}这一块，你最先想到的是什么？"
    return SAFE_FALLBACK_QUESTION
