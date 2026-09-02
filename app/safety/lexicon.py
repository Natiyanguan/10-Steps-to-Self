"""危机识别第一层：规则前置过滤器（路线文档 §3.4）。

三条硬要求：
  1. **零 LLM 依赖、零网络、零 IO**。LLM 全挂时这层必须仍然工作（§9.3）。
  2. 覆盖中文变体：谐音、拆字、拼音缩写、符号/空格插入、繁简、火星文。
  3. **假阴性远比假阳性可怕** —— 阈值偏保守，宁可多触发 MEDIUM。

延迟目标 < 1ms。因此用预编译正则 + 归一化，不做任何模型推理。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import zhconv

from app.schemas.enums import RiskLevel

# --------------------------------------------------------------------------
# 归一化：把变体写法折叠成可匹配的形式
# --------------------------------------------------------------------------

# 常见的规避手段：在关键词中间插入这些字符
_NOISE = re.compile(r"[\s\.\-_*~·。，,、!！?？:：;；\"'（）()\[\]【】<>《》/\\|+=#@\$%^&]+")

# 谐音与变体 → 标准形（只做保守、低歧义的映射）
_HOMOPHONE_MAP: tuple[tuple[str, str], ...] = (
    ("sha", "杀"),
    ("shā", "杀"),
    ("煞", "杀"),
    ("鲨", "杀"),
    ("刹", "杀"),
    ("zs", "自杀"),
    ("zsha", "自杀"),
    ("自我了断", "自杀"),
    ("了结自己", "自杀"),
    ("结束自己", "自杀"),
    ("寻短见", "自杀"),
    ("自我伤害", "自伤"),
    ("自我了结", "自杀"),
    ("qingsheng", "轻生"),
    ("自戕", "自杀"),
)


def normalize(text: str) -> str:
    """归一化：NFKC → 繁转简 → 小写 → 去噪声字符 → 谐音折叠。

    繁简转换用 `zhconv` 全量词库，而不是在词表里逐条枚举繁体写法：
    这里的**假阴性等于漏检危机**（§3.4），
    而手写繁体规则必然漏字 —— 靠枚举覆盖不了「輕生/自殘/撐不下去」等全部变体。
    """
    s = unicodedata.normalize("NFKC", text)
    s = zhconv.convert(s, "zh-cn")
    s = s.lower()
    s = _NOISE.sub("", s)
    for src, dst in _HOMOPHONE_MAP:
        s = s.replace(src, dst)
    return s


# --------------------------------------------------------------------------
# 词表：每条规则 = (规则名, 正则, 风险等级)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LexiconRule:
    name: str
    pattern: re.Pattern[str]
    level: RiskLevel
    note: str = ""


def _p(expr: str) -> re.Pattern[str]:
    return re.compile(expr)


# IMMINENT：表达了即刻的、有计划或正在进行的自伤/自杀
_IMMINENT_RULES: tuple[LexiconRule, ...] = (
    LexiconRule(
        "imminent_plan_now",
        _p(r"(现在|马上|立刻|今晚|今天|等下|一会儿)(就)?(去)?(自杀|自尽|轻生|去死|结束生命)"),
        RiskLevel.IMMINENT,
    ),
    LexiconRule(
        "imminent_method_possession",
        _p(r"(已经|我)?(准备好|买好|攒好|备好|拿到)了?(药|安眠药|刀|绳子|木炭|农药)"),
        RiskLevel.IMMINENT,
    ),
    LexiconRule(
        "imminent_in_progress",
        _p(r"(我)?(正在|刚刚|已经)(吃|服|割|跳|上吊|喝)(了)?(药|腕|楼|安眠药|农药)"),
        RiskLevel.IMMINENT,
    ),
    LexiconRule(
        "imminent_farewell",
        _p(r"(遗书|遗言|最后一条消息|别找我了|永别)"),
        RiskLevel.IMMINENT,
    ),
)

# HIGH：明确的自杀意念（但未表明即刻实施）
_HIGH_RULES: tuple[LexiconRule, ...] = (
    LexiconRule(
        "high_suicidal_ideation",
        _p(r"(想|要|打算|计划)(去)?(自杀|自尽|轻生|死掉|结束生命)"),
        RiskLevel.HIGH,
    ),
    LexiconRule(
        "high_want_to_die",
        _p(r"(我)?(想死|不想活|不想活了|活不下去|活够了|想消失)"),
        RiskLevel.HIGH,
    ),
    LexiconRule(
        "high_self_harm",
        _p(r"(自杀|自尽|自残|自伤|轻生|割腕|上吊|跳楼|服毒)"),
        RiskLevel.HIGH,
    ),
    LexiconRule(
        "high_harm_others",
        _p(r"(想|要|打算)(杀|弄死|伤害|报复)(他|她|他们|人|全家)"),
        RiskLevel.HIGH,
    ),
    # 繁体不需要单列规则：normalize() 已用 zhconv 全量转简，
    # 「自殺/輕生/撐不下去」等一律折叠到上面的简体规则上。
)

# MEDIUM：无望感、消失念头、严重物质滥用 —— 需要关怀性介入但不终止会话
_MEDIUM_RULES: tuple[LexiconRule, ...] = (
    LexiconRule(
        "medium_hopeless",
        _p(r"(没有希望|毫无意义|撑不下去|坚持不住|熬不下去|看不到未来)"),
        RiskLevel.MEDIUM,
    ),
    LexiconRule(
        "medium_worthless",
        _p(r"(我是|自己是)?(废物|累赘|多余的|负担|一无是处)"),
        RiskLevel.MEDIUM,
    ),
    LexiconRule("medium_disappear", _p(r"(消失|蒸发|从这个世界)(掉|了)?"), RiskLevel.MEDIUM),
    LexiconRule(
        "medium_substance",
        _p(r"(酗酒|喝到断片|嗑药|吸毒|滥用药物|靠药才能)"),
        RiskLevel.MEDIUM,
    ),
    LexiconRule("medium_abuse", _p(r"(被打|被虐待|家暴|被侵犯|被欺凌|被霸凌)"), RiskLevel.MEDIUM),
    LexiconRule("medium_self_hate", _p(r"(恨自己|讨厌自己|活着就是错)"), RiskLevel.MEDIUM),
)

ALL_RULES: tuple[LexiconRule, ...] = _IMMINENT_RULES + _HIGH_RULES + _MEDIUM_RULES


@dataclass(frozen=True, slots=True)
class LexiconHit:
    level: RiskLevel
    rule_names: tuple[str, ...]
    matched_text: str | None


def scan(text: str) -> LexiconHit | None:
    """扫描文本。命中则返回最高风险等级与全部命中规则名；未命中返回 None。

    纯函数、无副作用、可完全离线单测。
    """
    if not text:
        return None

    norm = normalize(text)
    hits: list[tuple[LexiconRule, str]] = []

    for rule in ALL_RULES:
        m = rule.pattern.search(norm)
        if m:
            hits.append((rule, m.group(0)))

    if not hits:
        return None

    from app.schemas.enums import RISK_ORDER

    hits.sort(key=lambda h: RISK_ORDER[h[0].level], reverse=True)
    top_level = hits[0][0].level
    return LexiconHit(
        level=top_level,
        rule_names=tuple(r.name for r, _ in hits),
        matched_text=hits[0][1],
    )


def is_suspicious(text: str) -> bool:
    """未命中词表但语义可疑 —— 决定是否调用第二层分类器（§3.4 控制成本）。

    保守设计：宁可多调一次分类器，也不要漏检。
    """
    if not text:
        return False
    norm = normalize(text)
    weak_signals = (
        "累", "崩", "撑", "扛不住", "喘不过气", "麻木", "空", "孤独",
        "绝望", "痛苦", "折磨", "受不了", "解脱", "离开", "结束",
        "眼泪", "失眠", "吃不下", "没意思",
    )
    return any(sig in norm for sig in weak_signals)
