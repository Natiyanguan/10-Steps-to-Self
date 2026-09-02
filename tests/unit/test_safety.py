"""安全层单测（路线文档 §12 / §3.4 / §6.4）。

全部零 LLM 依赖：这层必须在 LLM 完全不可用时仍然工作（§9.3）。
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from app.safety import lexicon, policy
from app.safety.evidence import (
    DEFAULT_THRESHOLD,
    normalize_for_match,
    validate_suggestion_links,
    verify_insights,
    verify_quote,
)
from app.safety.output_guard import (
    ViolationKind,
    check_question,
    check_report_text,
    safe_fallback,
)
from app.schemas.enums import (
    RiskLevel,
    RiskSource,
    SafetyActionKind,
    Sharpness,
    risk_at_least,
)
from app.schemas.session import Evidence, Insight, RiskAssessment, SessionState


def _state(**kw) -> SessionState:
    base = {"session_id": "s", "user_id": "u"}
    return SessionState(**{**base, **kw})


def _risk(level: RiskLevel, source: RiskSource = RiskSource.LEXICON) -> RiskAssessment:
    return RiskAssessment(level=level, source=source)


class TestLexicon:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("我想自杀", RiskLevel.HIGH),
            ("我不想活了", RiskLevel.HIGH),
            ("我想割腕", RiskLevel.HIGH),
            ("现在就去死", RiskLevel.IMMINENT),
            ("已经买好了药", RiskLevel.IMMINENT),
            ("我写好遗书了", RiskLevel.IMMINENT),
            ("撑不下去了", RiskLevel.MEDIUM),
            ("我就是个废物", RiskLevel.MEDIUM),
            ("我被家暴了", RiskLevel.MEDIUM),
            ("最近工作有点忙", None),
            ("今天天气不错", None),
            ("", None),
        ],
    )
    def test_levels(self, text: str, expected: RiskLevel | None) -> None:
        hit = lexicon.scan(text)
        assert (hit.level if hit else None) == expected

    @pytest.mark.parametrize(
        "variant",
        [
            "我想自杀",
            "我 想 自 杀",
            "我想自．杀",
            "我想自-杀",
            "我想自sha",
            "我想自殺",       # 繁体（zhconv 转换）
            "我想自　杀",     # 全角空格（NFKC）
            "我 想 自 殺 ",
        ],
    )
    def test_evasion_variants_all_caught(self, variant: str) -> None:
        """变体规避必须失效 —— 假阴性等于漏检危机（§3.4）。"""
        hit = lexicon.scan(variant)
        assert hit is not None, f"未命中：{variant!r}"
        assert risk_at_least(hit.level, RiskLevel.HIGH)

    def test_normalize_strips_noise(self) -> None:
        assert lexicon.normalize("自 杀") == lexicon.normalize("自杀")
        assert lexicon.normalize("撐不下去") == lexicon.normalize("撑不下去")

    def test_highest_level_wins(self) -> None:
        """同时命中多条时取最高等级。"""
        hit = lexicon.scan("我撑不下去了，已经买好了药")
        assert hit is not None
        assert hit.level is RiskLevel.IMMINENT
        assert len(hit.rule_names) >= 2

    def test_is_suspicious_triggers_classifier(self) -> None:
        """未命中词表但语义可疑 → 应调分类器（宁可多调，不可漏检）。"""
        assert lexicon.is_suspicious("感觉很麻木，什么都没意思")
        assert lexicon.is_suspicious("我好想解脱")
        assert not lexicon.is_suspicious("明天要开会")
        assert not lexicon.is_suspicious("")

    def test_scan_is_pure(self) -> None:
        """纯函数：同输入同输出，无副作用。"""
        text = "我想自杀"
        a, b = lexicon.scan(text), lexicon.scan(text)
        assert a == b


class TestPolicy:
    @pytest.mark.parametrize("level", [RiskLevel.HIGH, RiskLevel.IMMINENT])
    def test_high_risk_stops_immediately(self, level: RiskLevel) -> None:
        action = policy.decide(_risk(level), _state())
        assert action.kind is SafetyActionKind.STOP_WITH_RESOURCES
        assert action.is_stop
        assert not action.allow_further_questions
        assert action.show_crisis_resources
        assert action.block_deep_dimensions

    def test_medium_softens_but_continues(self) -> None:
        action = policy.decide(_risk(RiskLevel.MEDIUM), _state())
        assert action.kind is SafetyActionKind.SOFTEN
        assert action.allow_further_questions, "medium 不应终止会话"
        assert action.show_crisis_resources, "medium 应提示专业资源"
        assert action.max_sharpness is Sharpness.GENTLE
        assert action.block_deep_dimensions

    @pytest.mark.parametrize("level", [RiskLevel.NONE, RiskLevel.LOW])
    def test_low_risk_proceeds(self, level: RiskLevel) -> None:
        action = policy.decide(_risk(level), _state(sharpness=Sharpness.SHARP))
        assert action.kind is SafetyActionKind.PROCEED
        assert action.max_sharpness is Sharpness.SHARP
        assert not action.crisis_mode

    def test_classifier_unavailable_holds_conservatively(self) -> None:
        """无法评估安全时绝不继续深度提问（§9.3）。"""
        action = policy.decide(
            _risk(RiskLevel.LOW, RiskSource.UNAVAILABLE), _state(sharpness=Sharpness.SHARP)
        )
        assert action.kind is SafetyActionKind.HOLD_UNVERIFIABLE
        assert not action.allow_further_questions
        assert action.max_sharpness is Sharpness.GENTLE
        assert action.block_deep_dimensions

    def test_minor_forced_gentle(self) -> None:
        """未成年人强制 GENTLE，不可上调（§3.2 硬约束）。"""
        action = policy.decide(
            _risk(RiskLevel.NONE), _state(is_minor=True, sharpness=Sharpness.SHARP)
        )
        assert action.max_sharpness is Sharpness.GENTLE
        assert action.block_deep_dimensions
        assert any("未成年" in r for r in action.reasons)

    def test_sharpness_cap_is_one_way(self) -> None:
        """锐度只能下调不能上调 —— 安全侧单向棘轮。"""
        gentle_action = policy.decide(_risk(RiskLevel.MEDIUM), _state())
        assert policy.apply_sharpness_cap(Sharpness.SHARP, gentle_action) is Sharpness.GENTLE

        proceed = policy.decide(_risk(RiskLevel.NONE), _state(sharpness=Sharpness.SHARP))
        # 当前已是 GENTLE 时，PROCEED 不应把它拉高
        assert policy.apply_sharpness_cap(Sharpness.GENTLE, proceed) is Sharpness.GENTLE

    def test_crisis_mode_flag(self) -> None:
        assert policy.decide(_risk(RiskLevel.MEDIUM), _state()).crisis_mode
        assert not policy.decide(_risk(RiskLevel.NONE), _state()).crisis_mode

    def test_detect_evasion_needs_history(self) -> None:
        assert policy.detect_evasion(_state(), "短") is None

    def test_detect_evasion_on_length_drop(self) -> None:
        from app.domain.technique import Technique
        from app.schemas.session import Answer, QAPair, Question

        pairs = [
            QAPair(
                question=Question(
                    index=i, text="问题", technique=Technique.SCALING, sharpness=Sharpness.MODERATE
                ),
                answer=Answer(question_index=i, text="这是一个比较长的回答" * 5),
            )
            for i in (1, 2)
        ]
        state = _state(qa_pairs=pairs)
        assert policy.detect_evasion(state, "还行") is not None
        assert policy.detect_evasion(state, "这是一个比较长的回答" * 5) is None


class TestEvidence:
    USER: ClassVar[list[str]] = ["其实我最怕失去现在稳定的生活，但又不甘心一直这样耗下去。"]

    @pytest.mark.parametrize(
        ("quote", "expected"),
        [
            ("我最怕失去现在稳定的生活", True),
            ("不甘心一直这样耗下去", True),
            ("我最怕失去现在稳定的生活。", True),     # 多标点，应容忍
            ("我最怕失去現在穩定的生活", True),       # 繁体，应容忍
            ("我想创业赚大钱", False),               # 编造
            ("他觉得我应该辞职", False),             # 编造
            ("稳定", False),                        # 过短
            ("", False),
            ("。。。", False),                       # 只有标点
        ],
    )
    def test_verify_quote(self, quote: str, expected: bool) -> None:
        assert verify_quote(quote, self.USER).ok is expected

    def test_normalize_for_match(self) -> None:
        assert normalize_for_match("我最怕，失去！") == normalize_for_match("我最怕失去")
        assert normalize_for_match("穩定") == normalize_for_match("稳定")

    def test_insight_with_all_bad_evidence_dropped(self) -> None:
        """全部依据失效 → 丢弃整条洞察，不保留待人工判断（§6.4）。"""
        bad = Insight(
            statement="编造",
            evidence=[Evidence(question_index=1, quote="我想去环游世界")],
        )
        out = verify_insights([bad], user_texts=self.USER, answered_indices={1})
        assert not out.kept
        assert len(out.dropped) == 1
        assert out.pass_rate == 0.0

    def test_insight_keeps_only_verified_evidence(self) -> None:
        """部分依据失效 → 保留洞察但只留通过的依据。"""
        mixed = Insight(
            statement="重视稳定",
            evidence=[
                Evidence(question_index=1, quote="我最怕失去现在稳定的生活"),
                Evidence(question_index=1, quote="我想去环游世界"),
            ],
        )
        out = verify_insights([mixed], user_texts=self.USER, answered_indices={1})
        assert len(out.kept) == 1
        assert len(out.kept[0].evidence) == 1

    def test_nonexistent_question_index_rejected(self) -> None:
        ins = Insight(
            statement="X",
            evidence=[Evidence(question_index=9, quote="我最怕失去现在稳定的生活")],
        )
        out = verify_insights([ins], user_texts=self.USER, answered_indices={1, 2})
        assert not out.kept
        assert "不存在" in out.dropped[0].reason

    def test_pass_rate_empty_is_one(self) -> None:
        out = verify_insights([], user_texts=self.USER)
        assert out.pass_rate == 1.0

    def test_threshold_is_strict(self) -> None:
        """阈值必须足够严 —— 语义改写不能通过。"""
        assert DEFAULT_THRESHOLD >= 0.9
        # 语序调换属于改写，应拒绝
        assert not verify_quote("稳定的生活我最怕失去", self.USER).ok

    def test_validate_suggestion_links(self) -> None:
        assert validate_suggestion_links([0, 1, 5], 2) == [0, 1]
        assert validate_suggestion_links([-1, 3], 3) == []
        assert validate_suggestion_links([], 3) == []


class TestOutputGuard:
    @pytest.mark.parametrize(
        "text",
        [
            "你最怕失去什么？",
            "如果最了解你的人来回答，他会怎么说？",
            "0 到 10 分，现在大概几分？",
        ],
    )
    def test_valid_questions_pass(self, text: str) -> None:
        assert check_question(text).ok

    @pytest.mark.parametrize(
        ("text", "kind"),
        [
            ("我建议你先辞职试试。", ViolationKind.ADVICE_NOT_QUESTION),
            ("你应该和家人好好谈谈。", ViolationKind.ADVICE_NOT_QUESTION),
            ("难道你不觉得应该辞职吗？", ViolationKind.LEADING_QUESTION),
            ("既然这么痛苦，为什么不离开呢？", ViolationKind.LEADING_QUESTION),
            ("你这是典型的焦虑症。", ViolationKind.MEDICAL_CLAIM),
            ("你患有抑郁症，需要治疗。", ViolationKind.MEDICAL_CLAIM),
            ("放心，一定会好起来的。", ViolationKind.ABSOLUTE_PROMISE),
            ("作为一名执业心理咨询师，我认为…", ViolationKind.IMPERSONATION),
            ("建议你去停药观察一下。", ViolationKind.OUT_OF_SCOPE_ADVICE),
        ],
    )
    def test_violations_caught(self, text: str, kind: ViolationKind) -> None:
        result = check_question(text)
        assert not result.ok, f"应拦截：{text}"
        assert kind in {v.kind for v in result.violations}

    def test_non_question_rejected(self) -> None:
        """提问阶段的输出必须是问句。"""
        result = check_question("你看起来很累。")
        assert not result.ok
        assert ViolationKind.ADVICE_NOT_QUESTION in {v.kind for v in result.violations}

    def test_report_allows_advice_but_not_medical(self) -> None:
        """报告可以给建议（那是它的职责），但仍禁止医疗断言。"""
        assert check_report_text("建议你今晚写下最怕失去的三件事。").ok
        assert not check_report_text("你患有抑郁症。").ok

    def test_safe_fallback_is_valid_question(self) -> None:
        """兜底模板自身必须过 guard —— 否则降级路径会再次失败。"""
        assert check_question(safe_fallback()).ok
        assert check_question(safe_fallback("恐惧")).ok
