"""测试装配（路线文档 §12）。

FakeLLM 是整套测试的地基：让状态机测试确定、快速、免费。
这里提供构造**合法**领域输出的辅助函数 ——
故意不用随机数据：规划必须满足 §1.4 的维度约束，
否则测的是"校验器能不能拒绝"，而不是"流程能不能跑通"。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.domain.dimension import Dimension
from app.domain.mode import AskMode, batch_bounds
from app.domain.technique import Technique
from app.graph.deps import Deps
from app.graph.state import GraphState
from app.llm.base import TaskKind
from app.llm.gateway import LLMGateway, Route
from app.llm.providers.fake import FakeProvider, Scripted
from app.safety.resources import load_crisis_resources
from app.schemas.enums import ConsentScope, Phase, Sharpness
from app.schemas.session import Consent

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 一份满足 §1.4 全部约束的 10 问规划（MODERATE 锐度、非危机）：
#   段内维度合法 · 相邻不重复 · 覆盖 8 个维度 · 技术与维度/锐度匹配
VALID_PLAN: list[tuple[Dimension, Technique]] = [
    (Dimension.FACT, Technique.EXCEPTION),
    (Dimension.EMOTION, Technique.SCALING),
    (Dimension.RELATION, Technique.CIRCULAR),
    (Dimension.VALUE, Technique.VALUE_PROBE),
    (Dimension.FEAR, Technique.REFRAME),
    (Dimension.COST, Technique.COST_OF_LOSS),
    (Dimension.IDENTITY, Technique.MIRACLE),
    (Dimension.RESOURCE, Technique.EXCEPTION),
    (Dimension.VALUE, Technique.VALUE_PROBE),
    (Dimension.IDENTITY, Technique.CIRCULAR),
]

# 温和版规划：危机模式/未成年人下可用。
# 约束：不含 FEAR/IDENTITY（深度维度被 filter_for_safety 禁用），
# 且只用 crisis_safe 且 min_sharpness=1 的技术。
# COST 维度靠 SCALING 承载 —— 量尺只要求校准，不迫使对方正面设想丧失。
GENTLE_PLAN: list[tuple[Dimension, Technique]] = [
    (Dimension.FACT, Technique.EXCEPTION),
    (Dimension.EMOTION, Technique.SCALING),
    (Dimension.FACT, Technique.SCALING),
    (Dimension.VALUE, Technique.VALUE_PROBE),
    (Dimension.COST, Technique.SCALING),
    (Dimension.VALUE, Technique.MIRACLE),
    (Dimension.COST, Technique.SCALING),
    (Dimension.RESOURCE, Technique.EXCEPTION),
    (Dimension.VALUE, Technique.VALUE_PROBE),
    (Dimension.RESOURCE, Technique.MIRACLE),
]


def plan_json(lo: int, hi: int, table: list[tuple[Dimension, Technique]] | None = None) -> str:
    """构造 [lo, hi] 区间的合法 PlannerOutput JSON。"""
    src = table or VALID_PLAN
    slots = [
        {
            "index": i,
            "dimension": src[i - 1][0].value,
            "technique": src[i - 1][1].value,
            "intent": f"探索第{i}问的方向",
            "depends_on": [],
        }
        for i in range(lo, hi + 1)
    ]
    return json.dumps({"slots": slots, "rationale": "测试规划"}, ensure_ascii=False)


def script_plans(fake: FakeProvider, mode: AskMode, table=None) -> None:
    """按模式把各批次的规划响应排入队列。"""
    for lo, hi in batch_bounds(mode):
        fake.script(TaskKind.PLAN, Scripted(text=plan_json(lo, hi, table)))


def script_questions(fake: FakeProvider, count: int) -> None:
    for i in range(count):
        fake.script(
            TaskKind.ASK,
            Scripted(
                text=json.dumps(
                    {"text": f"第{i + 1}个测试问题，你怎么看？"}, ensure_ascii=False
                )
            ),
        )


def script_quality(fake: FakeProvider, count: int, score: float = 0.85) -> None:
    for _ in range(count):
        fake.script(
            TaskKind.QUALITY,
            Scripted(
                text=json.dumps(
                    {
                        "score": score,
                        "is_evasive": False,
                        "is_too_brief": False,
                        "missing_aspects": [],
                    }
                )
            ),
        )


#: 测试用回答。Analyst 的 Evidence 引用必须是它的**逐字子串**。
ANSWER_TEMPLATE = "我最怕失去现在稳定的生活，但又不甘心一直这样耗下去，第{n}轮的想法。"
QUOTE_A = "我最怕失去现在稳定的生活"
QUOTE_B = "不甘心一直这样耗下去"


def answer_text(n: int) -> str:
    return ANSWER_TEMPLATE.format(n=n)


def script_analysis(fake: FakeProvider, *, approved: bool = True) -> None:
    """排入三段式分析（§6.5）+ Critic 复核。"""
    fake.script(
        TaskKind.ANALYZE,
        Scripted(
            text=json.dumps(
                {
                    "surface_concern": "在稳定与改变之间摇摆",
                    "underlying_needs": [
                        {
                            "statement": "希望在不失去安全感的前提下获得成长",
                            "evidence": [{"question_index": 1, "quote": QUOTE_A}],
                            "confidence": "medium",
                        }
                    ],
                    "uncertainty_notes": [],
                },
                ensure_ascii=False,
            )
        ),
        Scripted(
            text=json.dumps(
                {
                    "inner_conflicts": [
                        {
                            "statement": "求稳与求变同时存在",
                            "evidence": [
                                {"question_index": 1, "quote": QUOTE_A},
                                {"question_index": 2, "quote": QUOTE_B},
                            ],
                            "confidence": "high",
                        }
                    ],
                    "uncertainty_notes": [],
                },
                ensure_ascii=False,
            )
        ),
        Scripted(
            text=json.dumps(
                {
                    "suggestions": [
                        {
                            "action": "把担心的具体内容写下来排序",
                            "first_step": "今晚睡前用十分钟列出最怕失去的三件事",
                            "linked_insights": [0, 1],
                        }
                    ],
                    "uncertainty_notes": [],
                },
                ensure_ascii=False,
            )
        ),
    )
    fake.script(
        TaskKind.CRITIQUE,
        Scripted(text=json.dumps({"approved": approved, "issues": []}, ensure_ascii=False)),
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        policy_version="test-policy-1",
        llm_api_key="test",
        llm_model_strong="test-strong",
        llm_model_fast="test-fast",
        crisis_resources_file=PROJECT_ROOT / "config" / "crisis_resources.yaml",
    )


@pytest.fixture
def fake() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def gateway(fake: FakeProvider) -> LLMGateway:
    return LLMGateway(routes=[Route(provider=fake, label="primary")])


@pytest.fixture
def deps(settings: Settings, gateway: LLMGateway) -> Deps:
    return Deps.build(
        settings=settings,
        gateway=gateway,
        prompts_root=PROJECT_ROOT / "prompts",
        crisis=load_crisis_resources(settings.crisis_resources_file),
    )


def make_state(
    *,
    mode: AskMode = AskMode.BATCHED,
    sharpness: Sharpness = Sharpness.MODERATE,
    with_consent: bool = True,
    concern: str = "我在纠结是否要离职创业",
    phase: Phase = Phase.INTAKE,
) -> GraphState:
    from datetime import UTC, datetime

    consent = (
        Consent(
            policy_version="test-policy-1",
            granted_at=datetime.now(UTC),
            scopes={ConsentScope.STORE_MEMORY},
        )
        if with_consent
        else None
    )
    return GraphState(
        session_id="s-test",
        user_id="u-test",
        phase=phase,
        mode=mode,
        sharpness=sharpness,
        concern=concern if phase is not Phase.INTAKE else "",
        consent=consent,
        inbox=concern if phase is Phase.INTAKE else None,
    )
