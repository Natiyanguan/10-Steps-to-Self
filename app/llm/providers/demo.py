"""DemoProvider —— 产出**结构合法**的假响应（路线文档 §12 / §13.2.1）。

与 `FakeProvider` 的分工：
  - `FakeProvider`：测试用，逐条编程预设响应（含故障注入）。
  - `DemoProvider`：跑通流程用，自动产出符合 schema 的响应。
    `tss chat --fake` 与 CI 的 `local` 档冒烟都用它。

它**从提示词里解析约束**（允许的维度、可用技术、用户原话），
而不是硬编码一份答案。因此它同时验证了两件事：
  1. 提示词真的把约束传达清楚了（否则解析不到）
  2. 领域约束是**可满足的** —— 存在至少一个合法解

内容质量当然没有意义（它不是模型），所以断言只针对流程与 schema，
绝不针对内容 —— 这正是 §2.5 对 TOY 档的定位。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from app.llm.base import ChatRequest, ChatResponse, TaskKind, Usage
from app.llm.capability import ModelTier, ProviderCaps, ZhQuality

_RE_RANGE = re.compile(r"需要规划第\s*(\d+)\s*到第\s*(\d+)\s*问")
_RE_SLOT = re.compile(r"^-\s*第\s*(\d+)\s*问（\S+?\s*段）：([^\n]+)$", re.MULTILINE)
#: 维度 → 配套技术。Demo 必须解析这个配对关系，
#: 因此如果提示词没把配对讲清楚，冒烟测试会立刻失败 —— 这正是我们要的信号。
_RE_DIM_GUIDE = re.compile(
    r"^-\s*`([a-z_]+)`（[^\n]*\n\s*可用技术：([^\n]+)$", re.MULTILINE
)
_RE_BACKTICKED = re.compile(r"`([a-z_]+)`")
_RE_ANSWER = re.compile(r"^答：(.+)$", re.MULTILINE)
_RE_QIDX = re.compile(r"【第\s*(\d+)\s*问】")

MIN_QUOTE_LEN = 8


def demo_caps() -> ProviderCaps:
    """声明为 FAST 档：能跑通 Safety/Quality/Summary，但**不能**承担 STRONG 任务。

    这正是本地小模型的真实定位（§2.5）。若要用它跑完整流程演示，
    需显式提升档位 —— 见 `demo_caps_strong`，仅用于离线冒烟。
    """
    return ProviderCaps(
        tier=ModelTier.FAST,
        context_window=32_000,
        max_output_tokens=2_048,
        supports_json_mode=True,
        supports_temperature=True,
        supports_streaming=True,
        zh_quality=ZhQuality.MEDIUM,
        tokenizer_hint="local",
        cost_per_1k_in=0.0,
        cost_per_1k_out=0.0,
        probed_at=datetime.now(UTC),
    )


def demo_caps_strong() -> ProviderCaps:
    """仅用于离线跑通全流程（CLI --fake / CI 冒烟）。

    真实本地小模型**不应**这样声明 —— 那正是 §2.4.3 要防的事。
    这里是"假模型假装够强"，而不是"弱模型被当成强模型"：
    它的输出不进任何真实用户的报告。
    """
    caps = demo_caps()
    return ProviderCaps(
        tier=ModelTier.STRONG,
        context_window=64_000,
        max_output_tokens=4_096,
        supports_json_mode=True,
        supports_temperature=caps.supports_temperature,
        supports_streaming=True,
        zh_quality=ZhQuality.HIGH,
        tokenizer_hint="local",
        cost_per_1k_in=0.0,
        cost_per_1k_out=0.0,
        probed_at=datetime.now(UTC),
    )


class DemoProvider:
    """按任务类型产出结构合法的假响应。"""

    def __init__(self, *, name: str = "demo", caps: ProviderCaps | None = None) -> None:
        self._name = name
        self._caps = caps or demo_caps_strong()
        self._emitted_dims: list[str] = []  # 跨批次记忆，保证相邻不重复
        self.calls: list[TaskKind] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def caps(self) -> ProviderCaps:
        return self._caps

    # ---- 各任务的响应构造 ----

    def _plan(self, prompt: str) -> str:
        m = _RE_RANGE.search(prompt)
        lo, hi = (int(m.group(1)), int(m.group(2))) if m else (1, 3)

        # 各槽位允许的维度
        allowed_by_index: dict[int, list[str]] = {}
        for idx_s, dims_s in _RE_SLOT.findall(prompt):
            allowed_by_index[int(idx_s)] = _RE_BACKTICKED.findall(dims_s)

        # 维度 → 配套技术（提示词若没讲清配对，这里会拿到空表 → 规划失败）
        techs_by_dim: dict[str, list[str]] = {
            dim: _RE_BACKTICKED.findall(techs)
            for dim, techs in _RE_DIM_GUIDE.findall(prompt)
        }

        counts = Counter(self._emitted_dims)
        slots: list[dict[str, object]] = []

        for i in range(lo, hi + 1):
            options = [d for d in allowed_by_index.get(i, []) if techs_by_dim.get(d)]
            if not options:
                # 解析不到可用配对：宁可产出明显非法的规划让校验器拒绝，
                # 也不要静默编一个看似合法的答案 —— 后者会掩盖提示词的问题。
                return json.dumps(
                    {
                        "slots": [],
                        "rationale": "demo 无法从提示词解析出维度与技术的配对关系",
                    },
                    ensure_ascii=False,
                )

            prev = self._emitted_dims[-1] if self._emitted_dims else None
            # 选"未与上一问重复且此前用得最少"的维度 —— 自然最大化覆盖度
            candidates = [d for d in options if d != prev] or options
            chosen = min(candidates, key=lambda d: (counts[d], options.index(d)))
            counts[chosen] += 1
            self._emitted_dims.append(chosen)

            # 技术只能从该维度自己的列表里取（这是本次修复的核心约束）
            pool = techs_by_dim[chosen]
            technique = pool[(i - 1) % len(pool)]

            slots.append(
                {
                    "index": i,
                    "dimension": chosen,
                    "technique": technique,
                    "intent": f"了解第{i}问在{chosen}维度上的具体情况",
                    "depends_on": [],
                }
            )

        return json.dumps(
            {"slots": slots, "rationale": "演示用规划：按段内允许维度轮换，技术取自该维度配对表"},
            ensure_ascii=False,
        )

    @staticmethod
    def _ask(prompt: str) -> str:
        m = re.search(r"这是第\s*(\d+)\s*问", prompt)
        idx = m.group(1) if m else "1"
        return json.dumps(
            {"text": f"（演示第{idx}问）关于这件事，现在最占据你注意力的是哪一部分？"},
            ensure_ascii=False,
        )

    @staticmethod
    def _quality() -> str:
        return json.dumps(
            {"score": 0.8, "is_evasive": False, "is_too_brief": False, "missing_aspects": []}
        )

    @staticmethod
    def _safety() -> str:
        # 规则层已在 LLM 之前跑过；这里只覆盖"语义可疑但未命中词表"的情况
        return json.dumps(
            {"level": "none", "reason": "演示模式默认无风险", "self_referential": True}
        )

    @staticmethod
    def _quotes(prompt: str) -> list[tuple[int, str]]:
        """从提示词里提取真实的用户原话片段。

        这样 Evidence 校验（§6.4 第三层）走的是真实路径 ——
        如果这里取错了，测试会以"洞察被丢弃"的形式告诉我们。
        """
        indices = [int(x) for x in _RE_QIDX.findall(prompt)]
        answers = _RE_ANSWER.findall(prompt)
        out: list[tuple[int, str]] = []
        for i, ans in enumerate(answers):
            text = ans.strip()
            if len(text) < MIN_QUOTE_LEN:
                continue
            qidx = indices[i] if i < len(indices) else i + 1
            out.append((qidx, text[:MIN_QUOTE_LEN + 6]))
        return out

    def _analyze(self, prompt: str) -> str:
        quotes = self._quotes(prompt)
        fallback = [(1, "（演示模式无可引用原话）")]
        picked = quotes or fallback

        if "inner_conflicts" in prompt:
            if len(picked) < 2:
                return json.dumps(
                    {"inner_conflicts": [], "uncertainty_notes": ["演示模式：材料不足以识别冲突"]},
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "inner_conflicts": [
                        {
                            "statement": "（演示）两种期待同时存在，彼此拉扯",
                            "evidence": [
                                {"question_index": picked[0][0], "quote": picked[0][1]},
                                {"question_index": picked[1][0], "quote": picked[1][1]},
                            ],
                            "confidence": "medium",
                        }
                    ],
                    "uncertainty_notes": [],
                },
                ensure_ascii=False,
            )

        if "suggestions" in prompt:
            m = re.search(r"只能用上面真实出现的编号（0 到 (\d+)）", prompt)
            max_idx = int(m.group(1)) if m else 0
            return json.dumps(
                {
                    "suggestions": [
                        {
                            "action": "（演示）把顾虑写下来并排序",
                            "first_step": "今晚睡前用十分钟，把最在意的三件事写在纸上",
                            "linked_insights": list(range(min(max_idx + 1, 2))),
                        }
                    ],
                    "uncertainty_notes": [],
                },
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "surface_concern": "（演示）你描述的困惑摘要",
                "underlying_needs": [
                    {
                        "statement": "（演示）希望在不失去安全感的前提下有所改变",
                        "evidence": [
                            {"question_index": picked[0][0], "quote": picked[0][1]}
                        ],
                        "confidence": "medium",
                    }
                ],
                "uncertainty_notes": ["演示模式：内容由假模型产出，不具备分析意义"],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _critique() -> str:
        return json.dumps({"approved": True, "issues": []}, ensure_ascii=False)

    @staticmethod
    def _summarize() -> str:
        return json.dumps(
            {"by_dimension": [], "preserved_quotes": []}, ensure_ascii=False
        )

    # ---- provider 协议 ----

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.calls.append(req.task)
        prompt = "\n".join(m.content for m in req.messages)

        builders = {
            TaskKind.PLAN: lambda: self._plan(prompt),
            TaskKind.ASK: lambda: self._ask(prompt),
            TaskKind.PROBE: lambda: self._ask(prompt),
            TaskKind.QUALITY: self._quality,
            TaskKind.SAFETY: self._safety,
            TaskKind.ANALYZE: lambda: self._analyze(prompt),
            TaskKind.CRITIQUE: self._critique,
            TaskKind.SUMMARIZE: self._summarize,
        }
        text = builders.get(req.task, lambda: "{}")()

        return ChatResponse(
            text=text,
            usage=Usage(prompt_tokens=len(prompt) // 3, completion_tokens=len(text) // 3),
            model="demo",
            provider=self._name,
            finish_reason="stop",
            latency_ms=1.0,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[str]:
        resp = await self.complete(req)
        for i in range(0, len(resp.text), 16):
            yield resp.text[i : i + 16]

    async def aclose(self) -> None:
        return None
