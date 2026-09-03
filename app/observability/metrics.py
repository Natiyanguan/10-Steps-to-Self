"""Prometheus 指标（路线文档 §11.3）。

指标是全局注册的（prometheus_client 的 REGISTRY），
`/metrics` 端点直接 `generate_latest()` 暴露。

覆盖四类（§11.3）：技术、质量、产品、安全。
「第几问流失最多」用 `tss_answered_questions` 直方图 —— 这是最有价值的产品指标，
它直接指出哪一问设计得太重。
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---- 技术指标 ----
TURN_DURATION = Histogram(
    "tss_turn_duration_seconds",
    "单轮 graph 处理时长（按结束 phase 分桶）",
    labelnames=["phase"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 60, 120),
)
LLM_CALLS = Counter(
    "tss_llm_calls_total", "LLM 调用次数", labelnames=["task", "provider"]
)
LLM_ERRORS = Counter(
    "tss_llm_errors_total", "LLM 错误次数", labelnames=["task"]
)
LLM_RETRIES = Counter(
    "tss_llm_retries_total", "LLM 重试次数", labelnames=["task"]
)
TOKENS = Counter(
    "tss_tokens_total", "token 消耗（prompt/completion）", labelnames=["kind"]
)
DEGRADATIONS = Counter(
    "tss_degradations_total", "降级事件次数", labelnames=["reason"]
)

# ---- 质量指标 ----
EVIDENCE_PASS_RATE = Gauge(
    "tss_evidence_pass_rate", "Evidence 逐字校验通过率（0-1）"
)
CRITIC_VETOES = Counter("tss_critic_vetoes_total", "Critic 否决次数")
PROBES = Counter("tss_probes_total", "追问触发次数")
STRUCTURED_FAILURES = Counter(
    "tss_structured_failures_total", "结构化输出解析失败次数", labelnames=["task"]
)

# ---- 产品指标 ----
SESSIONS = Counter(
    "tss_sessions_total", "会话数（按结局分桶）", labelnames=["outcome"]
)
#: 放弃点分布 —— 每个会话结束时记录"答到第几问"
ANSWERED_QUESTIONS = Histogram(
    "tss_answered_questions", "会话结束前回答的正题数（放弃点分布）",
    buckets=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
)

# ---- 安全指标 ----
RISK_EVENTS = Counter(
    "tss_risk_events_total", "危机风险事件次数", labelnames=["level"]
)
SHARPNESS_DOWNGRADES = Counter(
    "tss_sharpness_downgrades_total", "锐度自动降级次数"
)
OUTPUT_GUARD_HITS = Counter(
    "tss_output_guard_hits_total", "OutputGuard 拦截次数", labelnames=["kind"]
)
INJECTION_HITS = Counter("tss_injection_hits_total", "注入剥离次数")


class Metrics:
    """便捷封装：记录高层语义，而不是让调用方手动 .labels().inc()。"""

    def record_llm_call(
        self, *, task: str, provider: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        LLM_CALLS.labels(task=task, provider=provider).inc()
        TOKENS.labels(kind="prompt").inc(prompt_tokens)
        TOKENS.labels(kind="completion").inc(completion_tokens)

    def record_llm_error(self, task: str) -> None:
        LLM_ERRORS.labels(task=task).inc()

    def record_retry(self, task: str) -> None:
        LLM_RETRIES.labels(task=task).inc()

    def record_degradation(self, reason: str) -> None:
        DEGRADATIONS.labels(reason=reason).inc()

    def record_turn(self, phase: str, duration_s: float) -> None:
        TURN_DURATION.labels(phase=phase).observe(duration_s)

    def record_session_end(self, *, answered: int, outcome: str) -> None:
        SESSIONS.labels(outcome=outcome).inc()
        ANSWERED_QUESTIONS.observe(answered)

    def record_risk(self, level: str) -> None:
        RISK_EVENTS.labels(level=level).inc()


#: 全局单例（§13.2：observability 需要跨模块共享计数）
METRICS = Metrics()
