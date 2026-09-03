"""跨模块共享的枚举（路线文档 §3.4 / §3.5 / §4.3 / §3.2）。"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class RiskLevel(StrEnum):
    """危机风险等级（§3.4）。顺序有意义，用 `RISK_ORDER` 比较。"""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMINENT = "imminent"


RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.IMMINENT: 4,
}


def risk_at_least(level: RiskLevel, threshold: RiskLevel) -> bool:
    return RISK_ORDER[level] >= RISK_ORDER[threshold]


class RiskSource(StrEnum):
    LEXICON = "lexicon"        # 规则层（零 LLM 依赖，§3.4 第一层）
    CLASSIFIER = "classifier"  # 模型层
    MODERATION = "moderation"  # 供应商审核（可选第三信号）
    UNAVAILABLE = "unavailable"  # 分类器不可用，保守处理（§9.3）


class Sharpness(IntEnum):
    """提问锐度（§3.5）。数值可比较，用于与 TechniqueSpec.min_sharpness 对齐。"""

    GENTLE = 1
    MODERATE = 2
    SHARP = 3


DEFAULT_SHARPNESS = Sharpness.MODERATE


class Phase(StrEnum):
    """会话状态机阶段（§4.3）。"""

    INTAKE = "intake"
    CONSENT = "consent"
    PLAN = "plan"
    ASK = "ask"
    LISTEN = "listen"
    PROBE = "probe"
    ANALYZE = "analyze"
    CRITIC = "critic"
    REPORT = "report"
    CRISIS = "crisis"
    PAUSED = "paused"
    DONE = "done"


TERMINAL_PHASES = frozenset({Phase.CRISIS, Phase.DONE})


class ConsentScope(StrEnum):
    """分项授权（§3.2）。各自独立可拒绝，拒绝后功能降级而非拒绝服务。"""

    STORE_MEMORY = "store_memory"
    CROSS_SESSION_RECALL = "cross_session_recall"
    QUALITY_REVIEW = "quality_review"
    EVAL_USE = "eval_use"


class SafetyActionKind(StrEnum):
    """处置矩阵的动作（§3.4）。由 policy 纯函数决定，不由 LLM 自由决定。"""

    PROCEED = "proceed"                  # 正常提问
    SOFTEN = "soften"                    # 降锐度 + 关怀性回应 + 提示资源
    STOP_WITH_RESOURCES = "stop"         # 立即终止提问，输出危机资源
    HOLD_UNVERIFIABLE = "hold"           # 无法评估安全时的保守暂停（§9.3）


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DegradeReason(StrEnum):
    """降级原因（§9.5：任何影响输出质量的降级都必须可见）。"""

    LLM_FALLBACK_MODEL = "llm_fallback_model"
    LLM_FALLBACK_PROVIDER = "llm_fallback_provider"
    CAPABILITY_INSUFFICIENT = "capability_insufficient"
    CONTEXT_TRUNCATED = "context_truncated"
    RETRIEVAL_SKIPPED = "retrieval_skipped"
    RERANKER_TIMEOUT = "reranker_timeout"
    EMBEDDING_UNAVAILABLE = "embedding_unavailable"
    EMBEDDING_ROUTED_TO_CLOUD = "embedding_routed_to_cloud"  # 隐私相关，需审计（§3.3.1）
    REPORT_SECTION_FAILED = "report_section_failed"
    EVIDENCE_DROPPED = "evidence_dropped"
    SAFETY_CLASSIFIER_UNAVAILABLE = "safety_classifier_unavailable"
    QUESTION_GUARD_FALLBACK = "question_guard_fallback"  # 提问反复违规，降级到安全模板（§3.7）
    CRITIC_UNRESOLVED = "critic_unresolved"              # Critic 连续否决，输出带不确定性标注的报告
    INJECTION_STRIPPED = "injection_stripped"            # 用户输入中剥离了疑似注入指令（§3.7）
    HISTORY_COMPRESSED = "history_compressed"            # 早期问答被压成摘要（§6.3，保真度下降）
