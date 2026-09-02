"""LLM 抽象层的基础契约（路线文档 §2.1 / §2.4）。

硬约束：**模型名、供应商、endpoint 一律不出现在业务代码里**。
业务代码只说"我要做哪个任务"，由 Gateway 按配置路由到具体 provider。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class TaskKind(StrEnum):
    """任务类型 —— 业务代码只认这个，不认模型名（§2.4.1）。"""

    PLAN = "plan"                  # Planner：规划维度与技术序列
    ASK = "ask"                    # Interviewer：渲染问题
    PROBE = "probe"                # 追问生成
    ANALYZE = "analyze"            # Analyst：分析报告（分段，§6.5）
    CRITIQUE = "critique"          # Critic：复核依据
    SAFETY = "safety"              # 风险判级
    QUALITY = "quality"            # 回答质量评估
    SUMMARIZE = "summarize"        # 摘要压缩
    QUERY_REWRITE = "query_rewrite"  # HyDE / 多查询改写
    RERANK_JUDGE = "rerank_judge"  # 重排终审打分


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    task: TaskKind
    messages: tuple[Message, ...]
    json_mode: bool = False
    temperature: float | None = None
    max_output_tokens: int | None = None
    timeout_s: float | None = None
    prompt_version: str | None = None

    def with_messages(self, messages: tuple[Message, ...]) -> ChatRequest:
        return ChatRequest(
            task=self.task,
            messages=messages,
            json_mode=self.json_mode,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            timeout_s=self.timeout_s,
            prompt_version=self.prompt_version,
        )


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ChatResponse:
    text: str
    usage: Usage
    model: str
    provider: str
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """输出触顶 —— §6.5 报告分段要处理的核心失败模式。"""
        return self.finish_reason in ("length", "max_tokens")


# --------------------------------------------------------------------------
# 异常层次
# --------------------------------------------------------------------------
class LLMError(RuntimeError):
    """所有 LLM 相关错误的基类。"""

    retryable: bool = False


class LLMTimeout(LLMError):
    retryable = True


class LLMRateLimited(LLMError):
    retryable = True

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class LLMServerError(LLMError):
    """5xx —— 可重试。"""

    retryable = True


class LLMBadRequest(LLMError):
    """4xx 参数错误 —— **不可重试**：重试必然再失败，纯浪费配额（§9.2 第 2 条）。"""

    retryable = False


class LLMRefused(LLMError):
    """内容策略拒绝 —— 不可重试。"""

    retryable = False


class LLMUnavailable(LLMError):
    """降级链全部耗尽（§10.3 的 LLM_UNAVAILABLE）。"""

    retryable = True


class InsufficientCapabilityError(LLMError):
    """能力门控拒绝路由（§2.4.3 第 2 条）。**不是 bug，是设计**。"""

    retryable = False


class CircuitOpenError(LLMError):
    retryable = True


class StructuredOutputError(LLMError):
    """三次解析失败后显式失败 —— 不允许猜测填充默认值继续跑（§6.2）。"""

    retryable = False

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


# --------------------------------------------------------------------------
# Provider 协议
# --------------------------------------------------------------------------
@runtime_checkable
class ChatProvider(Protocol):
    """聊天补全 provider。DeepSeek 与本地 OpenAI 兼容服务共用同一实现。"""

    @property
    def name(self) -> str: ...

    @property
    def caps(self) -> Any:  # ProviderCaps，避免循环导入
        ...

    async def complete(self, req: ChatRequest) -> ChatResponse: ...

    def stream(self, req: ChatRequest) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CallRecord:
    """单次调用的计量记录（§9.2 第 7 条：成本可归因）。"""

    task: TaskKind
    provider: str
    model: str
    usage: Usage
    latency_ms: float
    attempt: int
    degraded: bool
    at: datetime
