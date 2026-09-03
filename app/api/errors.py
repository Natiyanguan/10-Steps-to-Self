"""API 错误码与 HTTP 映射（路线文档 §10.3）。

错误码是**契约**，前端据此决定重试/提示。因此集中定义，不在路由里各自造字符串。
`CRISIS_STOPPED` 是特殊情形：用 200 而非错误码 —— 它是正常产品路径，
前端弹红色报错框会加重危机中用户的负面体验。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.llm.base import (
    InsufficientCapabilityError,
    LLMError,
    LLMRateLimited,
    LLMUnavailable,
    StructuredOutputError,
)


@dataclass(frozen=True, slots=True)
class ApiError(Exception):
    """带错误码的业务异常，由全局 handler 转成统一 JSON 响应。"""

    code: str
    http_status: int
    message: str
    retryable: bool = False
    detail: str = ""


#: 预定义的错误码（§10.3）
CONSENT_REQUIRED = "CONSENT_REQUIRED"
SESSION_CONFLICT = "SESSION_CONFLICT"
SESSION_LOCKED = "SESSION_LOCKED"
RATE_LIMITED = "RATE_LIMITED"
BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
STRUCTURED_OUTPUT_FAILED = "STRUCTURED_OUTPUT_FAILED"
SESSION_EXPIRED = "SESSION_EXPIRED"
SESSION_NOT_FOUND = "SESSION_NOT_FOUND"  # 404，HANDOFF 之外补充
INVALID_REQUEST = "INVALID_REQUEST"      # 400


def session_not_found(session_id: str) -> ApiError:
    return ApiError(SESSION_NOT_FOUND, 404, f"会话不存在：{session_id}")


def session_conflict(session_id: str) -> ApiError:
    return ApiError(
        SESSION_CONFLICT, 409, "会话状态已变化，请重新获取后再提交", retryable=True
    )


def consent_required(reason: str = "需要先完成知情同意") -> ApiError:
    return ApiError(CONSENT_REQUIRED, 403, reason)


def invalid_request(message: str) -> ApiError:
    return ApiError(INVALID_REQUEST, 400, message)


def from_llm_error(exc: LLMError) -> ApiError:
    """把 LLM 层异常映射成 API 错误码。

    能力不足与降级链耗尽是两个最容易混淆的：前者是"当前没有模型能干活"，
    后者是"有模型但都试过了还不行"。两者都该 503，但 code 与 message 要区分清楚，
    否则排障时不知道是配置问题还是下游故障。
    """
    if isinstance(exc, InsufficientCapabilityError):
        return ApiError(
            LLM_UNAVAILABLE,
            503,
            "当前没有足以承担该任务的模型，会话已保留，可稍后继续",
            retryable=True,
            detail=str(exc),
        )
    if isinstance(exc, StructuredOutputError):
        return ApiError(
            STRUCTURED_OUTPUT_FAILED, 502, "模型输出解析失败", retryable=True, detail=str(exc)
        )
    if isinstance(exc, LLMRateLimited):
        return ApiError(RATE_LIMITED, 429, "上游限流，请稍后重试", retryable=True)
    if isinstance(exc, LLMUnavailable):
        return ApiError(LLM_UNAVAILABLE, 503, "模型服务暂时不可用", retryable=True, detail=str(exc))
    return ApiError(LLM_UNAVAILABLE, 503, "模型调用失败", retryable=True, detail=str(exc))
