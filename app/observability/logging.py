"""结构化日志配置（路线文档 §11.1）。

用 loguru 输出 JSON。关键约束（隐私红线）：
  - **用户原话不进日志**。调用方不该把 answer/concern 原文传给 logger；
    需要排错时用 `app.safety.injection.digest()` 记内容指纹。
  - `user_id` 一律哈希后记录。
  - Sentry 必须配 `before_send` 钩子剥离原文与 PII（异常上报是最容易泄露
    用户心理独白的通道）。这里只提供配置函数，Sentry DSN 从 settings 读。
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import Logger, logger

from app.config import Settings

_LOG_FORMAT = (
    "{time:YYYY-MM-DDTHH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}"
)


def setup_logging(settings: Settings) -> None:
    """配置 loguru：去掉默认 handler，换成 JSON sink。

    幂等：重复调用不叠加 sink。
    """
    logger.remove()

    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        serialize=True,
        format=_LOG_FORMAT,
        backtrace=False,
        diagnose=False,  # 生产不输出变量值 —— 可能含用户原文
    )

    # Sentry（可选）：配 before_send 剥离 PII 与原文
    if settings.sentry_dsn:
        _setup_sentry(settings.sentry_dsn)


def _setup_sentry(dsn: str) -> None:
    try:
        import sentry_sdk
    except ImportError:
        logger.warning("SENTRY_DSN 已配置但未安装 sentry-sdk，跳过")
        return

    from app.safety.injection import redact

    def before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any] | None:
        """剥离可能含用户心理独白的字段。§11.1：异常上报是最容易泄露的通道。"""
        for key in ("message", "formatted"):
            if key in event and isinstance(event[key], str):
                event[key] = redact(event[key], max_chars=500)
        # 面包屑里也可能有
        for crumb in (event.get("breadcrumbs") or {}).get("values") or []:
            if "message" in crumb and isinstance(crumb["message"], str):
                crumb["message"] = redact(crumb["message"], max_chars=200)
        return event

    sentry_sdk.init(dsn=dsn, before_send=before_send)


def get_logger() -> Logger:
    return logger
