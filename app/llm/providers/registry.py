"""按配置装配 provider 与降级链（路线文档 §2.1 / §9.2 第 4 条 / §13.2.1）。

这里是**唯一**出现模型名与 endpoint 的地方（§16.2 硬约束）。
业务代码只认 `TaskKind`，不认模型。

能力来源优先序：
  1. `config/providers.yaml`（§17 探测脚本实测写入）—— 最可信
  2. 内置保守默认值 —— 未探测时按最保守值处理（§17.4：绝不乐观假设）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from app.config import RuntimeProfile, Settings
from app.llm.capability import ModelTier, ProviderCaps, ZhQuality
from app.llm.gateway import Budget, LLMGateway, Route
from app.llm.providers.openai_compat import OpenAICompatProvider

# --------------------------------------------------------------------------
# 未探测时的保守默认能力
#
# 注意 supports_json_mode=False：未经 §17 探测就假设支持 JSON 模式，
# 会在生产以"解析失败"的形式暴露。宁可让能力门控先拒绝，
# 逼使用者去跑探测脚本 —— 那是一次性成本，而静默失败是持续成本。
# --------------------------------------------------------------------------
_UNPROBED_STRONG = ProviderCaps(
    tier=ModelTier.STRONG,
    context_window=64_000,
    max_output_tokens=4_096,
    supports_json_mode=False,
    supports_temperature=True,
    supports_streaming=True,
    zh_quality=ZhQuality.HIGH,
    probed_at=None,
)

_UNPROBED_FAST = ProviderCaps(
    tier=ModelTier.FAST,
    context_window=32_000,
    max_output_tokens=4_096,
    supports_json_mode=False,
    supports_temperature=True,
    supports_streaming=True,
    zh_quality=ZhQuality.MEDIUM,
    probed_at=None,
)

# 本地 minimind 级模型：上下文极小、无 JSON 模式、中文质量低。
# 这些默认值会让能力门控自动把它挡在 STRONG 与 SAFETY 之外（§2.5）。
_LOCAL_TOY = ProviderCaps(
    tier=ModelTier.TOY,
    context_window=1_024,
    max_output_tokens=512,
    supports_json_mode=False,
    supports_function_calling=False,
    supports_temperature=True,
    supports_streaming=True,
    zh_quality=ZhQuality.LOW,
    tokenizer_hint="local",
    probed_at=None,
)


def _caps_from_dict(raw: dict[str, Any]) -> ProviderCaps:
    probed = raw.get("probed_at")
    return ProviderCaps(
        tier=ModelTier.parse(raw.get("tier", "fast")),
        context_window=int(raw.get("context_window", 8_000)),
        max_output_tokens=int(raw.get("max_output_tokens", 2_048)),
        supports_json_mode=bool(raw.get("supports_json_mode", False)),
        supports_function_calling=bool(raw.get("supports_function_calling", False)),
        supports_temperature=bool(raw.get("supports_temperature", True)),
        supports_streaming=bool(raw.get("supports_streaming", True)),
        zh_quality=ZhQuality(str(raw.get("zh_quality", "medium"))),
        tokenizer_hint=str(raw.get("tokenizer_hint", "cl100k_base")),
        cost_per_1k_in=raw.get("cost_per_1k_in"),
        cost_per_1k_out=raw.get("cost_per_1k_out"),
        probed_at=(
            probed
            if isinstance(probed, datetime)
            else datetime.fromisoformat(str(probed))
            if probed
            else None
        ),
    )


def load_probed_caps(path: Path) -> dict[str, ProviderCaps]:
    """读取 §17 探测脚本的产出。文件不存在返回空 dict（走保守默认）。"""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    providers = data.get("providers") or {}
    return {str(k): _caps_from_dict(v) for k, v in providers.items() if isinstance(v, dict)}


class ProviderConfigError(RuntimeError):
    pass


def build_routes(settings: Settings) -> list[Route]:
    """按运行档位装配降级链。顺序即降级顺序（§9.2 第 4 条）。"""
    probed = load_probed_caps(settings.providers_file)
    routes: list[Route] = []

    def caps_for(key: str, default: ProviderCaps) -> ProviderCaps:
        return probed.get(key, default)

    if settings.runtime_profile is not RuntimeProfile.LOCAL:
        # 主路径：强档
        routes.append(
            Route(
                provider=OpenAICompatProvider(
                    name="cloud_strong",
                    base_url=settings.llm_base_url,
                    api_key=settings.llm_api_key,
                    model=settings.llm_model_strong,
                    caps=caps_for("cloud_strong", _UNPROBED_STRONG),
                    default_timeout_s=settings.llm_timeout_seconds,
                ),
                label="primary",
            )
        )
        # 同供应商快档：承接 FAST 任务，也作为强档故障时的部分兜底
        if settings.llm_model_fast:
            routes.append(
                Route(
                    provider=OpenAICompatProvider(
                        name="cloud_fast",
                        base_url=settings.llm_base_url,
                        api_key=settings.llm_api_key,
                        model=settings.llm_model_fast,
                        caps=caps_for("cloud_fast", _UNPROBED_FAST),
                        default_timeout_s=settings.llm_timeout_seconds,
                    ),
                    label="fallback_model",
                )
            )
        # 备用供应商（决策 #24）：缺失时 STRONG 任务无兜底，只能显式失败
        if settings.has_fallback_provider:
            routes.append(
                Route(
                    provider=OpenAICompatProvider(
                        name="cloud_backup",
                        base_url=settings.llm_fallback_base_url,
                        api_key=settings.llm_fallback_api_key,
                        model=settings.llm_fallback_model,
                        caps=caps_for("cloud_backup", _UNPROBED_STRONG),
                        default_timeout_s=settings.llm_timeout_seconds,
                    ),
                    label="fallback_provider",
                )
            )

    # 本地模型：始终排在最后。
    # 它能承担什么完全由 caps 决定 —— TOY 档会被门控挡在 STRONG/SAFETY 之外（§2.5）。
    if settings.local_llm_model:
        local_default = (
            _LOCAL_TOY
            if settings.local_llm_tier == "toy"
            else ProviderCaps(
                tier=ModelTier.FAST,
                context_window=32_000,
                max_output_tokens=2_048,
                supports_json_mode=False,  # 未探测前保守
                supports_temperature=True,
                supports_streaming=True,
                zh_quality=ZhQuality.MEDIUM,
                tokenizer_hint="local",
                probed_at=None,
            )
        )
        routes.append(
            Route(
                provider=OpenAICompatProvider(
                    name="local",
                    base_url=settings.local_llm_base_url,
                    api_key="",  # Ollama / vLLM 通常不需要
                    model=settings.local_llm_model,
                    caps=caps_for("local", local_default),
                    default_timeout_s=max(settings.llm_timeout_seconds, 300.0),
                ),
                label="local",
            )
        )

    if not routes:
        raise ProviderConfigError(
            "没有可用的 provider：cloud 档需要 LLM_API_KEY + LLM_MODEL_STRONG，"
            " local 档需要 LOCAL_LLM_MODEL。参考 .env.example。"
        )
    return routes


def build_gateway(settings: Settings, *, budget: Budget | None = None) -> LLMGateway:
    return LLMGateway(
        routes=build_routes(settings),
        default_temperature=settings.llm_temperature,
        budget=budget,
    )
