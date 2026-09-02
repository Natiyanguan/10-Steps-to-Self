"""minimind 适配（路线文档 §2.5 / docs/LOCAL_MODELS.md §4.4）。

**先读这一段再决定要不要用本文件。**

如果你的 minimind 已经有 OpenAI 兼容的 serve 接口（多数情况如此），
**不需要本文件** —— 直接用 `OpenAICompatProvider` 并在 `.env` 里填：

    LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
    LOCAL_LLM_MODEL=minimind
    LOCAL_LLM_TIER=toy

本文件只在 minimind 暴露的是**非 OpenAI 形状的原生接口**时才需要：
它把原生接口包装成 OpenAI 兼容形状，让上层完全无感。

核心约束（§2.5）：
  **不要在业务层为 minimind 开特例分支。**
  一旦业务代码里出现 `if provider == "minimind"`，provider 抽象就失效了，
  之后每接一个新模型都要改业务代码 —— 而这正是决策 #15 要避免的。
  所有差异都必须收在这一层里。

档位（§2.4.3）：
  minimind 一律是 `TOY` 档。它能干的事只有一件 ——
  让 CI 每次提交都验证"adapter / 流式 / 超时 / 重试链路真的通"。
  能力门控会自动把它挡在 Planner / Interviewer / Analyst / Critic / Safety 之外。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.llm.base import (
    ChatRequest,
    ChatResponse,
    LLMBadRequest,
    LLMServerError,
    LLMTimeout,
    Usage,
)
from app.llm.capability import ModelTier, ProviderCaps, ZhQuality


def minimind_caps(
    *,
    context_window: int = 1_024,
    max_output_tokens: int = 512,
) -> ProviderCaps:
    """minimind 的保守能力声明。

    默认值刻意保守：从零训练的小模型不支持 JSON 模式、中文质量低。
    这些值会让能力门控自动把它限制在 CI 冒烟用途上（§2.5）。

    **`tier` 不接受参数** —— 它永远是 TOY。
    若你训练出了真正够强的模型，那它就不是 minimind 量级了，
    应该走 `OpenAICompatProvider` + `tss-probe` 实测定档。
    """
    return ProviderCaps(
        tier=ModelTier.TOY,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        supports_json_mode=False,
        supports_function_calling=False,
        supports_temperature=True,
        supports_streaming=False,
        zh_quality=ZhQuality.LOW,
        tokenizer_hint="local",
        cost_per_1k_in=0.0,
        cost_per_1k_out=0.0,
        probed_at=None,
    )


class MinimindProvider:
    """把 minimind 原生 HTTP 接口包装成本项目的 provider 协议。

    假定的原生接口形状（按你的 serve 脚本调整 `_to_native` / `_from_native`）：

        POST {base_url}/generate
        请求  {"prompt": "...", "max_new_tokens": 128, "temperature": 0.7}
        响应  {"response": "..."}

    只需改这两个方法，其余（超时映射、错误分类、重试语义）都已就位。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "minimind",
        name: str = "local_minimind",
        caps: ProviderCaps | None = None,
        endpoint: str = "/generate",
        default_timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._caps = caps or minimind_caps()
        self._endpoint = endpoint
        self._default_timeout = default_timeout_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(default_timeout_s, connect=10.0),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def caps(self) -> ProviderCaps:
        return self._caps

    # ---- 形状转换：接不同的 serve 脚本只需改这两个方法 ----

    def _to_native(self, req: ChatRequest) -> dict[str, Any]:
        """把 messages 摊平成单个 prompt。

        minimind 级模型通常没有 chat template，因此用最朴素的拼接。
        若你的 SFT 训练用了特定模板（如 `<|im_start|>`），这里要对齐 ——
        **模板不一致会让模型表现显著变差**，而且不会报错，只会输出变得答非所问。
        """
        parts = [f"{m.role.value}: {m.content}" for m in req.messages]
        payload: dict[str, Any] = {
            "prompt": "\n".join(parts) + "\nassistant:",
            "max_new_tokens": min(
                req.max_output_tokens or self._caps.max_output_tokens,
                self._caps.max_output_tokens,
            ),
        }
        if self._caps.supports_temperature and req.temperature is not None:
            payload["temperature"] = req.temperature
        return payload

    @staticmethod
    def _from_native(data: dict[str, Any]) -> str:
        for key in ("response", "text", "output", "generated_text"):
            val = data.get(key)
            if isinstance(val, str):
                return val
        raise LLMServerError(f"minimind 响应中找不到文本字段：{str(data)[:200]}")

    # ---- provider 协议 ----

    async def complete(self, req: ChatRequest) -> ChatResponse:
        timeout = req.timeout_s or self._default_timeout
        started = time.perf_counter()

        try:
            resp = await self._client.post(
                self._endpoint, json=self._to_native(req), timeout=timeout
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"minimind 超时（{timeout}s）：{exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMServerError(f"minimind 网络错误：{exc}") from exc

        if resp.status_code >= 500:
            raise LLMServerError(f"{resp.status_code}：{resp.text[:300]}")
        if resp.status_code >= 400:
            # 4xx 不可重试（§9.2 第 2 条）
            raise LLMBadRequest(f"{resp.status_code}：{resp.text[:300]}")

        text = self._from_native(resp.json())
        latency_ms = (time.perf_counter() - started) * 1000

        # 本地推理通常不返回 usage。给出估算值以便成本计量链路不断档，
        # 估算不准无妨 —— 本地推理成本为 0，这里只为让指标字段有值。
        prompt_chars = sum(len(m.content) for m in req.messages)
        return ChatResponse(
            text=text,
            usage=Usage(prompt_tokens=prompt_chars // 2, completion_tokens=len(text) // 2),
            model=self._model,
            provider=self._name,
            finish_reason="stop",
            latency_ms=latency_ms,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[str]:
        """不支持真流式时退化为一次性返回。

        `caps.supports_streaming=False` 已声明这一点，
        上层不会因此误判 —— 声明与行为一致比"假装支持"重要。
        """
        resp = await self.complete(req)
        yield resp.text

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
