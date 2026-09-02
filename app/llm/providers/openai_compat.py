"""OpenAI 兼容协议 provider（路线文档 §2.1 / §2.5 / §9.2）。

**DeepSeek 与本地模型（Ollama / vLLM / minimind serve）共用这一个 adapter** ——
仅 `base_url` / `model` / `caps` 不同。这是决策 #15 的核心收益：
切换供应商 = 改配置，业务代码零改动。

用 httpx 直连而非 SDK（§16.3）：需要精确控制超时分级、
按 caps 裁剪请求参数、把 HTTP 状态精确映射到可重试/不可重试异常。
少一层封装更利于这些控制，且 respx 可直接 mock。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.llm.base import (
    ChatRequest,
    ChatResponse,
    LLMBadRequest,
    LLMError,
    LLMRateLimited,
    LLMRefused,
    LLMServerError,
    LLMTimeout,
    Usage,
)
from app.llm.capability import ProviderCaps


class OpenAICompatProvider:
    """任何暴露 `/chat/completions` 的 OpenAI 兼容服务。"""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        caps: ProviderCaps,
        default_timeout_s: float = 180.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._caps = caps
        self._default_timeout = default_timeout_s
        self._owns_client = client is None
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(default_timeout_s, connect=10.0),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def caps(self) -> ProviderCaps:
        return self._caps

    # ---- 请求构造：按 caps 裁剪参数（§9.2 第 8 条）----

    def _build_payload(self, req: ChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role.value, "content": m.content} for m in req.messages],
            "stream": stream,
        }

        # 不支持 temperature 的推理型模型会直接报 400 —— 不可重试，必须提前裁剪
        if self._caps.supports_temperature and req.temperature is not None:
            payload["temperature"] = req.temperature

        max_out = req.max_output_tokens or self._caps.max_output_tokens
        if max_out:
            payload["max_tokens"] = min(max_out, self._caps.max_output_tokens)

        # JSON 模式为主路径（决策 #20）；不支持时靠提示词约束 + 严格解析兜底
        if req.json_mode and self._caps.supports_json_mode:
            payload["response_format"] = {"type": "json_object"}

        return payload

    # ---- 错误映射：可重试性在这里定死（§9.2 第 2 条）----

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return

        body = resp.text[:500]
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            raise LLMRateLimited(
                f"429 限流：{body}",
                retry_after_s=float(retry_after) if retry_after else None,
            )
        if resp.status_code >= 500:
            raise LLMServerError(f"{resp.status_code} 服务端错误：{body}")
        if resp.status_code in (400, 422):
            # 内容策略拒绝通常也走 400，单独识别以便上层区分处理
            lowered = body.lower()
            if any(k in lowered for k in ("content_filter", "content policy", "safety")):
                raise LLMRefused(f"内容策略拒绝：{body}")
            raise LLMBadRequest(f"{resp.status_code} 请求错误（不可重试）：{body}")
        if resp.status_code in (401, 403):
            raise LLMBadRequest(f"{resp.status_code} 鉴权失败（检查 API key）：{body}")
        raise LLMBadRequest(f"{resp.status_code}：{body}")

    @staticmethod
    def _parse_usage(data: dict[str, Any]) -> Usage:
        usage = data.get("usage") or {}
        return Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )

    @staticmethod
    def _extract_text(choice: dict[str, Any]) -> str:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        # 部分推理型模型把正文与思维链分开返回；只取正文，思维链不进下游
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return ""

    # ---- 调用 ----

    async def complete(self, req: ChatRequest) -> ChatResponse:
        payload = self._build_payload(req, stream=False)
        timeout = req.timeout_s or self._default_timeout
        started = time.perf_counter()

        try:
            resp = await self._client.post(
                "/chat/completions", json=payload, timeout=timeout
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"请求超时（{timeout}s）：{exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMServerError(f"网络错误：{exc}") from exc

        self._raise_for_status(resp)
        latency_ms = (time.perf_counter() - started) * 1000

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMServerError(f"响应不是合法 JSON：{resp.text[:200]}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise LLMServerError(f"响应无 choices：{str(data)[:200]}")

        choice = choices[0]
        return ChatResponse(
            text=self._extract_text(choice),
            usage=self._parse_usage(data),
            model=str(data.get("model", self._model)),
            provider=self._name,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            latency_ms=latency_ms,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[str]:
        """SSE 流式（§9.4：首 token 时间是感知延迟关键）。"""
        if not self._caps.supports_streaming:
            resp = await self.complete(req)
            yield resp.text
            return

        payload = self._build_payload(req, stream=True)
        timeout = req.timeout_s or self._default_timeout

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload, timeout=timeout
            ) as http_resp:
                if http_resp.status_code >= 400:
                    await http_resp.aread()
                    self._raise_for_status(http_resp)

                async for line in http_resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk in ("", "[DONE]"):
                        continue
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    for ch in obj.get("choices") or []:
                        delta = (ch.get("delta") or {}).get("content")
                        if delta:
                            yield delta
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"流式请求超时（{timeout}s）：{exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMServerError(f"流式网络错误：{exc}") from exc

    async def probe(self) -> dict[str, Any]:
        """轻量连通性探测，供 `/ready` 与 §17 探测脚本使用。"""
        from app.llm.base import Message, Role, TaskKind

        req = ChatRequest(
            task=TaskKind.QUERY_REWRITE,
            messages=(Message(Role.USER, "ping"),),
            max_output_tokens=8,
            timeout_s=20.0,
        )
        try:
            resp = await self.complete(req)
        except LLMError as exc:
            return {"ok": False, "error": str(exc), "provider": self._name}
        return {
            "ok": True,
            "provider": self._name,
            "model": resp.model,
            "latency_ms": round(resp.latency_ms, 1),
        }

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
