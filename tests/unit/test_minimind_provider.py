"""MinimindProvider 适配层单测（docs/LOCAL_MODELS.md §4.4）。

用 respx mock HTTP，不需要真的跑起 minimind。
验证的是**形状转换与错误映射**，不涉及内容质量 —— TOY 档没有内容质量可言。
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.llm.base import (
    ChatRequest,
    LLMBadRequest,
    LLMServerError,
    LLMTimeout,
    Message,
    Role,
    TaskKind,
)
from app.llm.capability import ModelTier, ZhQuality
from app.llm.providers.local_minimind import MinimindProvider, minimind_caps

BASE = "http://127.0.0.1:8000"


def _req(text: str = "你好", **kw) -> ChatRequest:
    return ChatRequest(
        task=TaskKind.QUERY_REWRITE,
        messages=(Message(Role.USER, text),),
        **kw,
    )


class TestCaps:
    def test_tier_is_always_toy(self) -> None:
        """档位恒为 TOY —— 不接受参数覆盖（§2.4.3）。"""
        assert minimind_caps().tier is ModelTier.TOY
        # 即使调大上下文，档位也不变
        assert minimind_caps(context_window=32_000).tier is ModelTier.TOY

    def test_conservative_defaults(self) -> None:
        """默认值必须保守：不声称支持 JSON、中文质量低。"""
        caps = minimind_caps()
        assert caps.supports_json_mode is False
        assert caps.supports_streaming is False
        assert caps.zh_quality is ZhQuality.LOW
        assert caps.cost_per_1k_in == 0.0

    def test_declares_no_streaming_and_behaves_accordingly(self) -> None:
        """声明与行为必须一致 —— 不支持流式就别假装支持。"""
        assert minimind_caps().supports_streaming is False


class TestShapeConversion:
    @respx.mock
    @pytest.mark.asyncio
    async def test_flattens_messages_to_prompt(self) -> None:
        route = respx.post(f"{BASE}/generate").mock(
            return_value=httpx.Response(200, json={"response": "我很好"})
        )
        p = MinimindProvider(base_url=BASE)
        try:
            resp = await p.complete(
                ChatRequest(
                    task=TaskKind.QUERY_REWRITE,
                    messages=(
                        Message(Role.SYSTEM, "你是助手"),
                        Message(Role.USER, "你好"),
                    ),
                )
            )
        finally:
            await p.aclose()

        assert resp.text == "我很好"
        sent = route.calls.last.request
        body = sent.content.decode()
        # 摊平后应含各条消息，并以 assistant: 收尾引导续写
        assert "你是助手" in body
        assert "你好" in body
        assert "assistant:" in body

    @respx.mock
    @pytest.mark.asyncio
    async def test_caps_clamp_max_output(self) -> None:
        """请求超过 caps 上限时必须被裁剪到上限。"""
        route = respx.post(f"{BASE}/generate").mock(
            return_value=httpx.Response(200, json={"response": "ok"})
        )
        p = MinimindProvider(base_url=BASE, caps=minimind_caps(max_output_tokens=256))
        try:
            await p.complete(_req(max_output_tokens=99_999))
        finally:
            await p.aclose()

        payload = json.loads(route.calls.last.request.content)
        assert payload["max_new_tokens"] == 256

    @respx.mock
    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", ["response", "text", "output", "generated_text"])
    async def test_accepts_common_field_names(self, field: str) -> None:
        """不同 serve 脚本字段名不同 —— 都要能取到。"""
        respx.post(f"{BASE}/generate").mock(
            return_value=httpx.Response(200, json={field: "内容"})
        )
        p = MinimindProvider(base_url=BASE)
        try:
            assert (await p.complete(_req())).text == "内容"
        finally:
            await p.aclose()

    @respx.mock
    @pytest.mark.asyncio
    async def test_unknown_field_raises_clearly(self) -> None:
        """取不到文本时必须显式报错，而不是返回空字符串。

        返回空串会让上层以"schema 校验失败"的形式暴露，
        排查时会误以为是模型输出问题，而不是适配层没对上字段。
        """
        respx.post(f"{BASE}/generate").mock(
            return_value=httpx.Response(200, json={"unexpected": "x"})
        )
        p = MinimindProvider(base_url=BASE)
        try:
            with pytest.raises(LLMServerError, match="找不到文本字段"):
                await p.complete(_req())
        finally:
            await p.aclose()

    @respx.mock
    @pytest.mark.asyncio
    async def test_custom_endpoint(self) -> None:
        respx.post(f"{BASE}/api/chat").mock(
            return_value=httpx.Response(200, json={"response": "ok"})
        )
        p = MinimindProvider(base_url=BASE, endpoint="/api/chat")
        try:
            assert (await p.complete(_req())).text == "ok"
        finally:
            await p.aclose()


class TestErrorMapping:
    """可重试性必须映射正确（§9.2 第 2 条）：
    重试不可重试的错误纯属浪费；不重试可重试的错误则丢失了本可成功的调用。
    """

    @respx.mock
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 404, 422])
    async def test_4xx_not_retryable(self, status: int) -> None:
        respx.post(f"{BASE}/generate").mock(return_value=httpx.Response(status, text="bad"))
        p = MinimindProvider(base_url=BASE)
        try:
            with pytest.raises(LLMBadRequest) as ei:
                await p.complete(_req())
            assert ei.value.retryable is False
        finally:
            await p.aclose()

    @respx.mock
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [500, 502, 503])
    async def test_5xx_retryable(self, status: int) -> None:
        respx.post(f"{BASE}/generate").mock(return_value=httpx.Response(status, text="boom"))
        p = MinimindProvider(base_url=BASE)
        try:
            with pytest.raises(LLMServerError) as ei:
                await p.complete(_req())
            assert ei.value.retryable is True
        finally:
            await p.aclose()

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_maps_to_llm_timeout(self) -> None:
        respx.post(f"{BASE}/generate").mock(side_effect=httpx.ReadTimeout("slow"))
        p = MinimindProvider(base_url=BASE)
        try:
            with pytest.raises(LLMTimeout) as ei:
                await p.complete(_req())
            assert ei.value.retryable is True
        finally:
            await p.aclose()

    @respx.mock
    @pytest.mark.asyncio
    async def test_connect_error_maps_to_server_error(self) -> None:
        """minimind 没启动是最常见的情况 —— 必须给可重试的明确错误。"""
        respx.post(f"{BASE}/generate").mock(side_effect=httpx.ConnectError("refused"))
        p = MinimindProvider(base_url=BASE)
        try:
            with pytest.raises(LLMServerError) as ei:
                await p.complete(_req())
            assert ei.value.retryable is True
        finally:
            await p.aclose()


class TestStreamAndUsage:
    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_degrades_to_single_chunk(self) -> None:
        respx.post(f"{BASE}/generate").mock(
            return_value=httpx.Response(200, json={"response": "一二三"})
        )
        p = MinimindProvider(base_url=BASE)
        try:
            chunks = [c async for c in p.stream(_req())]
        finally:
            await p.aclose()
        assert chunks == ["一二三"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_usage_is_estimated_not_zero(self) -> None:
        """本地推理不返回 usage —— 给估算值，让计量链路不断档。"""
        respx.post(f"{BASE}/generate").mock(
            return_value=httpx.Response(200, json={"response": "这是一段回答内容"})
        )
        p = MinimindProvider(base_url=BASE)
        try:
            resp = await p.complete(_req("这是一个比较长的问题输入"))
        finally:
            await p.aclose()

        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0
        assert resp.provider == "local_minimind"
        assert resp.latency_ms >= 0
