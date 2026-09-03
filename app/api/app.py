"""FastAPI 应用工厂（路线文档 §10 / §13）。

`create_app(services=None)`：
  - 传 `services`（测试）：直接用，lifespan 不构造真实 provider。
  - 不传（生产）：lifespan 里按 settings 构造 gateway / graph / 数据库。

services 是**唯一**组装点：settings、gateway、graph、session_factory 都在这里
串起来，路由只通过 `request.app.state.services` 访问，不直接 import 全局单例 ——
这样测试才能注入 fake gateway。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import Settings, get_settings, run_startup_checks
from app.graph.build import build_graph
from app.graph.deps import Deps
from app.llm.gateway import LLMGateway
from app.storage.db import init_db, make_engine, make_session_factory


@dataclass(slots=True)
class Services:
    settings: Settings
    gateway: LLMGateway
    deps: Deps
    graph: Any
    session_factory: async_sessionmaker[AsyncSession]
    engine: AsyncEngine

    async def aclose(self) -> None:
        await self.gateway.aclose()
        await self.engine.dispose()


async def build_services(settings: Settings) -> Services:
    """生产路径：按 settings 构造全部依赖。"""
    from app.llm.providers.registry import build_gateway
    from app.observability.metrics import METRICS

    def _on_record(record: Any) -> None:
        """把每次 LLM 调用计进 Prometheus 指标（§11.3）。"""
        METRICS.record_llm_call(
            task=record.task.value,
            provider=record.provider,
            prompt_tokens=record.usage.prompt_tokens,
            completion_tokens=record.usage.completion_tokens,
        )

    # 启动自检：缺失关键配置就拒绝启动（§13.2.2）
    warnings = run_startup_checks(settings)
    gateway = build_gateway(settings, on_record=_on_record)
    deps = Deps.build(settings=settings, gateway=gateway)
    graph = build_graph(deps)
    engine = make_engine(settings.database_url)
    await init_db(engine)
    session_factory = make_session_factory(engine)
    if warnings:
        # 警告不阻断启动，但打到 stderr（§13.2.2：致命项在 run_startup_checks 里已 raise）
        import sys

        for w in warnings:
            print(f"[warn] {w}", file=sys.stderr)
    return Services(
        settings=settings,
        gateway=gateway,
        deps=deps,
        graph=graph,
        session_factory=session_factory,
        engine=engine,
    )


def create_app(services: Services | None = None) -> FastAPI:
    from app.api.routes import health, sessions

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            app.state.services = services
            yield
            return
        svc = await build_services(get_settings())
        app.state.services = svc
        try:
            yield
        finally:
            await svc.aclose()

    app = FastAPI(title="10-Steps-to-Self", version="0.1.0", lifespan=lifespan)

    # 全局异常 handler：把 ApiError 转成统一 JSON（§10.3）
    from app.api.errors import ApiError

    @app.exception_handler(ApiError)
    async def _api_error_handler(
        request: Request, exc: ApiError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
                **({"detail": exc.detail} if exc.detail else {}),
            },
            headers={"Retry-After": "1"} if exc.code == "RATE_LIMITED" else None,
        )

    app.include_router(sessions.router)
    app.include_router(health.router)
    return app
