"""健康检查与指标端点（路线文档 §10.1 / §11.3）。

`/health`  存活探针：进程活着就 200。
`/ready`   就绪探针：校验 DB 与 provider 连通。任一项失败 → 503。
           K8s 依赖它就绪探针来决定是否路由流量（§13.4）。
`/metrics` Prometheus 指标（§11.3）。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, PlainTextResponse

from app.api.app import Services
from app.api.deps import get_services
from app.llm.base import TaskKind
from app.storage.db import session_ctx

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(services: Annotated[Services, Depends(get_services)]) -> JSONResponse:
    failures: list[str] = []

    # DB 连通性：能开事务执行一个最小查询就算通过
    try:
        from sqlalchemy import text

        async with session_ctx(services.session_factory) as s:
            await s.execute(text("SELECT 1"))
    except Exception as exc:
        failures.append(f"db: {type(exc).__name__}")

    # provider 连通性：至少有能承担 STRONG 任务的 provider
    if not services.gateway.eligible_routes(TaskKind.ASK):
        failures.append("provider: no STRONG-capable provider")

    if failures:
        return JSONResponse({"status": "not_ready", "failures": failures}, status_code=503)
    return JSONResponse({"status": "ready"}, status_code=200)


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    # 必须先 import metrics 模块 —— 指标在模块级构造时注册进 REGISTRY，
    # 只 import generate_latest 不会触发 `app.observability.metrics` 的模块加载。
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    from app.observability import metrics as _metrics  # noqa: F401

    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
