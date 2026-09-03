"""FastAPI 依赖注入（路线文档 §10 / §13.2）。

路由通过 `get_services` 访问组装好的 services，不 import 全局单例 ——
测试注入 fake services 时零改动。
"""

from __future__ import annotations

from typing import cast

from fastapi import Request

from app.api.app import Services


def get_services(request: Request) -> Services:
    return cast(Services, request.app.state.services)
