"""数据库引擎与连接工厂（路线文档 §5.2 / §13.2.2）。

P1 用 SQLite + `create_all`（简单、可测试、零外部依赖）。
P5 迁 Postgres 时：改 `DATABASE_URL` + 换成 alembic 迁移，
模型层（`models.py`）与仓储层（`repo.py`）不用动 —— SQLAlchemy 方言屏蔽差异。

这里不 import 任何业务模块：storage 是基础设施，不依赖 graph / agents。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.storage.models import Base


def make_engine(url: str, *, echo: bool = False) -> AsyncEngine:
    """构造 async engine。SQLite 需要 `check_same_thread=False`。"""
    kwargs: dict[str, Any] = {"echo": echo}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_async_engine(url, **kwargs)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """建表（P1 用 create_all，见模块 docstring）。幂等。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def session_ctx(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """提供单个事务的 session。用法：`async with`。

    必须加 `@asynccontextmanager`：裸的 async generator 对象不支持
    `async with`（会报 "async_generator object does not support the
    asynchronous context manager protocol"）—— 这是测试抓到的 bug。
    """
    async with factory() as session:
        yield session
