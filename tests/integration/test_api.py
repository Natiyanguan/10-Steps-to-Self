"""HTTP API 集成测试（路线文档 §10）。

用 fake services（DemoProvider + 临时 SQLite）+ httpx ASGITransport，
全部在 pytest-asyncio 的同一事件循环里 —— 避免 TestClient 跨循环导致
engine 的 async 连接失效。

验证：创建会话 → 提交回答 → 获取状态 → 健康检查 → 指标，以及错误码映射。
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from app.api.app import Services, create_app
from app.config import Settings
from app.graph.build import build_graph
from app.graph.deps import Deps
from app.llm.gateway import LLMGateway, Route
from app.llm.providers.demo import DemoProvider
from app.storage.db import init_db, make_engine, make_session_factory
from tests.conftest import PROJECT_ROOT

POLICY = "test-policy-1"


@pytest_asyncio.fixture
async def services(tmp_path) -> Services:
    settings = Settings(
        _env_file=None,
        policy_version=POLICY,
        llm_api_key="test",
        llm_model_strong="demo",
        llm_model_fast="demo",
    )
    gw = LLMGateway(routes=[Route(provider=DemoProvider(), label="primary")])
    deps = Deps.build(settings=settings, gateway=gw, prompts_root=PROJECT_ROOT / "prompts")
    graph = build_graph(deps)
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = make_engine(db_url)
    await init_db(engine)
    factory = make_session_factory(engine)
    return Services(
        settings=settings, gateway=gw, deps=deps, graph=graph,
        session_factory=factory, engine=engine,
    )


@pytest_asyncio.fixture
async def client(services):
    app = create_app()
    app.state.services = services
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await services.aclose()


def _create_body(concern: str = "我在纠结是否要离职创业"):
    return {
        "concern": concern,
        "mode": "progressive",
        "sharpness": "moderate",
        "consent": {"policy_version": POLICY, "scopes": ["store_memory"]},
    }


@pytest.mark.asyncio
async def test_create_session(client) -> None:
    r = await client.post("/api/v1/sessions", json=_create_body())
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["session_id"]
    assert data["phase"] in ("listen", "ask", "consent")
    # progressive 模式第一轮应发出第 1 问
    assert data["open_questions"], "第一轮应发出一个问题"


@pytest.mark.asyncio
async def test_full_round_trip(client) -> None:
    """创建 → 提交 10 个回答 → 得到报告。"""
    r = await client.post("/api/v1/sessions", json=_create_body())
    session_id = r.json()["session_id"]

    for i in range(10):
        r = await client.post(
            f"/api/v1/sessions/{session_id}/answers",
            json={"text": f"我最怕失去现在稳定的生活，第{i + 1}轮想法。"},
        )
        assert r.status_code == 200, r.text

    data = r.json()
    assert data["phase"] == "done"
    assert data["report"] is not None


@pytest.mark.asyncio
async def test_get_session(client) -> None:
    r = await client.post("/api/v1/sessions", json=_create_body())
    session_id = r.json()["session_id"]

    r2 = await client.get(f"/api/v1/sessions/{session_id}")
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id


@pytest.mark.asyncio
async def test_session_not_found_404(client) -> None:
    r = await client.get("/api/v1/sessions/nonexistent")
    assert r.status_code == 404
    assert r.json()["code"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_submit_without_open_questions_400(client) -> None:
    """没有待答问题就提交 → 400。"""
    r = await client.post("/api/v1/sessions", json=_create_body())
    session_id = r.json()["session_id"]
    # 未回答任何问题，直接再提交一个空文本会被 min_length 拦，改用完成后再提交
    for i in range(10):
        r = await client.post(
            f"/api/v1/sessions/{session_id}/answers",
            json={"text": f"我最怕失去现在稳定的生活，第{i + 1}轮。"},
        )
    # 已 done，再提交 → 400
    r = await client.post(
        f"/api/v1/sessions/{session_id}/answers", json={"text": "再回答一次"}
    )
    assert r.status_code == 400
    assert r.json()["code"] == "INVALID_REQUEST"


@pytest.mark.asyncio
async def test_crisis_returns_200_not_error(client) -> None:
    """危机信号：终止会话，但 HTTP 是 200（CRISIS_STOPPED 是正常路径，§10.3）。"""
    r = await client.post(
        "/api/v1/sessions",
        json=_create_body(concern="我最近一直在想自杀，撑不下去了"),
    )
    assert r.status_code in (200, 201), r.text
    data = r.json()
    assert data["phase"] == "crisis"
    # 危机事件应包含配置中的资源
    events = data["events"]
    assert any(e["event"] == "crisis_stopped" for e in events)


@pytest.mark.asyncio
async def test_health(client) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ready(client) -> None:
    r = await client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_metrics(client) -> None:
    r = await client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "tss_" in body  # 至少暴露了我们的指标
    assert "process_" in body or "python_" in body  # prometheus 自带进程指标


@pytest.mark.asyncio
async def test_consent_required_without_consent(client) -> None:
    """不传 consent → 停在 consent 阶段，返回 consent_required 事件。"""
    body = _create_body()
    del body["consent"]
    r = await client.post("/api/v1/sessions", json=body)
    assert r.status_code == 201
    data = r.json()
    assert data["phase"] == "consent"
    assert data["consent_status"] == "none"
    assert any(e["event"] == "consent_required" for e in data["events"])
