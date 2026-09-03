"""storage 层单测（路线文档 §5.2 / §9.1）。

用 SQLite in-memory（StaticPool 保证所有连接共享同一内存库）。
覆盖：会话 round-trip、乐观锁冲突、审计 append、同意版本化、状态提取。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.schemas.enums import ConsentScope, Phase, RiskLevel
from app.schemas.session import Consent, SessionState
from app.storage.db import init_db, make_engine, make_session_factory
from app.storage.repo import (
    AuditRepo,
    ConsentRepo,
    SessionConflictError,
    SessionNotFoundError,
    SessionRepo,
    deserialize_state,
    extract_session_state,
    serialize_state,
)
from tests.conftest import make_state


@pytest_asyncio.fixture
async def factory() -> async_sessionmaker:
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    yield make_session_factory(engine)
    await engine.dispose()


def _session_id(state: SessionState) -> str:
    return state.session_id


class TestSerialize:
    def test_round_trip(self) -> None:
        state = make_state()
        ss = extract_session_state(state)
        raw = serialize_state(ss)
        back = deserialize_state(raw)
        assert back == ss
        assert back.session_id == ss.session_id

    def test_extract_drops_runtime_fields(self) -> None:
        """从 GraphState 提取 SessionState，丢弃运行时字段（emissions/inbox/…）。"""
        gs = make_state()
        ss = extract_session_state(gs)
        # SessionState 不该有 GraphState 独有的字段
        assert not hasattr(ss, "emissions")
        assert not hasattr(ss, "inbox")
        assert not hasattr(ss, "safety_action")
        # 但保留领域字段
        assert ss.concern or ss.concern == ""
        assert ss.revision == 0


class TestSessionRepo:
    @pytest.mark.asyncio
    async def test_save_and_load(self, factory) -> None:
        state = extract_session_state(make_state())
        async with factory() as s:
            repo = SessionRepo(s)
            await repo.save(state)
            await s.commit()

        async with factory() as s:
            repo = SessionRepo(s)
            loaded = await repo.load(state.session_id)
        assert loaded is not None
        assert loaded.session_id == state.session_id
        assert loaded.user_id == state.user_id
        assert loaded.phase == state.phase

    @pytest.mark.asyncio
    async def test_load_missing_returns_none(self, factory) -> None:
        async with factory() as s:
            repo = SessionRepo(s)
            assert await repo.load("nonexistent") is None

    @pytest.mark.asyncio
    async def test_load_or_raise(self, factory) -> None:
        async with factory() as s:
            repo = SessionRepo(s)
            with pytest.raises(SessionNotFoundError):
                await repo.load_or_raise("nonexistent")

    @pytest.mark.asyncio
    async def test_update_with_revision_increments(self, factory) -> None:
        state = extract_session_state(make_state())
        async with factory() as s:
            repo = SessionRepo(s)
            await repo.save(state)
            await s.commit()

        # 第一次更新：revision 0 → 1
        async with factory() as s:
            repo = SessionRepo(s)
            updated = await repo.update_with_revision(state)
            await s.commit()
        assert updated.revision == 1

        # 第二次更新：revision 1 → 2
        async with factory() as s:
            repo = SessionRepo(s)
            state2 = updated.model_copy(update={"phase": Phase.PLAN})
            updated2 = await repo.update_with_revision(state2)
            await s.commit()
        assert updated2.revision == 2

    @pytest.mark.asyncio
    async def test_update_with_revision_conflict(self, factory) -> None:
        """乐观锁：用过期 revision 更新时抛 SessionConflictError（§9.1）。"""
        state = extract_session_state(make_state())
        async with factory() as s:
            repo = SessionRepo(s)
            await repo.save(state)
            await s.commit()

        # 第一次更新成功，revision 变成 1
        async with factory() as s:
            repo = SessionRepo(s)
            await repo.update_with_revision(state)
            await s.commit()

        # 用**旧的** state（revision 还是 0）再更新 → 冲突
        async with factory() as s:
            repo = SessionRepo(s)
            with pytest.raises(SessionConflictError):
                await repo.update_with_revision(state)

    @pytest.mark.asyncio
    async def test_revision_persists_after_reload(self, factory) -> None:
        """回归测试：update 后**重新 load**，revision 必须等于更新后的版本。

        曾有一个 bug：`revision` 列写成了 +1，但 `state_json` 里序列化的还是旧
        revision —— 下次 load 读回旧版本号，乐观锁形同虚设。
        """
        state = extract_session_state(make_state())
        async with factory() as s:
            repo = SessionRepo(s)
            await repo.save(state)
            await s.commit()

        async with factory() as s:
            repo = SessionRepo(s)
            updated = await repo.update_with_revision(state)
            await s.commit()
        assert updated.revision == 1

        # 重新 load，revision 必须一致
        async with factory() as s:
            repo = SessionRepo(s)
            reloaded = await repo.load(state.session_id)
        assert reloaded is not None
        assert reloaded.revision == 1, "state_json 里的 revision 与 revision 列不同步"

    @pytest.mark.asyncio
    async def test_status_maps_to_completed(self, factory) -> None:
        state = extract_session_state(make_state())
        state = state.model_copy(update={"phase": Phase.DONE})
        async with factory() as s:
            repo = SessionRepo(s)
            await repo.save(state)
            await s.commit()
        # 直接查冗余列
        from sqlalchemy import select

        from app.storage.models import SessionRecord

        async with factory() as s:
            record = (await s.execute(
                select(SessionRecord).where(SessionRecord.session_id == state.session_id)
            )).scalar_one()
        assert record.status == "completed"


class TestAuditRepo:
    @pytest.mark.asyncio
    async def test_append(self, factory) -> None:
        async with factory() as s:
            repo = AuditRepo(s)
            await repo.append(
                session_id="sess-1",
                user_id_hash="hash123",
                risk_level=RiskLevel.HIGH,
                source="lexicon",
                matched_rules=["high_self_harm"],
                action_taken="stop_with_resources",
            )
            await s.commit()

        from sqlalchemy import select

        from app.storage.models import SafetyAuditRecord

        async with factory() as s:
            rows = list((await s.execute(select(SafetyAuditRecord))).scalars())
        assert len(rows) == 1
        assert rows[0].risk_level == "high"
        assert rows[0].matched_rules == ["high_self_harm"]


class TestConsentRepo:
    @pytest.mark.asyncio
    async def test_save_and_load(self, factory) -> None:
        consent = Consent(
            policy_version="v1",
            granted_at=datetime.now(UTC),
            scopes={ConsentScope.STORE_MEMORY, ConsentScope.CROSS_SESSION_RECALL},
        )
        async with factory() as s:
            repo = ConsentRepo(s)
            await repo.save("user-1", consent)
            await s.commit()

        async with factory() as s:
            repo = ConsentRepo(s)
            loaded = await repo.load("user-1", "v1")
        assert loaded is not None
        assert loaded.policy_version == "v1"
        assert loaded.scopes == {ConsentScope.STORE_MEMORY, ConsentScope.CROSS_SESSION_RECALL}
        assert loaded.is_active

    @pytest.mark.asyncio
    async def test_version_isolation(self, factory) -> None:
        """同一用户不同版本是独立记录（§3.2 版本化）。"""
        async with factory() as s:
            repo = ConsentRepo(s)
            await repo.save(
                "user-1",
                Consent(policy_version="v1", granted_at=datetime.now(UTC), scopes=set()),
            )
            await s.commit()

        async with factory() as s:
            repo = ConsentRepo(s)
            assert await repo.load("user-1", "v1") is not None
            assert await repo.load("user-1", "v2") is None

    @pytest.mark.asyncio
    async def test_withdraw(self, factory) -> None:
        """撤回同意：is_active 变 False，但记录保留（审计）。"""
        async with factory() as s:
            repo = ConsentRepo(s)
            await repo.save(
                "user-1",
                Consent(policy_version="v1", granted_at=datetime.now(UTC), scopes=set()),
            )
            await s.commit()

        withdrawn = Consent(
            policy_version="v1",
            granted_at=datetime.now(UTC),
            scopes=set(),
            withdrawn_at=datetime.now(UTC),
        )
        async with factory() as s:
            repo = ConsentRepo(s)
            await repo.save("user-1", withdrawn)
            await s.commit()

        async with factory() as s:
            repo = ConsentRepo(s)
            loaded = await repo.load("user-1", "v1")
        assert loaded is not None
        assert not loaded.is_active
