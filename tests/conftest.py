import pytest
from fastapi.testclient import TestClient

import app.api.routes_chat as routes_chat
from app.api.deps import get_llm
from app.llm.fake_provider import FakeProvider
from app.main import app


async def _empty_search(session, query, **kwargs):
    """检索要连真实数据库并调 embedding 接口，单测里一律替换掉。"""
    return []


@pytest.fixture
async def db_session():
    """集成测试专用会话。

    不能用 app.db.session 的共享引擎：pytest-asyncio 每个测试跑在独立事件循环上，
    而连接池里的连接会绑定到创建它的那个循环，跨循环复用会报
    'NoneType' object has no attribute 'send'。用 NullPool 一次性连接绕开这个坑，
    也不必为了测试去改生产环境的连接池配置。
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from app.core.config import get_settings

    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def client(monkeypatch):
    """用 FakeProvider 替换真实 LLM，并截断检索，测试不依赖任何 API key 与数据库。

    注意不使用 `with TestClient(app)`，避免触发 lifespan 去连数据库。
    """
    monkeypatch.setattr(routes_chat, "search", _empty_search)
    app.dependency_overrides[get_llm] = lambda: FakeProvider()
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()
