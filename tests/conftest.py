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
def client(monkeypatch):
    """用 FakeProvider 替换真实 LLM，并截断检索，测试不依赖任何 API key 与数据库。

    注意不使用 `with TestClient(app)`，避免触发 lifespan 去连数据库。
    """
    monkeypatch.setattr(routes_chat, "search", _empty_search)
    app.dependency_overrides[get_llm] = lambda: FakeProvider()
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()
