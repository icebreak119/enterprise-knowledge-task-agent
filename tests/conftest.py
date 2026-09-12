import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_llm
from app.llm.fake_provider import FakeProvider
from app.main import app


@pytest.fixture
def client():
    """用 FakeProvider 替换真实 LLM，测试不依赖任何 API key。

    注意不使用 `with TestClient(app)`，避免触发 lifespan 去连数据库。
    """
    app.dependency_overrides[get_llm] = lambda: FakeProvider()
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()
