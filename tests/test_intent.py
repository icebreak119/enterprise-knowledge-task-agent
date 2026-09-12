import pytest

from app.llm.base import ChatMessage
from app.llm.fake_provider import FakeProvider
from app.schemas.intent import Intent, IntentResult


@pytest.mark.parametrize(
    "text,expected",
    [
        ("差旅报销标准是什么", Intent.KNOWLEDGE_QA),
        ("张三买了哪些产品", Intent.BUSINESS_QUERY),
        ("帮我取消这张订单", Intent.TASK_EXECUTION),
        ("我要投诉，转人工", Intent.HUMAN_HANDOFF),
        ("今天天气不错", Intent.CHITCHAT),
    ],
)
async def test_fake_provider_classifies(text: str, expected: Intent):
    provider = FakeProvider()
    response = await provider.achat(
        [ChatMessage(role="user", content=text)], response_model=IntentResult
    )
    assert response.parsed is not None
    assert response.parsed.intent == expected


async def test_fake_provider_plain_answer():
    provider = FakeProvider()
    response = await provider.achat([ChatMessage(role="user", content="你好")])
    assert "离线模式" in response.content
