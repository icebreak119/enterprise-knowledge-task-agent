import pytest

from app.core.config import Settings
from app.llm.base import ChatMessage
from app.llm.errors import LLMRateLimitError, LLMParseError
from app.llm.openai_provider import (
    OpenAICompatProvider,
    _parse_extra_body,
    _should_retry,
    _to_llm_error,
)
from app.schemas.intent import Intent, IntentResult


class _Message:
    def __init__(self, content: str, parsed=None):
        self.content = content
        self.parsed = parsed
        self.tool_calls = None


class _Choice:
    def __init__(self, message):
        self.message = message


class _Raw:
    def __init__(self, message):
        self.choices = [_Choice(message)]
        self.model = "test-model"
        self.usage = None


class _StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _provider() -> OpenAICompatProvider:
    settings = Settings(
        llm_api_key="test-key",
        llm_base_url="http://localhost:1/v1",
        llm_model="test-model",
        llm_max_retries=1,
    )
    return OpenAICompatProvider(settings)


async def test_json_schema_path(monkeypatch):
    provider = _provider()

    async def fake_request(self, method, **kwargs):
        assert method == "parse"
        return _Raw(_Message(content="", parsed=IntentResult(intent=Intent.CHITCHAT)))

    monkeypatch.setattr(OpenAICompatProvider, "_request", fake_request)
    response = await provider.achat(
        [ChatMessage(role="user", content="hi")], response_model=IntentResult
    )
    assert response.parsed is not None
    assert response.parsed.intent is Intent.CHITCHAT
    assert provider._json_schema_supported is True


async def test_fallback_to_json_object(monkeypatch):
    """厂商不支持 json_schema 时，应降级为 json_object 并手动校验。"""
    provider = _provider()
    seen: dict = {}

    async def fake_request(self, method, **kwargs):
        if method == "parse":
            raise RuntimeError("unsupported response_format")
        seen.update(kwargs)
        payload = '{"intent": "knowledge_qa", "need_retrieval": true, "confidence": 0.9}'
        return _Raw(_Message(content=payload))

    monkeypatch.setattr(OpenAICompatProvider, "_request", fake_request)
    response = await provider.achat(
        [ChatMessage(role="user", content="报销标准")], response_model=IntentResult
    )
    assert seen["response_format"] == {"type": "json_object"}
    assert response.parsed is not None
    assert response.parsed.intent is Intent.KNOWLEDGE_QA
    assert response.parsed.need_retrieval is True
    assert provider._json_schema_supported is False


async def test_parse_error_raises(monkeypatch):
    provider = _provider()

    async def fake_request(self, method, **kwargs):
        if method == "parse":
            raise RuntimeError("unsupported response_format")
        return _Raw(_Message(content="这不是 JSON"))

    monkeypatch.setattr(OpenAICompatProvider, "_request", fake_request)
    with pytest.raises(LLMParseError):
        await provider.achat(
            [ChatMessage(role="user", content="x")], response_model=IntentResult
        )


async def test_rate_limit_does_not_disable_json_schema(monkeypatch):
    """429 是可恢复错误，不能让 json_schema 被永久判定为不支持。"""
    provider = _provider()

    async def fake_request(self, method, **kwargs):
        if method == "parse":
            raise _StatusError(429)
        raise AssertionError("不应在限流时走降级路径")

    monkeypatch.setattr(OpenAICompatProvider, "_request", fake_request)
    with pytest.raises(LLMRateLimitError):
        await provider.achat(
            [ChatMessage(role="user", content="x")], response_model=IntentResult
        )
    assert provider._json_schema_supported is None


async def test_extra_body_merged(monkeypatch):
    """厂商私有参数应原样并入请求体（如 Qwen 的 enable_thinking）。"""
    provider = OpenAICompatProvider(
        Settings(
            llm_api_key="test-key",
            llm_model="test-model",
            llm_max_retries=1,
            llm_extra_body='{"enable_thinking": false}',
        )
    )
    seen: dict = {}

    async def fake_request(self, method, **kwargs):
        seen.update(kwargs)
        return _Raw(_Message(content="ok"))

    monkeypatch.setattr(OpenAICompatProvider, "_request", fake_request)
    await provider.achat([ChatMessage(role="user", content="x")])
    assert seen["extra_body"] == {"enable_thinking": False}


def test_invalid_extra_body_fails_fast():
    assert _parse_extra_body("") == {}
    with pytest.raises(ValueError):
        _parse_extra_body("{not json}")
    with pytest.raises(ValueError):
        _parse_extra_body("[1, 2]")


def test_retry_policy():
    assert _should_retry(_StatusError(429)) is True
    assert _should_retry(_StatusError(500)) is True
    assert _should_retry(_StatusError(400)) is False


def test_error_mapping():
    assert isinstance(_to_llm_error(_StatusError(429)), LLMRateLimitError)
