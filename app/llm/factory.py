import logging
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.fake_provider import FakeProvider
from app.llm.openai_provider import OpenAICompatProvider

logger = logging.getLogger(__name__)


@lru_cache
def get_llm_provider() -> LLMProvider:
    """按配置返回 Provider 单例；未配置 API key 时自动降级为离线实现。"""
    settings: Settings = get_settings()

    if not settings.llm_enabled:
        logger.warning("未配置 LLM_API_KEY，使用 FakeProvider（离线模式）")
        return FakeProvider()

    if settings.llm_provider == "openai":
        return OpenAICompatProvider(settings)

    raise ValueError(f"不支持的 llm_provider：{settings.llm_provider}")
