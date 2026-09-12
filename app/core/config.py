from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，通过环境变量或项目根目录下 .env 覆盖。"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Enterprise Agent"
    debug: bool = False

    database_url: str = (
        "postgresql+asyncpg://agent:agent@localhost:5432/enterprise_agent"
    )
    redis_url: str = "redis://localhost:6379/0"

    # LLM（OpenAI 兼容协议，Day 2 起使用；base_url 可切 GLM / DeepSeek / 本地 vLLM）
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048  # 推理模型会先输出思维链，需留足余量
    llm_timeout: float = 60
    llm_max_retries: int = 3
    # enabled / disabled / auto：auto 表示不向厂商传 thinking 参数。
    # GLM-4.x 这类推理模型默认会输出思维链并占用 max_tokens，建议显式 disabled。
    llm_thinking: str = "auto"

    # Embedding（Day 4 起使用；换模型时必须同步 embedding_dim 并重建向量列）
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_base_url: str = ""

    @property
    def llm_enabled(self) -> bool:
        """是否配置了可用的 LLM 凭据；未配置时自动降级到离线 Provider。"""
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
