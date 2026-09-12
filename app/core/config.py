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

    # LLM 相关（Day 2 起使用）
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()