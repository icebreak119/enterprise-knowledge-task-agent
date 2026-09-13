from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 相对路径会按"当前工作目录"解析，换目录跑（CI、别的机器、别的终端）就读不到 .env。
# 用绝对路径锚定到仓库根目录，保证在哪里启动行为一致。
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """应用配置，通过环境变量或项目根目录下 .env 覆盖。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE), env_file_encoding="utf-8", extra="ignore"
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
    # 厂商私有参数的逃生舱：JSON 对象，原样并入请求体，留空则不传。
    # 各家关闭思维链的字段并不统一，写死一种会绑死厂商：
    #   智谱 GLM      {"thinking": {"type": "disabled"}}
    #   阿里云 Qwen3  {"enable_thinking": false}
    #   本地 vLLM     视启动参数而定
    llm_extra_body: str = ""

    # Embedding（换模型时必须同步 embedding_dim 并重建向量列）
    # 实测：阿里云百炼 qwen3.7-text-embedding / -flash 均为 1024 维
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_base_url: str = ""
    # 批量上限由厂商接口决定，超过会被拒
    embedding_batch_size: int = 32

    @property
    def embedding_enabled(self) -> bool:
        """是否配置了可用的 embedding 凭据；未配置时入库链路不可用。"""
        return bool(self.llm_api_key)

    # 检索
    rag_top_k: int = 5
    # 余弦距离阈值：超过则认为"库里没有足够相近的资料"→ 拒答。
    # 这是保守初值，必须由评测集的距离分布标定后才算数。
    rag_max_distance: float = 0.5
    # 已废止版本切片的距离惩罚：降权但不删除，保留"旧版本说过什么"的可追溯性。
    # 这个值应由评测集扫参确定，不是拍脑袋的最优值。
    rag_deprecated_penalty: float = 0.15
    # 降权后废止版本会被挤下去，因此需要多召回一些再重排
    rag_oversample: int = 3

    @property
    def llm_enabled(self) -> bool:
        """是否配置了可用的 LLM 凭据；未配置时自动降级到离线 Provider。"""
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
