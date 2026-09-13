"""向量化：把文本批量转成向量。

走和 LLM 一样的 OpenAI 兼容协议（`/embeddings`），换厂商只改配置。
维度必须与 `document_chunks.embedding` 列一致，否则写入直接报错——
这是本项目最容易踩的坑，见 config.embedding_dim 的注释。
"""

from __future__ import annotations

from typing import Protocol

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings, get_settings
from app.llm.openai_provider import _should_retry


class Embedder(Protocol):
    """向量化接口。业务代码只依赖它，不直接 import SDK。"""

    @property
    def dim(self) -> int:
        """向量维度，写入前用它校验与库表列是否一致。"""
        ...

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """把文本列表转成向量列表，顺序与输入一一对应。"""
        ...


class OpenAICompatEmbedder:
    """OpenAI 兼容协议的 /embeddings 实现。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        batch_size: int | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._batch_size = batch_size or self._settings.embedding_batch_size
        # 与 LLM 共用密钥与站点；base_url 为空时 SDK 会回落官方地址
        self._client = _build_client(self._settings)

    @property
    def dim(self) -> int:
        """配置声明的维度。

        注意：这是"我们以为的维度"，可能与模型实际返回的不一致。
        首次真实调用后应当用 `len(vec)` 校验一次，不一致要立刻报错，
        而不是等写入 pgvector 时才炸。
        """
        return self._settings.embedding_dim

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。

        契约（实现时按此验收）：
        1. 按 `batch_size` 切批，逐批请求；空输入直接返回 `[]`，不要发请求。
        2. 结果顺序必须与输入一致（厂商可能乱序返回，按返回的 index 字段归位）。
        3. 每批校验 `len(vec) == self.dim`，不一致抛 `ValueError` 并带上模型名与实测维度。
        4. 失败重试只针对连接/超时/429/5xx，与 `app/llm/openai_provider.py` 的策略保持一致。
        5. 拼接各批结果后返回，长度必须等于输入长度。

        Raises:
            ValueError: 维度与配置不符。
        """
        if not texts:
            return []

        vectors: list[list[float] | None] = [None] * len(texts)
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            raw = await self._request(batch)
            for item in raw.data:
                vector = list(item.embedding)
                if len(vector) != self.dim:
                    raise ValueError(
                        f"embedding 维度不符：模型 {self._settings.embedding_model} "
                        f"实测 {len(vector)} 维，配置 embedding_dim={self.dim}。"
                        "请同步 settings.embedding_dim 并重建向量列。"
                    )
                vectors[start + item.index] = vector

        return [vector for vector in vectors if vector is not None]

    async def _request(self, batch: list[str]):
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(self._settings.llm_max_retries),
            # 与 LLM 侧保持一致：免费额度 429 常持续数秒，窗口太窄等于白重试
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_exception(_should_retry),
        ):
            with attempt:
                return await self._client.embeddings.create(
                    model=self._settings.embedding_model, input=batch
                )


def _build_client(settings: Settings):
    """构造 AsyncOpenAI 客户端。

    与 app/llm/openai_provider.py 用同一套凭据；单独抽出来是为了让测试能替换。
    """
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.embedding_base_url or settings.llm_base_url or None,
        timeout=settings.llm_timeout,
        max_retries=0,  # 重试在 aembed 内自己做
    )
