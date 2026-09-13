"""入库编排：load → chunk → embed → 落库。

放在这里而不是塞进路由，是为了让"入库"能被脚本、测试和后续的 HTTP 接口共用，
也方便在评测时反复重建索引。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.rag.chunker import chunk_document
from app.rag.embeddings import Embedder, OpenAICompatEmbedder
from app.rag.loader import load_document


@dataclass(frozen=True)
class IngestResult:
    """一次入库的结果，便于脚本打印与测试断言。

    Attributes:
        document_id: documents 表主键。
        source: 来源路径。
        chunk_count: 本次写入的切片数；跳过时为库里已有的数量。
        skipped: True 表示同 source + 同 version 已存在，本次未重复写入。
    """

    document_id: int
    source: str
    chunk_count: int
    skipped: bool


async def ingest_file(
    session: AsyncSession,
    path: str | Path,
    *,
    embedder: Embedder | None = None,
    version: str | None = None,
    status: str = "active",
) -> IngestResult:
    """把单个文件入库。

    契约（实现时按此验收）：
    1. 顺序固定：`load_document` → `chunk_document` → `embedder.aembed` → 落库。
    2. **幂等**：先按 `source` 查 Document。
       - 同 `source` 且同 `version` → 直接返回 `skipped=True`，不重写。
       - 同 `source` 但 `version` 不同 → 视为文档更新：删掉旧 chunks 再写新的，
         并更新 Document.version。**旧向量必须删**，否则检索会召回过期条款
         （这是"政策更新后仍引用旧规定"的直接原因）。
    3. 先 flush Document 拿到 id，再批量写 DocumentChunk，同一事务提交。
    4. 切片为空抛 `ValueError`——空文档几乎总是解析失败，静默入库会让后面的
       "检索不到"变得极难排查。
    5. 未配置 API Key（`settings.embedding_enabled` 为 False）时抛 `RuntimeError`，
       不要带着空 embedding 写库。

    实现时需要补的导入（现在不引入，避免出现未使用的导入）：
        from sqlalchemy import delete, select
        from app.models.document import Document
        from app.models.document_chunk import DocumentChunk
        from app.rag.chunker import chunk_document
        from app.rag.loader import load_document

    Args:
        session: 异步会话，由调用方控制事务边界。
        path: 文件路径。
        embedder: 向量化实现；为 None 时用 `OpenAICompatEmbedder`。
        version: 文档版本，来自文件名或调用方；None 时按"无版本"处理。
        status: active / deprecated。废止版本入库时必须显式标记，
            检索侧靠它做降权，否则过期条款会和现行条款同等被召回。
    """
    settings = get_settings()
    if not settings.embedding_enabled:
        raise RuntimeError("未配置 LLM_API_KEY，无法生成向量；拒绝写入空 embedding。")

    document = load_document(path)
    chunks = chunk_document(document)
    if not chunks:
        raise ValueError(f"文档没有切出任何内容，解析可能失败：{path}")

    embedder = embedder or OpenAICompatEmbedder(settings)
    vectors = await embedder.aembed([chunk.content for chunk in chunks])

    source = str(path)
    existing = (
        await session.execute(select(Document).where(Document.source == source))
    ).scalar_one_or_none()

    if (
        existing is not None
        and existing.version == version
        and existing.status == status
    ):
        count = (
            await session.execute(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_id == existing.id)
            )
        ).scalar_one()
        return IngestResult(existing.id, source, count, skipped=True)

    if existing is None:
        existing = Document(
            name=document.title, source=source, version=version, status=status
        )
        session.add(existing)
        await session.flush()
    else:
        # 换版本 = 文档已更新：旧向量必须删，否则检索会召回过期条款
        await session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == existing.id)
        )
        existing.version = version
        existing.status = status

    session.add_all(
        DocumentChunk(
            document_id=existing.id,
            content=chunk.content,
            metadata_=chunk.to_metadata(),
            embedding=vector,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    )
    await session.commit()
    return IngestResult(existing.id, source, len(chunks), skipped=False)
