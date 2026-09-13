"""检索：把问题变成带来源的切片列表。

只做检索，不做生成——生成仍走 app/llm，两者可以分别单测和评测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.rag.embeddings import Embedder, OpenAICompatEmbedder

# 回答里引用的切片编号，形如 [12]
_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class RetrievedChunk:
    """一条检索结果。字段够回答时拼引用，也够事后做归因分析。

    Attributes:
        chunk_id: document_chunks 主键，同时作为回答里的引用编号。
        content: 切片正文（已含标题路径）。
        source: 来源文件路径。
        title: 文档标题。
        section_path: 标题路径，如 "二、保修服务政策 > 2.1 保修期限与范围"。
        version: 文档版本，用于判断有没有引到已废止版本。
        distance: 余弦距离，越小越相关。
    """

    chunk_id: int
    content: str
    source: str
    title: str
    section_path: str
    version: str | None
    status: str = "active"
    # 原始余弦距离：用于拒答阈值与事后分析，不因降权而改变
    distance: float = 0.0
    # 排序用距离：废止版本会被加上惩罚，仅影响顺序
    penalized_distance: float = 0.0

    @property
    def is_deprecated(self) -> bool:
        return self.status == "deprecated"


async def search(
    session: AsyncSession,
    query: str,
    *,
    embedder: Embedder | None = None,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """向量检索 top_k 条最相关切片。

    用原生 SQL 而非 ORM：embedding 与查询向量的距离运算走 SQL 最直观，
    也避免 ORM 绑定向量类型时的适配问题；已实测可用。

    结果按距离升序返回（越靠前越相关）。空库或未入库时返回空列表，
    由调用方决定是拒答还是降级。
    """
    settings = settings or get_settings()
    embedder = embedder or OpenAICompatEmbedder(settings)
    top_k = top_k if top_k is not None else settings.rag_top_k

    vectors = await embedder.aembed([query])
    if not vectors:
        return []

    rows = (
        await session.execute(
            text(
                """
                select c.id,
                       c.content,
                       coalesce(c.metadata ->> 'source', '')       as source,
                       coalesce(c.metadata ->> 'title', '')        as title,
                       coalesce(c.metadata ->> 'section_path', '') as section_path,
                       d.version,
                       d.status,
                       c.embedding <=> :vec                         as distance
                from document_chunks c
                join documents d on d.id = c.document_id
                order by c.embedding <=> :vec
                limit :limit
                """
            ),
            # 多取一些：降权后废止版本会被挤下去，只取 top_k 会漏掉现行版本
            {"vec": str(vectors[0]), "limit": top_k * settings.rag_oversample},
        )
    ).fetchall()

    chunks = [
        RetrievedChunk(
            chunk_id=row.id,
            content=row.content,
            source=row.source,
            title=row.title,
            section_path=row.section_path,
            version=row.version,
            status=row.status or "active",
            distance=float(row.distance),
            penalized_distance=float(row.distance),
        )
        for row in rows
    ]
    return apply_version_penalty(chunks, settings.rag_deprecated_penalty)[:top_k]


def apply_version_penalty(
    chunks: list[RetrievedChunk], penalty: float
) -> list[RetrievedChunk]:
    """给已废止版本的切片加距离惩罚后重排。

    为什么要重排而不是直接过滤掉废止版本：过滤会让"旧版本说过什么"彻底查不到，
    而这个场景里对比新旧差异恰恰是有价值的。降权则保留可追溯性，
    只是不再让它抢在现行版本前面。

    penalty 是可调参数，它本身就是实验结果——需要用评测集扫一遍找合适的值。
    """
    if penalty <= 0:
        return list(chunks)

    penalized = [
        replace(chunk, penalized_distance=chunk.distance + penalty)
        if chunk.is_deprecated
        else chunk
        for chunk in chunks
    ]
    return sorted(penalized, key=lambda chunk: chunk.penalized_distance)


def build_context(chunks: list[RetrievedChunk]) -> str:
    """把切片拼成给模型看的资料块，每条带 [编号]。

    编号必须用 chunk_id：这样模型标注的引用能直接对应回库里那一条，
    事后可以验证"引用的到底是哪一片"。
    """
    return "\n\n".join(f"[{chunk.chunk_id}] {chunk.content}" for chunk in chunks)


def should_refuse(chunks: list[RetrievedChunk], max_distance: float) -> bool:
    """是否应当拒答。

    **拒答由后端判定，不交给模型。** 模型"觉得自己不知道"是不可靠的，
    而"库里没有足够相近的资料"是一个可以用阈值表达的事实。

    阈值需要由评测集标定（看相关/不相关样本的距离分布再定义），
    这里的默认值只是保守初值。

    注意比较的是**重排后第一条的原始距离**，不是全集合里最小的原始距离：
    因此当唯一相近的只有已废止版本时（它被降权排到了后面），这里会判为拒答。
    这是有意为之——宁可说"没找到依据"，也不拿废止条款作答。
    """
    if not chunks:
        return True
    return chunks[0].distance > max_distance


def cited_chunk_ids(answer: str) -> set[int]:
    """从回答里抽出被引用到的切片编号。

    抽不出来不代表回答错了（可能确实不需要引用），
    但"给了资料却一个引用都没有"通常是问题的信号，值得记进评测指标。
    """
    return {int(match) for match in _CITATION_RE.findall(answer)}
