"""知识检索工具：将 search_knowledge 包装为统一 Tool 接口。

这个工具放在 registry 之外而不是直接调用 retriever.search，
是为了让知识检索和业务工具用同一套注册、调用和审计链路。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.retriever import RetrievedChunk, search as rag_search

SEARCH_KNOWLEDGE_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "用户问题原文"},
    },
    "required": ["query"],
}


async def search_knowledge(params: dict, session: AsyncSession) -> str:
    """从售后制度文档中检索相关内容。"""
    chunks: list[RetrievedChunk] = await rag_search(session, params["query"])
    if not chunks:
        return "未找到相关制度条款。"
    parts = [f"共找到 {len(chunks)} 条相关条款（按相关度排序）："]
    for c in chunks[:5]:
        note = "【已废止】" if c.is_deprecated else ""
        parts.append(
            f"  [{c.chunk_id}] {note}{c.version} {c.section_path}\n"
            f"    {c.content[:120]}"
        )
    return "\n".join(parts)