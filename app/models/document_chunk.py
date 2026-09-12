from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# 与 embedding 模型维度对应，Day 4 起接入 RAG 后按所选模型调整。
VECTOR_DIM = 1536


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    content: Mapped[str] = mapped_column(Text)
    # 列名固定为 metadata，属性用 metadata_ 避免与方法名冲突
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON)
    embedding: Mapped[list | None] = mapped_column(Vector(VECTOR_DIM))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())