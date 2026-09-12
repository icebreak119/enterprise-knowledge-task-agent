from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import get_settings
from app.db.base import Base

# 维度统一来自配置：更换 embedding 模型时必须同步 settings.embedding_dim
# 并用 Alembic 重建该列，否则新旧向量维度不一致会导致写入失败。
VECTOR_DIM = get_settings().embedding_dim


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