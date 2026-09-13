"""add hnsw index on embedding

Revision ID: 56eaa6294d23
Revises: c78af1e0ef36
Create Date: 2026-09-13 23:06:13.543168

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '56eaa6294d23'
down_revision: Union[str, Sequence[str], None] = 'c78af1e0ef36'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


INDEX_NAME = "idx_document_chunks_embedding_hnsw"


def upgrade() -> None:
    """HNSW 索引：加速余弦距离（<=>）的近似最近邻检索。

    参数说明：
    - vector_cosine_ops — 与检索使用的 <=> 运算符一致
    - m = 16 — pgvector 默认，平衡精度与构建速度
    - ef_construction = 64 — pgvector 默认

    IF NOT EXISTS 保证幂等：已建索引的表上再次 upgrade 不会报错。
    """
    op.execute(
        # 将长 SQL 拼接成一行，避免 alembic 对换行的处理问题
        (
            "CREATE INDEX IF NOT EXISTS %(name)s "
            "ON document_chunks "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
        % {"name": INDEX_NAME}
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS %s" % INDEX_NAME)
