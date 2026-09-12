import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.base import Base
import app.models  # noqa: F401  确保所有模型注册到 Base.metadata

logger = logging.getLogger(__name__)

VECTOR_TABLE = "document_chunks"


async def _ensure_vector_extension(engine: AsyncEngine) -> bool:
    """尝试创建 pgvector 扩展。

    部分环境（无扩展权限的托管 PG、未装插件的本机实例）会创建失败，
    此时返回 False，由调用方跳过依赖 vector 类型的表，其余表照常建立。
    """
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        return True
    except Exception as exc:  # noqa: BLE001  任何失败都按“不支持向量”处理
        logger.warning("pgvector 不可用，跳过 %s 表：%s", VECTOR_TABLE, exc)
        return False


async def init_db(engine: AsyncEngine) -> None:
    """创建 pgvector 扩展并建表（Day 1 用 create_all，后续迁移切到 Alembic）。"""
    has_vector = await _ensure_vector_extension(engine)

    tables = [
        table
        for table in Base.metadata.sorted_tables
        if has_vector or table.name != VECTOR_TABLE
    ]

    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=tables)
        )
