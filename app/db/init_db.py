import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.base import Base
import app.models  # noqa: F401  确保所有模型注册到 Base.metadata

logger = logging.getLogger(__name__)

async def _ensure_vector_extension(engine: AsyncEngine) -> bool:
    """确保 pgvector 扩展存在。

    部分环境（无扩展权限的托管 PG、未装插件的本机实例）会创建失败，
    此时返回 False：迁移里建 document_chunks 会失败，但其余表仍可建立。
    """
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        return True
    except Exception as exc:  # noqa: BLE001  任何失败都按“不支持向量”处理
        logger.warning("pgvector 不可用，向量检索将不可用：%s", exc)
        return False


async def init_db(engine: AsyncEngine) -> None:
    """启动自检：确认数据库已迁移到最新，不再自己建表。

    表结构一律由 Alembic 管理（见 alembic/versions）。这里只做两件事：
    1. 确保 pgvector 扩展存在；
    2. 发现表缺失时给出明确错误，而不是让 SQL 抛一个看不懂的
       "relation does not exist"。
    """
    await _ensure_vector_extension(engine)

    async with engine.begin() as conn:
        existing = {
            row[0]
            for row in await conn.execute(
                text("select tablename from pg_tables where schemaname = 'public'")
            )
        }

    expected = {table.name for table in Base.metadata.sorted_tables}
    missing = expected - existing
    if missing:
        raise RuntimeError(
            f"数据库缺少表 {sorted(missing)}，请先执行：alembic upgrade head"
        )
