"""防止"改了模型却忘了生成迁移"再次发生。

需要本机 PostgreSQL，连不上时自动跳过。
"""

import pytest
from sqlalchemy import text


async def test_schema_matches_models(db_session):
    """模型里声明的表必须都在库里存在。

    少了就说明有人加了模型/字段却没生成迁移——这个债已经欠过两次，
    所以加一条测试守住它。
    """
    import app.models  # noqa: F401
    from app.db.base import Base

    try:
        existing = {
            row[0]
            for row in await db_session.execute(
                text("select tablename from pg_tables where schemaname = 'public'")
            )
        }
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据库不可用，跳过迁移一致性检查：{exc}")

    expected = {table.name for table in Base.metadata.sorted_tables}
    missing = expected - existing
    assert not missing, f"缺少表 {sorted(missing)}，请先执行 alembic upgrade head"


def test_alembic_has_single_head():
    """只能有一个 head，多个分支会让 upgrade 行为不确定。"""
    from pathlib import Path

    from alembic.config import Config
    from alembic.script import ScriptDirectory

    # 用绝对路径：否则从仓库根目录以外跑 pytest 会找不到配置
    ini_path = Path(__file__).resolve().parents[1] / "alembic.ini"
    script_dir = ScriptDirectory.from_config(Config(str(ini_path)))
    heads = script_dir.get_heads()
    assert len(heads) == 1, f"存在多个 head：{heads}"


async def test_hnsw_index_exists(db_session):
    """HNSW 索引必须存在，否则数据量上来后的检索性能不可接受。"""
    try:
        rows = (
            await db_session.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE tablename = 'document_chunks' AND indexdef LIKE '%hnsw%'"
                )
            )
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"无法查询索引信息：{exc}")

    assert len(rows) >= 1, (
        "document_chunks 缺少 HNSW 索引，请执行 alembic upgrade head"
    )
    assert "hnsw" in rows[0][0]
    assert "vector_cosine_ops" in rows[0][0]
