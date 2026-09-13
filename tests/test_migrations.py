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
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config("alembic.ini")
    script = ScriptDirectory.from_config(config)
    heads = script.get_heads()
    assert len(heads) == 1, f"存在多个 head：{heads}"
