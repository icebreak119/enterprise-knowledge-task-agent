"""把 data/policies 下的制度文档入库。

用法（必须在仓库根目录执行，source 用相对路径，保证幂等键稳定）：

    python scripts/ingest_corpus.py

版本号从文件名里的 `-Vx.y` 解析；改了文档内容就升版本号，旧切片会被删除重建。
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

from app.db.session import async_session_factory
from app.rag import ingest_file

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "policies"
VERSION_RE = re.compile(r"-V(\d+\.\d+)")


def _version_of(path: Path) -> str | None:
    matched = VERSION_RE.search(path.stem)
    return f"V{matched.group(1)}" if matched else None


def _status_of(path: Path) -> str:
    """样例语料用文件名标注废止状态。

    真实系统里这个状态应当来自文档管理系统（DMS）或入库接口的显式字段，
    这里只是把"必须显式标记"这件事做出来。
    """
    return "deprecated" if "已废止" in path.name else "active"


async def main() -> int:
    files = sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        print(f"未在 {CORPUS_DIR} 找到任何 .md 文档")
        return 1

    async with async_session_factory() as session:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            result = await ingest_file(
                session,
                relative,
                version=_version_of(path),
                status=_status_of(path),
            )
            state = "跳过（已存在同版本）" if result.skipped else "写入"
            print(
                f"{state}  {relative}  version={_version_of(path)} "
                f"status={_status_of(path)}  切片={result.chunk_count}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
