"""RAG 骨架的冒烟测试：保证包可导入、数据结构契约不漂移。

实现填完后，这里应当补上真正的切分与入库测试。
"""

import pytest

from app.rag import Chunk, IngestResult, LoadedDocument, chunk_document, ingest_file, load_document
from app.rag.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP


def test_public_api_importable():
    """骨架能被导入，且默认参数自洽（overlap 必须小于窗口）。"""
    assert 0 <= DEFAULT_OVERLAP < DEFAULT_CHUNK_SIZE


def test_chunk_to_metadata_excludes_content():
    """正文不进 metadata：它单独占 content 列，重复存一份只会让库变大。"""
    chunk = Chunk(
        content="差旅住宿标准为一线城市每晚 500 元。",
        source="policies/travel.md",
        title="员工手册",
        section_path="差旅管理 > 住宿标准",
        chunk_index=3,
        char_start=120,
        char_end=141,
    )
    metadata = chunk.to_metadata()

    assert "content" not in metadata
    assert metadata["section_path"] == "差旅管理 > 住宿标准"
    assert metadata["chunk_index"] == 3
    assert metadata["char_start"] == 120


def test_loaded_document_defaults_to_empty_metadata():
    doc = LoadedDocument(source="a.md", title="员工手册", text="# 差旅")
    assert doc.metadata == {}


def test_ingest_result_is_frozen():
    result = IngestResult(document_id=1, source="a.md", chunk_count=2, skipped=False)
    with pytest.raises(Exception):
        result.chunk_count = 3  # type: ignore[misc]


def test_stubs_raise_not_implemented():
    """占位实现：填完后这条会失败，届时替换成真实断言。"""
    with pytest.raises(NotImplementedError):
        load_document("a.md")
    with pytest.raises(NotImplementedError):
        chunk_document(LoadedDocument(source="a.md", title="t", text="正文"))


async def test_ingest_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        await ingest_file(None, "a.md")  # type: ignore[arg-type]
