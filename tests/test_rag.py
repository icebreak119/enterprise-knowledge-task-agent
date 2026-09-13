"""RAG 入库链路测试：切分与解析不依赖网络，可纯单测。

入库（ingest）涉及真实数据库与 embedding 接口，放集成测试，这里不覆盖。
"""

from pathlib import Path

import pytest

from app.rag import Chunk, chunk_document, load_document
from app.rag.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP

CORPUS = (
    Path(__file__).resolve().parents[1] / "data" / "policies" / "售后服务管理制度-V2.0.md"
)
DEPRECATED = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "policies"
    / "售后服务管理制度-V1.1-已废止.md"
)


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


def test_default_params_are_consistent():
    assert 0 <= DEFAULT_OVERLAP < DEFAULT_CHUNK_SIZE


def test_loader_reads_title_and_source():
    doc = load_document(CORPUS)
    assert doc.title == "公司售后服务管理制度（样例）"
    assert doc.source == str(CORPUS)
    assert doc.metadata["suffix"] == ".md"


def test_loader_rejects_unsupported_suffix(tmp_path):
    bad = tmp_path / "a.pdf"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持"):
        load_document(bad)


def test_loader_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_document("data/policies/不存在的文件.md")


def test_chunk_never_crosses_headings():
    """绝不能跨标题拼接——否则不同条款会混进同一个 chunk，检索到就是噪音。"""
    doc = load_document(CORPUS)
    chunks = chunk_document(doc)

    for chunk in chunks:
        # 保修除外情形与投诉处理是不同小节，不应出现在同一片里
        assert not (
            "保修除外情形" in chunk.section_path and "投诉处理流程" in chunk.section_path
        )


def test_chunk_keeps_section_path_in_content():
    """标题路径要进正文：否则"不保修"这种短 chunk 检索到也不知道在说什么。"""
    doc = load_document(CORPUS)
    chunks = chunk_document(doc)

    assert all(chunk.content.startswith("【") for chunk in chunks)
    assert any("保修期限与范围" in chunk.section_path for chunk in chunks)


def test_table_rows_stay_with_header():
    """表格整体成块；被切开时每片都要带表头，否则后半段列含义全丢。"""
    doc = load_document(CORPUS)
    chunks = chunk_document(doc, chunk_size=200, overlap=32)

    table_chunks = [c for c in chunks if "保修期限与范围" in c.section_path]
    assert table_chunks, "应当切出保修期限小节"

    for chunk in table_chunks:
        if "|" in chunk.content:
            assert "产品类别" in chunk.content, "含表格行的 chunk 必须带上表头"


def test_char_offsets_point_back_to_source():
    """char_start/char_end 必须能对应回原文，否则没法做原文定位。"""
    doc = load_document(CORPUS)
    for chunk in chunk_document(doc):
        body = chunk.content.split("\n", 1)[1]  # 去掉拼进去的标题行
        assert doc.text[chunk.char_start : chunk.char_end].rstrip() == body.rstrip()


def test_blank_only_blocks_are_dropped():
    doc = load_document(CORPUS)
    assert all(chunk.content.strip() for chunk in chunk_document(doc))


@pytest.mark.parametrize(
    ("chunk_size", "overlap"), [(512, 512), (512, 600), (0, 0), (100, -1)]
)
def test_invalid_window_params(chunk_size, overlap):
    doc = load_document(CORPUS)
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=chunk_size, overlap=overlap)


def test_deprecated_version_produces_different_warranty_text():
    """V1.1 与 V2.0 必须切出不同内容，版本冲突 Bad Case 才有意义。"""
    current = load_document(CORPUS)
    deprecated = load_document(DEPRECATED)

    current_chunks = [c.content for c in chunk_document(current)]
    deprecated_chunks = [c.content for c in chunk_document(deprecated)]

    assert any("36 个月" in c for c in current_chunks)
    assert any("旗舰型设备 | 24 个月" in c for c in deprecated_chunks)
