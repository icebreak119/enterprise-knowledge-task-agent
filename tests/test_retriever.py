"""检索与引用：纯逻辑单测 + 需要数据库的集成测试（连不上自动跳过）。"""

import pytest

from app.rag.retriever import (
    RetrievedChunk,
    apply_version_penalty,
    build_context,
    cited_chunk_ids,
    search,
    should_refuse,
)


def _chunk(
    chunk_id: int, distance: float, version: str = "V2.0", status: str = "active"
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"【制度】条款内容 {chunk_id}",
        source="data/policies/demo.md",
        title="售后制度",
        section_path="二、保修服务政策",
        version=version,
        status=status,
        distance=distance,
        penalized_distance=distance,
    )


def test_should_refuse_when_nothing_retrieved():
    """一条都没召回到，必须拒答——不能让模型凭常识补。"""
    assert should_refuse([], 0.5) is True


def test_should_refuse_when_best_hit_too_far():
    assert should_refuse([_chunk(1, 0.62)], 0.5) is True


def test_should_not_refuse_when_close_enough():
    assert should_refuse([_chunk(1, 0.18)], 0.5) is False


def test_only_best_distance_matters():
    """排序靠后的远结果不该导致拒答，只看最像的那条。"""
    assert should_refuse([_chunk(1, 0.20), _chunk(2, 0.80)], 0.5) is False


def test_build_context_numbers_chunks_with_id():
    context = build_context([_chunk(7, 0.1), _chunk(9, 0.2)])
    assert context.startswith("[7] ")
    assert "[9] " in context


def test_build_context_empty():
    assert build_context([]) == ""


def test_cited_chunk_ids():
    assert cited_chunk_ids("保修 36 个月 [12]，退款 3-5 天 [13]。") == {12, 13}


def test_cited_chunk_ids_ignores_other_numbers():
    assert cited_chunk_ids("保修 36 个月") == set()


def test_penalty_pushes_deprecated_below_active():
    """废止版本即使语义更近，也要排在现行版本之后。"""
    old = _chunk(9, 0.25, "V1.1", "deprecated")
    current = _chunk(20, 0.26, "V2.0", "active")

    ranked = apply_version_penalty([old, current], 0.15)
    assert [c.chunk_id for c in ranked] == [20, 9]


def test_penalty_keeps_original_distance_intact():
    """惩罚只影响排序，不能篡改原始距离——拒答阈值和分析都依赖它。"""
    old = _chunk(9, 0.25, "V1.1", "deprecated")
    ranked = apply_version_penalty([old], 0.15)
    assert ranked[0].distance == 0.25
    assert ranked[0].penalized_distance == 0.40


def test_penalty_zero_keeps_original_order():
    old = _chunk(9, 0.25, "V1.1", "deprecated")
    current = _chunk(20, 0.26, "V2.0", "active")
    ranked = apply_version_penalty([current, old], 0)
    assert [c.chunk_id for c in ranked] == [20, 9]


def test_deprecated_is_not_deleted_by_penalty():
    """降权不是删除：废止版本仍要留在结果里，否则查不到历史条款。"""
    old = _chunk(9, 0.25, "V1.1", "deprecated")
    ranked = apply_version_penalty([old], 0.15)
    assert len(ranked) == 1
    assert ranked[0].is_deprecated


async def test_search_returns_results_sorted_by_distance():
    """集成测试：需要本机 PostgreSQL 与 embedding 接口，连不上就跳过。"""
    from sqlalchemy import text

    from app.db.session import async_session_factory

    try:
        async with async_session_factory() as session:
            await session.execute(text("select 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"数据库不可用，跳过检索集成测试：{exc}")

    async with async_session_factory() as session:
        chunks = await search(session, "旗舰型设备整机保修多久？", top_k=3)

    if not chunks:
        pytest.skip("库里没有切片，先跑 scripts/ingest_corpus.py")

    assert len(chunks) <= 3
    # 排序依据是惩罚后距离；原始距离不再单调（废止版本会被压下去）
    assert [c.penalized_distance for c in chunks] == sorted(
        c.penalized_distance for c in chunks
    )
    assert all(c.chunk_id and c.content for c in chunks)
