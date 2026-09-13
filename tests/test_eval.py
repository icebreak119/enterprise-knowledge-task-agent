"""评测指标与数据集的单元测试（不依赖数据库）。"""

import pytest

from app.eval import aggregate, chunk_matches, evaluate_citation, evaluate_retrieval
from app.eval.dataset import EvalCase, load_cases
from app.rag.retriever import RetrievedChunk


def _chunk(chunk_id: int, version: str, section: str, distance: float = 0.2):
    return RetrievedChunk(
        chunk_id=chunk_id,
        content="条款",
        source=f"data/policies/制度-{version}.md",
        title="售后制度",
        section_path=section,
        version=version,
        distance=distance,
    )


def test_load_bundled_cases():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cases = load_cases(root / "evals" / "cases.jsonl")
    assert len(cases) >= 40
    assert all(c.id and c.question for c in cases)
    # 题型要齐，否则指标覆盖不到
    assert {c.category for c in cases} == {
        "direct",
        "paraphrase",
        "version_trap",
        "should_refuse",
        "historical_lookup",
        "cross_section",
    }
    # id 不能重复，否则失败样例定位不到
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_chunk_matches_requires_all_needles():
    chunk = _chunk(1, "V2.0", "二、保修服务政策 > 2.1 保修期限与范围")
    assert chunk_matches(chunk, ["V2.0", "2.1"])
    assert not chunk_matches(chunk, ["V2.0", "三、退换货"])
    assert not chunk_matches(chunk, ["V1.1"])


def test_recall_is_one_when_evidence_found():
    case = EvalCase(
        id="c1", question="保修多久", evidence=[["V2.0", "2.1"]], expected_version="V2.0"
    )
    result = evaluate_retrieval(
        case, [_chunk(1, "V2.0", "二、保修服务政策 > 2.1 保修期限与范围")], False
    )
    assert result.recall == 1.0
    assert result.version_ok is True


def test_version_error_detected_when_old_version_ranks_first():
    """旧版本排第一必须被判为失败——这就是要修的那个问题。"""
    case = EvalCase(
        id="c2", question="紧急故障多久到现场", evidence=[["V2.0", "四、服务响应标准"]],
        expected_version="V2.0",
    )
    result = evaluate_retrieval(
        case,
        [
            _chunk(9, "V1.1", "四、服务响应标准", 0.25),
            _chunk(20, "V2.0", "四、服务响应标准", 0.26),
        ],
        False,
    )
    assert result.version_ok is False
    assert result.is_failure


def test_refuse_mismatch_is_failure():
    case = EvalCase(id="c3", question="团建去哪", should_refuse=True)
    assert evaluate_retrieval(case, [], False).is_failure
    assert not evaluate_retrieval(case, [], True).is_failure


def test_citation_accuracy():
    case = EvalCase(id="c4", question="保修多久", evidence=[["V2.0", "2.1"]])
    retrieved = [_chunk(5, "V2.0", "二、保修服务政策 > 2.1 保修期限与范围")]
    base = evaluate_retrieval(case, retrieved, False)

    cited = evaluate_citation(_with_cited(base, {5}))
    assert cited.citation_ok is True

    not_cited = evaluate_citation(_with_cited(base, {999}))
    assert not_cited.citation_ok is False


def _with_cited(result, cited_ids):
    from dataclasses import replace

    return replace(result, cited_ids=cited_ids)


def test_aggregate_ignores_none_samples():
    results = [
        evaluate_retrieval(EvalCase(id="a", question="x", should_refuse=True), [], True),
        evaluate_retrieval(EvalCase(id="b", question="y", should_refuse=True), [], False),
    ]
    summary = aggregate(results)
    assert summary["refuse_accuracy"] == 0.5
    assert summary["version_accuracy"] is None  # 没有可评估样本


def test_aggregate_on_empty():
    assert aggregate([]) == {}
