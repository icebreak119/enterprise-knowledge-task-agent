"""评测指标。

刻意分成两层：
- 检索层指标（recall / version / refuse）只跑 search，便宜、可高频跑；
- 引用层指标要调 LLM 生成回答，贵，单独跑。

不要把所有指标绑在一次运行里，否则成本高到没人愿意跑第二次。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from app.eval.dataset import EvalCase
from app.rag.retriever import RetrievedChunk


def chunk_matches(chunk: RetrievedChunk, needles: list[str]) -> bool:
    """证据定位串里的每个关键词都出现在来源或章节路径中才算命中。"""
    haystack = f"{chunk.source} {chunk.section_path}"
    return all(needle in haystack for needle in needles)


def evidence_hits(case: EvalCase, retrieved: list[RetrievedChunk]) -> list[bool]:
    """按 case.evidence 的顺序，返回每条证据是否被召回到。"""
    return [
        any(chunk_matches(chunk, needles) for chunk in retrieved)
        for needles in case.evidence
    ]


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    refused: bool = False
    cited_ids: set[int] = field(default_factory=set)
    recall: float = 0.0
    version_ok: bool | None = None
    refuse_ok: bool = True
    citation_ok: bool | None = None

    @property
    def is_failure(self) -> bool:
        return not self.refuse_ok or self.recall < 1.0 or self.version_ok is False


def evaluate_retrieval(
    case: EvalCase, retrieved: list[RetrievedChunk], refused: bool
) -> CaseResult:
    """只评估检索与拒答，不评估回答。"""
    hits = evidence_hits(case, retrieved)
    recall = sum(hits) / len(hits) if hits else (0.0 if not case.should_refuse else 1.0)

    version_ok: bool | None = None
    if case.expected_version and retrieved:
        version_ok = retrieved[0].version == case.expected_version

    return CaseResult(
        case=case,
        retrieved=retrieved,
        refused=refused,
        recall=recall,
        version_ok=version_ok,
        refuse_ok=refused == case.should_refuse,
    )


def evaluate_citation(result: CaseResult) -> CaseResult:
    """在检索结果基础上，补充引用正确率（需要回答已生成）。"""
    case = result.case
    if case.should_refuse or not case.evidence:
        return replace(result, citation_ok=None)

    expected_ids = {
        chunk.chunk_id
        for chunk in result.retrieved
        for needles in case.evidence
        if chunk_matches(chunk, needles)
    }
    return replace(result, citation_ok=bool(expected_ids & result.cited_ids))


def aggregate(results: list[CaseResult]) -> dict[str, float | None]:
    """汇总指标。None 表示该指标本次没有可评估的样本。"""
    total = len(results)
    if not total:
        return {}

    def _rate(values: list[bool | None]) -> float | None:
        scored = [v for v in values if v is not None]
        if not scored:
            return None
        return round(sum(scored) / len(scored), 4)

    return {
        "cases": total,
        "recall_at_k": round(sum(r.recall for r in results) / total, 4),
        "version_accuracy": _rate([r.version_ok for r in results]),
        "refuse_accuracy": _rate([r.refuse_ok for r in results]),
        "citation_accuracy": _rate([r.citation_ok for r in results]),
        "failures": sum(1 for r in results if r.is_failure),
    }
