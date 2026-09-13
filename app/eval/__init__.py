"""RAG 效果评测：数据集 + 指标。

用法见 scripts/run_eval.py。
"""

from app.eval.dataset import EvalCase, load_cases
from app.eval.metrics import (
    CaseResult,
    aggregate,
    chunk_matches,
    evaluate_citation,
    evaluate_retrieval,
)

__all__ = [
    "CaseResult",
    "EvalCase",
    "aggregate",
    "chunk_matches",
    "evaluate_citation",
    "evaluate_retrieval",
    "load_cases",
]
