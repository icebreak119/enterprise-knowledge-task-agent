"""跑评测集。

两种模式：

    python scripts/run_eval.py                    # 只评检索，快，不需要起服务
    python scripts/run_eval.py --mode full        # 走真实 /api/chat，连引用一起评

full 模式需要服务已在 http://127.0.0.1:8000 运行（可用 --base-url 改）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from app.core.config import get_settings
from app.db.session import async_session_factory
from app.eval import aggregate, evaluate_citation, evaluate_retrieval, load_cases
from app.eval.dataset import EvalCase
from app.eval.metrics import CaseResult  # noqa: F401  用于类型标注
from app.rag.retriever import RetrievedChunk, search, should_refuse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evals" / "cases.jsonl"


async def _run_retrieval(
    cases: list[EvalCase], top_k: int, penalty: float | None = None
) -> list[CaseResult]:
    settings = get_settings()
    if penalty is not None:
        settings = settings.model_copy(
            update={"rag_deprecated_penalty": penalty}
        )
    results: list[CaseResult] = []
    async with async_session_factory() as session:
        for case in cases:
            retrieved = await search(session, case.question, top_k=top_k, settings=settings)
            refused = should_refuse(retrieved, settings.rag_max_distance)
            results.append(evaluate_retrieval(case, retrieved, refused))
    return results


async def _run_full(
    cases: list[EvalCase], base_url: str, top_k: int
) -> list[CaseResult]:
    """走真实接口：拿到 refused 与 citations，再补检索结果用于计算 recall。"""
    import httpx

    settings = get_settings()
    results: list[CaseResult] = []

    async with async_session_factory() as session, httpx.AsyncClient(
        timeout=120
    ) as http:
        for case in cases:
            response = await http.post(
                f"{base_url}/api/chat", json={"user_id": 1, "message": case.question}
            )
            response.raise_for_status()
            body = response.json()

            retrieved = await search(session, case.question, top_k=top_k)
            result = evaluate_retrieval(case, retrieved, body.get("refused", False))
            result = replace(
                result,
                cited_ids={c["chunk_id"] for c in body.get("citations", [])},
            )
            results.append(evaluate_citation(result))
    return results


def _report(results: list[CaseResult], top_k: int, mode: str) -> None:
    summary = aggregate(results)
    print(f"\n=== 评测结果（mode={mode}, top_k={top_k}）===")
    for key, value in summary.items():
        print(f"  {key:18s} {value}")

    failures = [r for r in results if r.is_failure]
    if failures:
        print(f"\n--- 失败样例 {len(failures)} 条 ---")
        for result in failures:
            case = result.case
            top = result.retrieved[0] if result.retrieved else None
            reasons = []
            if not result.refuse_ok:
                reasons.append("拒答判定错误")
            if result.recall < 1.0:
                reasons.append(f"证据召回 {result.recall:.0%}")
            if result.version_ok is False:
                reasons.append(f"引到 {top.version if top else '?'}")
            print(f"  [{case.category}] {case.id}: {' / '.join(reasons)}")
            if top:
                print(f"      top1 -> {top.version} | {top.section_path[-40:]}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="覆盖 rag_deprecated_penalty，用于扫参看敏感度",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    top_k = args.top_k or get_settings().rag_top_k
    print(f"载入 {len(cases)} 道题")

    if args.mode == "retrieval":
        results = await _run_retrieval(cases, top_k, args.penalty)
    else:
        results = await _run_full(cases, args.base_url, top_k)

    _report(results, top_k, args.mode)

    if args.json_out:
        payload = {
            "mode": args.mode,
            "top_k": top_k,
            "summary": aggregate(results),
            "cases": [
                {
                    "id": r.case.id,
                    "category": r.case.category,
                    "recall": r.recall,
                    "version_ok": r.version_ok,
                    "refuse_ok": r.refuse_ok,
                    "citation_ok": r.citation_ok,
                    "top1_version": r.retrieved[0].version if r.retrieved else None,
                    "top1_distance": r.retrieved[0].distance if r.retrieved else None,
                }
                for r in results
            ],
        }
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n明细已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
