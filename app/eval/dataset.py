"""评测集数据结构。

用"定位串"而不是数据库主键来标注证据：重新入库后 chunk_id 会变，
而"来源文件名 + 章节路径"是稳定的，评测集才不会因为重建索引而失效。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """一道评测题。

    Attributes:
        id: 唯一标识，便于定位失败样例。
        question: 用户问题原文。
        category: 题型。direct / version_trap / should_refuse / cross_section。
        evidence: 证据列表，每项是一组定位关键词。**每组内的关键词必须同时出现在**
            `chunk.source + chunk.section_path` 中，才算命中该条证据；
            组与组之间是"都要召回"的关系。
        expected_version: 正确版本（版本陷阱题必填），用于统计版本错误率。
        should_refuse: 该题是否应当拒答（库里确实没有相关资料）。
        answer_keywords: 回答中应当出现的关键词，用于判断答案是否正确。
        notes: 标注理由，方便日后回看为什么这么标。
    """

    id: str
    question: str
    category: str = "direct"
    evidence: list[list[str]] = Field(default_factory=list)
    expected_version: str | None = None
    should_refuse: bool = False
    answer_keywords: list[str] = Field(default_factory=list)
    notes: str = ""


def load_cases(path: str | Path) -> list[EvalCase]:
    """从 JSONL 读取评测集，每行一题。"""
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            cases.append(EvalCase.model_validate_json(line))
    return cases
