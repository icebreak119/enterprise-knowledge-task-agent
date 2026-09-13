"""文档切分：把 LoadedDocument 切成带结构信息的 Chunk。

默认按**字符**滑窗（不是 token）——避免为了计数引入 tiktoken 依赖。
后续做切片实验时，可以把 `split_units` 换成真正的 tokenizer 再对比效果，
那时才有对照意义；现在先固定一个可复现的基线。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.rag.loader import LoadedDocument

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


@dataclass(frozen=True)
class _Block:
    """段内的一个自然块：连续的非空行，表格整体算一块。"""

    text: str
    start: int
    is_table: bool


@dataclass(frozen=True)
class _Section:
    path: str
    blocks: list[_Block]


@dataclass(frozen=True)
class Chunk:
    """一个切片。字段刻意做成强类型而不是裸 dict，便于检索后拼引用。

    Attributes:
        content: 切片正文。
        source: 来源文件路径（与 LoadedDocument.source 一致）。
        title: 所属文档标题。
        section_path: 标题路径，用 " > " 连接，如 "差旅管理 > 报销标准"。
            作用有两个：给用户看引用来源，给检索当元数据过滤条件。
        chunk_index: 在本文档内的序号，从 0 开始，用于按原文顺序重排检索结果。
        char_start: 在 LoadedDocument.text 中的起始下标（含）。
        char_end: 结束下标（不含），便于回原文定位与高亮。
    """

    content: str
    source: str
    title: str
    section_path: str
    chunk_index: int
    char_start: int
    char_end: int

    def to_metadata(self) -> dict[str, object]:
        """转成写入 document_chunks.metadata 的字典。"""
        data = asdict(self)
        data.pop("content")
        return data


def chunk_document(
    doc: LoadedDocument,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """把文档切成 Chunk 列表。

    契约（实现时按此验收）：
    1. **先按标题分段，再在段内滑窗**——绝不跨标题拼接，
       否则"报销标准"和"处罚条例"会混进同一个 chunk，检索到就是噪音。
    2. Markdown 的 `#`~`######` 视作标题，进入 `section_path`；标题行本身也要保留在
       chunk 正文里（模型需要它才知道这段讲什么）。
    3. 段内滑窗步长 = `chunk_size - overlap`，最后一片不足 `chunk_size` 也保留。
    4. `char_start` / `char_end` 必须能对应回 `doc.text`，用于原文定位。
    5. 空白段（strip 后为空）直接丢弃，不产出 chunk。
    6. 单个标题下的内容若短于 `chunk_size`，就只产出一个 chunk，不要为了凑数重复切片。

    Args:
        doc: 解析后的文档。
        chunk_size: 每片最大字符数。
        overlap: 相邻片的重叠字符数，用于缓解跨窗截断。

    Raises:
        ValueError: `overlap >= chunk_size`（会导致死循环）或参数为负。
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正数，当前 {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap 不能为负，当前 {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap({overlap}) 必须小于 chunk_size({chunk_size})，否则滑窗会原地打转"
        )

    chunks: list[Chunk] = []
    for section in _parse_sections(doc.text):
        # 把标题路径拼进正文：模型只有看见"这是哪一节"，才知道这段在讲什么。
        # 没有这一行，"不保修"这类短 chunk 检索到也毫无用处。
        path = section.path or doc.title
        header = f"【{path}】\n"

        for block in section.blocks:
            for piece, rel_start, rel_end in _split_block(
                block.text, chunk_size, overlap, block.is_table
            ):
                chunks.append(
                    Chunk(
                        content=header + piece,
                        source=doc.source,
                        title=doc.title,
                        section_path=path,
                        chunk_index=len(chunks),
                        char_start=block.start + rel_start,
                        char_end=block.start + rel_end,
                    )
                )
    return chunks


def _parse_sections(text: str) -> list[_Section]:
    """按标题层级把全文切成若干节，节内按空行聚成块。"""
    sections: list[_Section] = []
    stack: list[str] = []
    blocks: list[_Block] = []
    current: list[str] = []
    block_start = 0
    pos = 0

    def flush_block() -> None:
        nonlocal current
        if current:
            body = "".join(current).rstrip()
            current = []
            if body.strip():
                blocks.append(_Block(body, block_start, _is_table(body)))

    for line in text.splitlines(keepends=True):
        heading = _HEADING_RE.match(line)
        if heading:
            flush_block()
            if blocks:
                sections.append(_Section(" > ".join(stack), blocks))
                blocks = []
            level = len(heading.group(1))
            stack = stack[: level - 1] + [heading.group(2)]
        elif not line.strip():
            flush_block()
            block_start = pos + len(line)
        else:
            if not current:
                block_start = pos
            current.append(line)
        pos += len(line)

    flush_block()
    if blocks:
        sections.append(_Section(" > ".join(stack), blocks))
    return sections


def _is_table(block: str) -> bool:
    lines = block.splitlines()
    return len(lines) >= 2 and lines[0].lstrip().startswith("|")


def _split_block(
    text: str, chunk_size: int, overlap: int, is_table: bool
) -> list[tuple[str, int, int]]:
    """把块切成 (文本, 块内起始, 块内结束)。表格按行切且每片带表头。"""
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]
    if is_table:
        return _split_table(text, chunk_size)

    step = chunk_size - overlap
    pieces: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        pieces.append((text[start:end], start, end))
        if end >= len(text):
            break
        start += step
    return pieces


def _split_table(text: str, chunk_size: int) -> list[tuple[str, int, int]]:
    """表格超长时按行切；每片都带上表头，否则后半部分的列含义全丢。"""
    lines = text.splitlines(keepends=True)
    if len(lines) < 3:
        return [(text, 0, len(text))]

    header = lines[0] + lines[1]
    pos = len(header)
    pieces: list[tuple[str, int, int]] = []
    current = ""
    current_start = pos

    for line in lines[2:]:
        if current and len(header) + len(current) + len(line) > chunk_size:
            pieces.append((header + current, current_start, current_start + len(current)))
            current = ""
            current_start = pos
        current += line
        pos += len(line)

    if current:
        pieces.append((header + current, current_start, current_start + len(current)))
    return pieces
