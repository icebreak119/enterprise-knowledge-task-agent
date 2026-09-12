"""文档切分：把 LoadedDocument 切成带结构信息的 Chunk。

默认按**字符**滑窗（不是 token）——避免为了计数引入 tiktoken 依赖。
后续做切片实验时，可以把 `split_units` 换成真正的 tokenizer 再对比效果，
那时才有对照意义；现在先固定一个可复现的基线。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.rag.loader import LoadedDocument

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64


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
    raise NotImplementedError
