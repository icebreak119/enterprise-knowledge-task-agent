"""文档解析：把单份文件读成纯文本 + 来源元信息。

第一期只支持 .md / .txt，刻意绕开 pdf/docx——它们的解析坑（扫描件、表格、页眉页脚）
会拖慢主链路，等检索闭环跑通、评测集建好之后再补。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".markdown"}


@dataclass(frozen=True)
class LoadedDocument:
    """一份文档的解析结果。

    Attributes:
        source: 原始路径（相对路径即可），用于引用展示与入库去重。
        title: 文档标题。Markdown 取首个 H1；没有则退化成文件名（去扩展名）。
        text: 纯文本正文。Markdown 保留 `#` 标记，供 chunker 识别标题层级。
        metadata: 其余元信息，如后缀名、字节数、文件 mtime。
    """

    source: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def load_document(path: str | Path) -> LoadedDocument:
    """把单个文件解析成 LoadedDocument。

    契约：
    - 支持 `SUPPORTED_SUFFIXES` 内的后缀；其余后缀抛 `ValueError`，不要静默跳过，
      否则"某类文档没进来"会变成一个很难查的静默失败。
    - 统一按 UTF-8 读取；编码错误抛 `UnicodeDecodeError` 而不是 `errors="ignore"`
      （忽略会带来肉眼看不见的乱码 chunk）。
    - `title`：Markdown 取首个 `# ` 行，没有则用文件名。
    - `text` 不做任何清洗（不去空行、不折叠空白），清洗交给 chunker，
      保证 char_start/char_end 能对应回原文。

    Raises:
        FileNotFoundError: 路径不存在。
        ValueError: 后缀不支持。
    """
    raise NotImplementedError
