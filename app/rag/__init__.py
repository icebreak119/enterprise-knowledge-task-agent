"""RAG 入库与检索（Day 4 拆分到 Day 3 之前先做）。

分四步：load（解析） → chunk（切分） → embed（向量化） → retrieve（检索）。
本包只负责前三步加检索，不负责生成；生成仍走 app/llm。
"""

from app.rag.chunker import Chunk, chunk_document
from app.rag.embeddings import Embedder, OpenAICompatEmbedder
from app.rag.ingest import IngestResult, ingest_file
from app.rag.loader import LoadedDocument, load_document

__all__ = [
    "Chunk",
    "Embedder",
    "IngestResult",
    "LoadedDocument",
    "OpenAICompatEmbedder",
    "chunk_document",
    "ingest_file",
    "load_document",
]
