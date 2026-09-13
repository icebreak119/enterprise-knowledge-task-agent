from pydantic import BaseModel, Field


class Citation(BaseModel):
    """回答引用到的原文切片。"""

    chunk_id: int
    source: str
    title: str
    section_path: str
    version: str | None = None


class ChatRequest(BaseModel):
    user_id: int | None = None
    conversation_id: int | None = None
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    conversation_id: int | None = None
    reply: str
    intent: str | None = None
    need_human: bool = False
    model: str | None = None
    # 知识类回答引用的原文；非知识类为空
    citations: list[Citation] = []
    # True 表示库里没找到足够相近的资料，回答为后端判定的标准拒答话术
    refused: bool = False
