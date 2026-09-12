from pydantic import BaseModel, Field


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
