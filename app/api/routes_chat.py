from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest, session: AsyncSession = Depends(get_session)
) -> ChatResponse:
    # Day 1 占位实现：仅回显，Day 2 接入 LLM Provider。
    return ChatResponse(conversation_id=req.conversation_id, reply=f"[stub] 收到：{req.message}")