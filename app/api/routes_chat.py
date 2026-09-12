import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import LLMDep
from app.db.session import get_session
from app.llm.base import ChatMessage
from app.llm.errors import LLMError
from app.prompts import ANSWER_PROMPT, INTENT_PROMPT, SYSTEM_PROMPT
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.intent import Intent, IntentResult

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    session: AsyncSession = Depends(get_session),  # Day 5 起用于会话与消息落库
    llm: LLMDep = None,
) -> ChatResponse:
    # 1) 意图识别：结构化输出，失败时降级为闲聊，不阻断对话
    intent_result: IntentResult | None = None
    try:
        intent_resp = await llm.achat(
            [
                ChatMessage(role="system", content=INTENT_PROMPT),
                ChatMessage(role="user", content=req.message),
            ],
            response_model=IntentResult,
            temperature=0.0,
        )
        intent_result = intent_resp.parsed
    except LLMError as exc:
        logger.warning("意图识别失败，按闲聊继续：%s", exc)

    # 2) 生成回答（Day 4 起在此之前插入知识检索结果）
    try:
        answer = await llm.achat(
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=f"{ANSWER_PROMPT}\n\n用户问题：{req.message}",
                ),
            ]
        )
    except LLMError as exc:
        logger.exception("生成回答失败")
        raise HTTPException(status_code=502, detail=f"模型服务不可用：{exc}") from exc

    intent = intent_result.intent.value if intent_result else None
    need_human = bool(
        intent_result
        and intent_result.intent in (Intent.HUMAN_HANDOFF, Intent.TASK_EXECUTION)
    )
    logger.info("chat intent=%s user_id=%s usage=%s", intent, req.user_id, answer.usage)

    return ChatResponse(
        conversation_id=req.conversation_id,
        reply=answer.content,
        intent=intent,
        need_human=need_human,
        model=answer.model,
    )
