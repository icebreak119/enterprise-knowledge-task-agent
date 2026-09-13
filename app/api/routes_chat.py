import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import LLMDep
from app.core.config import get_settings
from app.db.session import get_session
from app.llm.base import ChatMessage
from app.llm.errors import LLMError
from app.prompts import ANSWER_PROMPT, INTENT_PROMPT, SYSTEM_PROMPT
from app.rag.retriever import (
    RetrievedChunk,
    build_context,
    cited_chunk_ids,
    search,
    should_refuse,
)
from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.schemas.intent import Intent, IntentResult

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)

# 拒答话术固定，不交给模型生成：同一输入永远同一输出，便于审计与统计
REFUSAL_MESSAGE = "当前资料中没有找到依据。建议补充相关制度文档，或转人工咨询。"


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    session: AsyncSession = Depends(get_session),  # Day 5 起用于会话与消息落库
    llm: LLMDep = None,
) -> ChatResponse:
    settings = get_settings()

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

    # 2) 知识检索：只在明确需要时检索，避免每次请求都白跑一次 embedding
    need_retrieval = bool(intent_result and intent_result.need_retrieval)
    retrieved: list[RetrievedChunk] = []
    if need_retrieval:
        if not settings.embedding_enabled:
            logger.warning("未配置 embedding，知识类问题无法检索，将直接拒答")
        else:
            retrieved = await search(session, req.message)

    # 3) 拒答由后端判定：库里没有足够相近的资料就是没有，不交给模型"觉得"
    if need_retrieval and should_refuse(retrieved, settings.rag_max_distance):
        logger.info(
            "chat refused user_id=%s best_distance=%s",
            req.user_id,
            retrieved[0].distance if retrieved else None,
        )
        return ChatResponse(
            conversation_id=req.conversation_id,
            reply=REFUSAL_MESSAGE,
            intent=intent_result.intent.value,
            need_human=False,
            model=None,
            refused=True,
        )

    # 4) 生成回答
    context = build_context(retrieved)
    materials = f"可用资料：\n{context}" if context else "（本次没有可用资料）"
    try:
        answer = await llm.achat(
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=f"{ANSWER_PROMPT}\n\n{materials}\n\n用户问题：{req.message}",
                ),
            ]
        )
    except LLMError as exc:
        logger.exception("生成回答失败")
        raise HTTPException(status_code=502, detail=f"模型服务不可用：{exc}") from exc

    # 5) 回贴引用：只认回答里真正标注过的编号，没标就是不引用
    cited = cited_chunk_ids(answer.content)
    citations = [
        Citation(
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            title=chunk.title,
            section_path=chunk.section_path,
            version=chunk.version,
        )
        for chunk in retrieved
        if chunk.chunk_id in cited
    ]

    intent = intent_result.intent.value if intent_result else None
    need_human = bool(
        intent_result
        and intent_result.intent in (Intent.HUMAN_HANDOFF, Intent.TASK_EXECUTION)
    )
    logger.info(
        "chat intent=%s user_id=%s retrieved=%s cited=%s usage=%s",
        intent,
        req.user_id,
        len(retrieved),
        len(citations),
        answer.usage,
    )

    return ChatResponse(
        conversation_id=req.conversation_id,
        reply=answer.content,
        intent=intent,
        need_human=need_human,
        model=answer.model,
        citations=citations,
    )
