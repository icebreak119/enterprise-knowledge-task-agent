import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import LLMDep
from app.core.config import get_settings
from app.db.session import get_session
from app.llm.base import ChatMessage, ToolCall
from app.llm.errors import LLMError
from app.prompts import ANSWER_PROMPT, INTENT_PROMPT, SYSTEM_PROMPT, TOOL_PROMPT
from app.rag.retriever import (
    RetrievedChunk,
    build_context,
    cited_chunk_ids,
    search,
    should_refuse,
)
from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.schemas.intent import Intent, IntentResult
from app.tools import get_openai_tools, run_tool

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "当前资料中没有找到依据。建议补充相关制度文档，或转人工咨询。"
# 模型给了空 content 时的兜底，避免用户收到空回复
EMPTY_ANSWER_FALLBACK = "已查询到相关数据，但未能生成完整回答。请重试，或转人工咨询。"
# 工具调用循环上限：防止模型无限调用（如搜索 → 再搜索 → 再搜索）
_MAX_TOOL_ROUNDS = 5


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    session: AsyncSession = Depends(get_session),
    llm: LLMDep = None,
) -> ChatResponse:
    settings = get_settings()

    # ── 1) 意图识别 ──────────────────────────────────────
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

    intent = intent_result.intent.value if intent_result else None
    need_tools = bool(intent_result and intent_result.need_tools)
    need_retrieval = bool(intent_result and intent_result.need_retrieval)

    # ── 2a) 分支：知识检索（沿用原有逻辑） ──────────────
    if need_retrieval and not need_tools:
        return await _handle_knowledge_qa(req, session, llm, intent_result, settings)

    # ── 2b) 分支：工具调用（BUSINESS_QUERY / TASK_EXECUTION） ──
    if need_tools:
        answer_content, tools_used = await _handle_tool_call(req, session, llm)
    else:
        # 闲聊 / 意图识别失败兜底
        answer_content = await _fallback_chat(req, llm)
        tools_used = []

    need_human = bool(
        intent_result
        and intent_result.intent in (Intent.HUMAN_HANDOFF, Intent.TASK_EXECUTION)
    )

    logger.info(
        "chat intent=%s user_id=%s tools=%s need_human=%s",
        intent,
        req.user_id,
        tools_used or "无",
        need_human,
    )

    return ChatResponse(
        conversation_id=req.conversation_id,
        reply=answer_content,
        intent=intent,
        need_human=need_human,
        # OpenAICompatProvider 把模型名放在配置里，FakeProvider 直接暴露 .model
        model=getattr(llm, "model", None) or settings.llm_model,
    )


# ── 子流程：知识检索（与之前逻辑一致） ──────────────────────


async def _handle_knowledge_qa(
    req: ChatRequest,
    session: AsyncSession,
    llm: LLMDep,
    intent_result: IntentResult,
    settings,
) -> ChatResponse:
    retrieved: list[RetrievedChunk] = []
    if not settings.embedding_enabled:
        logger.warning("未配置 embedding，知识类问题无法检索，将直接拒答")
    else:
        retrieved = await search(session, req.message)

    if should_refuse(retrieved, settings.rag_max_distance):
        return ChatResponse(
            conversation_id=req.conversation_id,
            reply=REFUSAL_MESSAGE,
            intent=intent_result.intent.value,
            need_human=False,
            model=None,
            refused=True,
        )

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

    return ChatResponse(
        conversation_id=req.conversation_id,
        reply=answer.content,
        intent=intent_result.intent.value,
        need_human=False,
        model=answer.model,
        citations=citations,
    )


# ── 子流程：工具调用循环 ──────────────────────────────────


async def _handle_tool_call(
    req: ChatRequest,
    session: AsyncSession,
    llm: LLMDep,
) -> tuple[str, list[str]]:
    """多轮工具调用循环：调用 LLM → 解析 tool_calls → 执行 → 再调用。

    返回 (最终回答, 本轮实际调用过的工具名列表)。
    工具名要记下来：出问题时能直接看出是模型没调工具，还是工具本身返回错。
    """
    messages = [
        ChatMessage(role="system", content=TOOL_PROMPT),
        ChatMessage(role="user", content=req.message),
    ]
    tools = get_openai_tools()
    tools_used: list[str] = []

    for _round in range(_MAX_TOOL_ROUNDS):
        try:
            response = await llm.achat(messages, tools=tools, temperature=0)
        except LLMError as exc:
            logger.exception("工具调用回合失败")
            return f"工具调用服务异常：{exc}", tools_used

        if not response.tool_calls:
            # 模型决定直接输出文字，不再调用工具
            return _non_empty(response.content), tools_used

        # 把 assistant 的 tool_calls 放回对话历史
        messages.append(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=response.tool_calls,
            )
        )

        for tc in response.tool_calls:
            tools_used.append(tc.name)
            result = await _safe_execute_tool(tc, session)
            messages.append(
                ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tc.id,
                )
            )

    # 超过轮次上限：把已收集到的工具结果交回模型做最终归纳
    messages.append(
        ChatMessage(
            role="user",
            content="已达到工具调用次数上限，请基于以上工具返回结果直接给出最终回答。",
        )
    )
    try:
        final = await llm.achat(messages, temperature=0)
        return _non_empty(final.content), tools_used
    except LLMError as exc:
        logger.warning("工具循环收尾失败：%s", exc)
        return EMPTY_ANSWER_FALLBACK, tools_used


def _non_empty(content: str | None) -> str:
    """模型可能只返回 tool_calls 而 content 为空，不能让用户收到空字符串。"""
    text = (content or "").strip()
    return text or EMPTY_ANSWER_FALLBACK


async def _safe_execute_tool(tc: ToolCall, session: AsyncSession) -> str:
    """执行单个工具，捕获所有异常，始终返回文本结果。"""
    try:
        return await run_tool(tc.name, tc.arguments, session)
    except KeyError:
        logger.error("未知工具调用：%s", tc.name)
        return f"错误：工具「{tc.name}」不存在，无法执行。"
    except Exception as exc:  # noqa: BLE001
        logger.exception("工具 %s 执行失败", tc.name)
        return f"工具「{tc.name}」执行出错：{exc}"


# ── 子流程：闲聊兜底 ──────────────────────────────────


async def _fallback_chat(req: ChatRequest, llm: LLMDep) -> str:
    """意图识别失败或闲聊时直接调用模型。"""
    try:
        resp = await llm.achat(
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=req.message),
            ]
        )
        return resp.content
    except LLMError as exc:
        logger.warning("闲聊生成失败：%s", exc)
        return f"[无法生成回答] {exc}"