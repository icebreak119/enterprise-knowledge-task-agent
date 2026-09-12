"""离线 Provider：没有 API key 或跑测试时使用，不发起任何网络请求。"""

import re

from pydantic import BaseModel

from app.llm.base import ChatMessage, LLMResponse
from app.schemas.intent import Intent, IntentResult

# 关键词 → 意图，覆盖常见的企业场景，方便离线联调
# 顺序即优先级：转人工 > 写操作 > 知识问答 > 业务查询
_RULES: list[tuple[str, Intent, bool]] = [
    (r"转人工|人工客服|投诉", Intent.HUMAN_HANDOFF, False),
    (r"帮我(修改|取消|删除|创建|下单|更新)", Intent.TASK_EXECUTION, True),
    (r"报销|差旅|制度|规定|流程|政策|文档|手册", Intent.KNOWLEDGE_QA, True),
    (r"订单|客户|金额|买了|付款|退款|合同", Intent.BUSINESS_QUERY, True),
]


def _guess(text: str) -> IntentResult:
    for pattern, intent, need_tools in _RULES:
        if re.search(pattern, text):
            return IntentResult(
                intent=intent,
                need_retrieval=intent is Intent.KNOWLEDGE_QA,
                need_tools=need_tools and intent is not Intent.KNOWLEDGE_QA,
                confidence=0.6,
                rationale=f"命中离线规则：{pattern}",
            )
    return IntentResult(
        intent=Intent.CHITCHAT,
        confidence=0.5,
        rationale="未命中任何规则，按闲聊处理",
    )


class FakeProvider:
    """按关键词返回固定意图，保证本地起服务和跑单测都不依赖真实凭据。"""

    def __init__(self, model: str = "fake-model") -> None:
        self.model = model

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[BaseModel] | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        user_text = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )

        if response_model is IntentResult:
            return LLMResponse(content="", parsed=_guess(user_text), model=self.model)

        return LLMResponse(
            content=f"[离线模式] 未配置 LLM_API_KEY，无法生成回答。收到：{user_text}",
            parsed=None,
            model=self.model,
        )
