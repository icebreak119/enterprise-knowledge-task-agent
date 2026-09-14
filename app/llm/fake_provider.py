"""离线 Provider：没有 API key 或跑测试时使用，不发起任何网络请求。"""

import json
import re

from pydantic import BaseModel

from app.llm.base import ChatMessage, LLMResponse, ToolCall
from app.schemas.intent import Intent, IntentResult

# 关键词 → 意图，覆盖常见的企业场景，方便离线联调
# 顺序即优先级：转人工 > 写操作 > 知识问答 > 业务查询
_RULES: list[tuple[str, Intent, bool]] = [
    (r"转人工|人工客服|投诉", Intent.HUMAN_HANDOFF, False),
    (r"帮我(修改|取消|删除|创建|下单|更新)", Intent.TASK_EXECUTION, True),
    (r"报销|差旅|制度|规定|流程|政策|文档|手册|保修|退换货|退款|响应", Intent.KNOWLEDGE_QA, True),
    (r"订单|客户|金额|买了|付款|退款|合同", Intent.BUSINESS_QUERY, True),
]

# 业务查询里常见的中文姓名，用于离线模式下构造工具参数
_NAME_RE = re.compile(r"([\u4e00-\u9fa5]{2,3})(?:的)?(?:信息|资料|情况|订单|联系方式)")


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


def _fake_tool_call(text: str, tools: list[dict]) -> ToolCall | None:
    """按关键词挑一个工具，用于离线验证工具调用链路。

    只服务于测试与无 Key 环境，不追求真实语义理解。
    """
    available = {tool["function"]["name"] for tool in tools}

    if "query_order" in available:
        matched = re.search(r"A\d{4}", text)
        if matched:
            args = {"order_no": matched.group(0)}
        elif re.search(r"退款|refunded", text):
            args = {"status": "refunded"}
        else:
            args = {"status": "paid"}
        return ToolCall(id="call_order_1", name="query_order", arguments=json.dumps(args))

    if "query_customer" in available:
        matched = _NAME_RE.search(text)
        name = matched.group(1) if matched else "张三"
        return ToolCall(
            id="call_customer_1",
            name="query_customer",
            arguments=json.dumps({"name": name}, ensure_ascii=False),
        )

    return None


class FakeProvider:
    """按关键词返回固定意图，保证本地起服务和跑单测都不依赖真实凭据。

    带 tools 时模拟一轮工具调用：第一次返回 tool_call，
    收到工具结果后返回最终文本——这样工具调用循环在离线状态下也能被测到。
    """

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

        if tools:
            has_tool_result = any(m.role == "tool" for m in messages)
            if not has_tool_result:
                call = _fake_tool_call(user_text, tools)
                if call is not None:
                    return LLMResponse(
                        content="", tool_calls=[call], model=self.model
                    )
            tool_outputs = [m.content for m in messages if m.role == "tool"]
            joined = "\n".join(tool_outputs) if tool_outputs else "（无工具结果）"
            return LLMResponse(
                content=f"[离线模式] 工具返回：\n{joined}",
                model=self.model,
            )

        return LLMResponse(
            content=f"[离线模式] 未配置 LLM_API_KEY，无法生成回答。收到：{user_text}",
            parsed=None,
            model=self.model,
        )
