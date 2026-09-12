"""结构化输出模型：意图识别结果。

Day 5 的 Router 会直接消费这个结构，所以字段设计一次到位。
"""

from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    KNOWLEDGE_QA = "knowledge_qa"  # 问制度 / 产品文档，需要检索
    BUSINESS_QUERY = "business_query"  # 查客户 / 订单，需要工具
    TASK_EXECUTION = "task_execution"  # 写操作，必须人工确认
    HUMAN_HANDOFF = "human_handoff"  # 直接转人工
    CHITCHAT = "chitchat"  # 闲聊，直接回答


class IntentResult(BaseModel):
    intent: Intent
    need_retrieval: bool = False
    need_tools: bool = False
    slots: dict[str, object] = Field(default_factory=dict)
    confidence: float = 0.0
    rationale: str = ""
