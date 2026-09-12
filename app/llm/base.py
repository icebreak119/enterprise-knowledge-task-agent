"""LLM Provider 抽象层。

业务代码只依赖本模块的接口，不直接 import 任何厂商 SDK，
这样换模型 / 换厂商只需要改配置或新增一个实现类。
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

Role = str  # "system" | "user" | "assistant" | "tool"


@dataclass
class ChatMessage:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass
class ToolCall:
    """Day 3 起使用：模型要求调用的工具。"""

    id: str
    name: str
    arguments: str  # JSON 字符串，由工具自身的 args_model 校验


@dataclass
class LLMResponse:
    content: str
    parsed: BaseModel | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    model: str | None = None
    reasoning: str | None = None  # 推理模型的思维链，仅用于排查问题


@runtime_checkable
class LLMProvider(Protocol):
    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """发起一次对话。

        response_model 非空时，返回值中的 parsed 会是对应 Pydantic 模型的实例。
        """
        ...
