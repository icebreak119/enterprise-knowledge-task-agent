"""工具描述与执行接口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolDef:
    """一个可被 LLM 调用的工具。

    每个工具包含：
    - name / description：LLM 理解它做什么用。
    - parameters：JSON Schema，LLM 据此判断传入什么参数。
    - execute：实际业务逻辑，参数已由外层校验好。
    - needs_human_approval：写操作等涉及用户授权的步骤，标记后路由层先请示用户再执行。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Any  # async (params: dict[str, Any]) -> str
    needs_human_approval: bool = False

    def openai_tool(self) -> dict[str, Any]:
        """返回 OpenAI Function Calling 格式的 tool 定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }