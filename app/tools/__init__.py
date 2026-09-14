"""工具（Function Calling）注册与调用。

架构：工具集中注册在 registry 中，路由层不直接 import 各个工具模块。
"""

from app.tools.base import ToolDef
from app.tools.registry import get_openai_tools, run_tool

__all__ = [
    "ToolDef",
    "get_openai_tools",
    "run_tool",
]