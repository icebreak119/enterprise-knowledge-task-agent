"""工具注册与工厂。

所有工具集中注册在此，路由层只依赖本模块的 `get_openai_tools()` 与 `run_tool()`。
避免路由层直接 import 各个工具模块。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.tools.base import ToolDef
from app.tools.business import (
    QUERY_CUSTOMER_PARAMS,
    QUERY_ORDER_PARAMS,
    query_customer,
    query_order,
)
from app.tools.retrieval_tool import search_knowledge, SEARCH_KNOWLEDGE_PARAMS

# 注册表：name → ToolDef（session 在调用时注入，不在定义时绑定）
_TOOLS: dict[str, ToolDef] = {}


def register_tool(tool: ToolDef) -> None:
    """注册一个工具。重名会直接覆盖，便于测试替换。"""
    _TOOLS[tool.name] = tool


# ── 注册已知工具 ────────────────────────────────────────

register_tool(ToolDef(
    name="search_knowledge",
    description="从售后制度文档中检索相关信息，适用于保修、退换货、服务响应标准等知识类问题",
    parameters=SEARCH_KNOWLEDGE_PARAMS,
    execute=search_knowledge,
))
register_tool(ToolDef(
    name="query_order",
    description="按订单号、产品名称或订单状态查询销售订单信息",
    parameters=QUERY_ORDER_PARAMS,
    execute=query_order,
))
register_tool(ToolDef(
    name="query_customer",
    description="按客户姓名查询客户基本信息",
    parameters=QUERY_CUSTOMER_PARAMS,
    execute=query_customer,
))


def get_openai_tools() -> list[dict]:
    """返回当前全部工具定义的 OpenAI Function Calling 格式，供 LLM 调用。"""
    return [tool.openai_tool() for tool in _TOOLS.values()]


async def run_tool(name: str, arguments: str, session: AsyncSession) -> str:
    """执行指定工具。

    Args:
        name: 工具名，必须存在于注册表中。
        arguments: JSON 字符串参数。
        session: 数据库会话，透传给工具函数。

    Returns:
        工具执行结果的文本描述。

    Raises:
        KeyError: 未注册的工具名。
    """
    import json

    tool = _TOOLS.get(name)
    if tool is None:
        raise KeyError(f"未注册的工具：{name}，可用工具有：{list(_TOOLS)}")
    return await tool.execute(json.loads(arguments), session)