"""业务数据查询类工具。

调用方负责注入 AsyncSession，工具本身不做权限判断。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.customer import Customer
from app.models.order import Order

# ── JSON Schema 定义（集中放，便于 review） ──────────────────────

QUERY_ORDER_PARAMS = {
    "type": "object",
    "properties": {
        "order_no": {"type": "string", "description": "订单号，如 A1001"},
        "product_name": {"type": "string", "description": "产品名称关键字，模糊匹配"},
        "status": {"type": "string", "enum": ["paid", "refunded", "pending"]},
        "customer_name": {
            "type": "string",
            "description": "客户姓名，用于查询某位客户名下的订单；问「张三买了什么」时必须带上",
        },
    },
    "anyOf": [
        {"required": ["order_no"]},
        {"required": ["product_name"]},
        {"required": ["status"]},
        {"required": ["customer_name"]},
    ],
}

QUERY_CUSTOMER_PARAMS = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "客户姓名，模糊匹配"},
    },
    "required": ["name"],
}

# ── 工具函数（注入 session） ──────────────────────────────────


async def query_order(params: dict, session: AsyncSession) -> str:
    """查询订单：订单号精确 / 产品名模糊 / 按状态 / 按客户名，支持组合。

    按客户名筛选是必需的：否则「张三买了什么」会返回库里所有订单，
    在只有一个客户时"碰巧正确"，客户一多就是错的。
    """
    stmt = select(Order)
    if customer_name := params.get("customer_name"):
        stmt = stmt.join(Customer).where(Customer.name.ilike(f"%{customer_name}%"))
    if order_no := params.get("order_no"):
        stmt = stmt.where(Order.order_no == order_no)
    if product := params.get("product_name"):
        stmt = stmt.where(Order.product_name.ilike(f"%{product}%"))
    if status := params.get("status"):
        stmt = stmt.where(Order.status == status)
    rows = (await session.execute(stmt.limit(10))).scalars().all()
    if not rows:
        return "未找到符合条件的订单。"
    parts = [f"共找到 {len(rows)} 条订单："]
    for o in rows:
        parts.append(
            f"  [{o.order_no}] {o.product_name} | {o.amount}元 | {o.status} | {o.created_at.strftime('%Y-%m-%d')}"
        )
    return "\n".join(parts)


async def query_customer(params: dict, session: AsyncSession) -> str:
    """查询客户：按姓名模糊匹配。"""
    stmt = select(Customer).where(Customer.name.ilike(f"%{params['name']}%"))
    rows = (await session.execute(stmt.limit(5))).scalars().all()
    if not rows:
        return f"未找到名字包含「{params['name']}」的客户。"
    parts = [f"共找到 {len(rows)} 位客户："]
    for c in rows:
        parts.append(
            f"  {c.name} | 电话 {c.phone} | 等级 {c.level} | 状态 {c.status}"
        )
    return "\n".join(parts)