"""种子数据：插入演示用户/客户/订单，便于后续 Tool Calling 阶段联调。"""

import asyncio
from decimal import Decimal

from sqlalchemy import func, select

from app.db.base import Base
from app.db.session import async_session_factory
from app.models.customer import Customer
from app.models.order import Order
from app.models.user import User
import app.models  # noqa: F401


async def seed() -> None:
    async with async_session_factory() as session:
        # 幂等：已存在数据则跳过
        count = await session.scalar(select(func.count(User.id)))
        if count:
            print("已存在数据，跳过 seed。")
            return

        user = User(
            username="zhangsan",
            email="zhangsan@corp.com",
            department="销售部",
            role="员工",
        )
        session.add(user)
        await session.flush()

        customer = Customer(name="张三", phone="13800000000", level="普通", status="active")
        session.add(customer)
        await session.flush()

        session.add_all(
            [
                Order(customer_id=customer.id, order_no="A1001", product_name="企业版License", amount=Decimal("19999.00"), status="paid"),
                Order(customer_id=customer.id, order_no="A1002", product_name="私有化部署", amount=Decimal("88000.00"), status="paid"),
                Order(customer_id=customer.id, order_no="A1003", product_name="年度维保", amount=Decimal("15999.00"), status="refunded"),
            ]
        )
        await session.commit()
        print("seed 完成：张三 + 3 条订单。")


if __name__ == "__main__":
    asyncio.run(seed())