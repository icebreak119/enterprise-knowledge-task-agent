from datetime import datetime

from sqlalchemy import String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    source: Mapped[str | None] = mapped_column(String(255))
    version: Mapped[str | None] = mapped_column(String(32))
    # active / deprecated。必须显式标记，不能靠"版本号最大即现行"来推断：
    # 只召回到旧版本时那种推断会直接失效，而这正是要防的事故。
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())