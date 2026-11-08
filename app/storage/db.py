"""SQLite 存储：SQLAlchemy 2.0 async ORM。

对应 docs/technical-design.md §6.1–§6.2：
- papers / cards / sessions 三张业务表，列名/类型/约束/索引与 §6.2.1–§6.2.3 一致
- checkpoints 表由 langgraph-checkpoint-sqlite 自动创建，不写 ORM 模型
- 每次新连接执行 PRAGMA journal_mode=WAL / foreign_keys=ON / busy_timeout=5000
- 时间统一存 naive UTC（SQLite 无时区类型；读取时按 UTC 解释）
"""
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, Text, event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """当前 UTC 时间的 naive 表示（SQLite 无时区类型，naive 保证 roundtrip）。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    paper_id: Mapped[str] = mapped_column(Text, primary_key=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    method_found: Mapped[bool | None] = mapped_column(Boolean)
    error_msg: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[int | None] = mapped_column(Integer)
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    meta_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    # passive_deletes=True：删除由 DB 层 ON DELETE CASCADE 处理，ORM 不置空/不逐条删子行
    cards: Mapped[list["Card"]] = relationship(back_populates="paper", passive_deletes=True)


class Card(Base):
    __tablename__ = "cards"

    card_id: Mapped[str] = mapped_column(Text, primary_key=True)
    paper_id: Mapped[str] = mapped_column(
        Text, ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False, index=True
    )
    system_formula: Mapped[str] = mapped_column(Text, nullable=False)
    system_normalized: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    calculation_type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    functional: Mapped[str] = mapped_column(Text, nullable=False)
    functional_family: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    parameters_json: Mapped[str] = mapped_column(Text, nullable=False)
    surface_json: Mapped[str | None] = mapped_column(Text)
    missing_fields: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    validation_json: Mapped[str] = mapped_column(Text, nullable=False)
    provenance_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    review_json: Mapped[str | None] = mapped_column(Text)
    review_status: Mapped[str] = mapped_column(Text, nullable=False)
    chroma_id: Mapped[str] = mapped_column(Text, nullable=False)
    chroma_indexed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    schema_version: Mapped[str] = mapped_column(Text, nullable=False)
    rules_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, onupdate=utcnow
    )

    # 纯 ORM 关系（不改表结构）：保证同事务 flush 时 papers 先于 cards 插入
    paper: Mapped["Paper"] = relationship(back_populates="cards")

    __table_args__ = (
        Index("ix_cards_system_functional", "system_normalized", "functional_family"),
    )


class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(Text, primary_key=True)
    active_workflow: Mapped[str | None] = mapped_column(Text)
    thread_id: Mapped[str | None] = mapped_column(Text)
    slots_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    last_action: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, onupdate=utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)


def create_engine(url: str) -> AsyncEngine:
    """创建 async engine；每次新连接执行 SQLite PRAGMA（不依赖一次性初始化）。"""
    engine = create_async_engine(url)

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine
