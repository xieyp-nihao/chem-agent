"""初始化数据库：目录骨架 + 业务三表 + LangGraph checkpoints 表。

对应 docs/technical-design.md §6.1 / §6.2.4：
- checkpoints 表由 langgraph-checkpoint-sqlite 的 AsyncSqliteSaver.setup() 创建，不写 ORM 模型
- PRAGMA 由 db.create_engine 的 connect 事件在每次新连接时执行，无需本脚本一次性设置

用法（项目根目录）：
    python scripts/init_db.py
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 允许 python scripts/init_db.py 直跑

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy.engine import make_url

from app.config import settings
from app.storage.db import Base, create_engine
from app.storage.files import FileStore

EXPECTED_TABLES = ("papers", "cards", "sessions", "checkpoints", "writes")


async def main() -> None:
    data_root = Path(settings.DATA_DIR)
    FileStore(data_root).ensure_base_dirs()
    print(f"[dirs] base directories ready under {data_root}")

    engine = create_engine(settings.DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print("[tables] business tables created via SQLAlchemy")

    db_path = make_url(settings.DATABASE_URL).database
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        await saver.setup()
    print(f"[checkpoint] langgraph checkpoint tables created at {db_path}")

    con = sqlite3.connect(db_path)
    try:
        all_tables = {
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        print("[status]")
        for table in EXPECTED_TABLES:
            print(f"  [{'OK' if table in all_tables else 'MISSING'}] {table}")
        extra = sorted(all_tables - set(EXPECTED_TABLES))
        if extra:
            print("  [info] other tables:", ", ".join(extra))
        print("  [pragma] journal_mode =", con.execute("PRAGMA journal_mode").fetchone()[0])
    finally:
        con.close()


if __name__ == "__main__":
    asyncio.run(main())
