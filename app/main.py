"""FastAPI 应用装配（§5.4 / §8.2 S6）。

- lifespan：目录骨架 + 建表（幂等）→ AsyncSqliteSaver（复用 S2 的
  data/chem-agent.db 同一文件，setup() 幂等，上下文管理器负责释放连接）
  → 组装 services → 编译文献图挂到 app.state
- Chroma 不可用只记 warning，不阻断启动
- 不做 CORS / /health / 其他路由（S8）
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api import calc, chat, documents, files
from app.config import settings
from app.services.llm_extractor import LLMExtractor
from app.services.mp_client import MPClient
from app.services.pdf_parser import PDFParser
from app.services.query_cards import QueryCards
from app.services.rule_validator import RuleValidatorService
from app.services.surface_builder import SurfaceBuilder
from app.services.vasp_generator import VASPGenerator
from app.services.vector_searcher import VectorSearcher
from app.storage.chroma import ChromaStore
from app.storage.db import Base, create_engine
from app.storage.files import FileStore
from app.workflows.calc.graph import build_calc_graph
from app.workflows.calc.nodes import CalcServices
from app.workflows.literature.graph import build_literature_graph
from app.workflows.literature.nodes import LiteratureServices

logger = logging.getLogger(__name__)


def _default_services(session_factory, file_store: FileStore, chroma_dir: str) -> LiteratureServices:
    chroma_store = ChromaStore(Path(chroma_dir), settings.EMBEDDING_MODEL)
    if not chroma_store.is_available():
        logger.warning("Chroma/嵌入模型不可用，语义检索将降级（不影响 SQL 路径）")
    return LiteratureServices(
        pdf_parser=PDFParser(),
        llm_extractor=LLMExtractor(),
        rule_validator=RuleValidatorService(),
        query_cards=QueryCards(session_factory),
        vector_searcher=VectorSearcher(chroma_store),
        session_factory=session_factory,
        file_store=file_store,
    )


def create_app(
    lit_services: LiteratureServices | None = None,
    calc_services: CalcServices | None = None,
    *,
    db_path: str | None = None,
    data_dir: str | None = None,
    chroma_dir: str | None = None,
    mp_api_key: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        data_root = Path(data_dir or settings.DATA_DIR)
        file_store = FileStore(data_root)
        file_store.ensure_base_dirs()

        url = f"sqlite+aiosqlite:///{str(db_path).replace(chr(92), '/')}" if db_path else settings.DATABASE_URL
        engine = create_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        cp_path = db_path or make_url(settings.DATABASE_URL).database
        async with AsyncSqliteSaver.from_conn_string(cp_path) as saver:
            await saver.setup()  # 幂等：复用 S2 建的表
            lit = lit_services or _default_services(
                session_factory, file_store, chroma_dir or settings.CHROMA_DIR
            )
            lit_graph = build_literature_graph(saver, lit)

            calc_svc = calc_services or _default_calc_services(
                session_factory, file_store, lit, data_root,
                mp_api_key or settings.MP_API_KEY,
            )
            calc_graph = build_calc_graph(saver, calc_svc)

            app.state.literature_graph = lit_graph
            app.state.calc_graph = calc_graph
            app.state.session_factory = session_factory
            app.state.file_store = file_store
            app.state.query_cards = lit.query_cards
            app.state.vector_searcher = lit.vector_searcher
            app.state.engine = engine
            # 共享 LLM 客户端（thinking disabled）；orchestrator 复用，测试可注入替身
            app.state.llm_model = getattr(lit.llm_extractor, "_model", None)
            yield
        await engine.dispose()

    app = FastAPI(title="chem-agent", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(calc.router)
    app.include_router(files.router)

    @app.get("/health")
    async def health(request: Request):
        """轻量探测（§7.8）：llm/mp 只做配置检查，不调付费接口。"""
        from fastapi.responses import JSONResponse

        components = {
            "sqlite": "ok",
            "checkpointer": "ok",
            "llm": "ok" if settings.DEEPSEEK_API_KEY else "unavailable",
            "mp": "ok" if settings.MP_API_KEY else "unavailable",
            "chroma": "unavailable",
            "embedding_model": "unavailable",
        }
        try:
            engine = request.app.state.engine
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception:
            components["sqlite"] = "unavailable"
        try:
            await request.app.state.literature_graph.aget_state(
                {"configurable": {"thread_id": "__health__"}}
            )
        except Exception:
            components["checkpointer"] = "unavailable"
        try:
            if request.app.state.vector_searcher.is_available():
                components["chroma"] = "ok"
                components["embedding_model"] = "ok"
        except Exception:
            pass

        down = components["sqlite"] != "ok" or components["checkpointer"] != "ok"
        degraded = any(v == "unavailable" for v in components.values())
        status = "down" if down else ("degraded" if degraded else "ok")
        rid = documents._request_id(request)
        return JSONResponse(
            status_code=503 if down else 200,
            content=documents.make_envelope(
                0, status, {"type": "health", "status": status, "components": components}, rid
            ),
            headers={"X-Request-ID": rid},
        )

    return app


def _default_calc_services(
    session_factory, file_store: FileStore, lit: LiteratureServices,
    data_root: Path, mp_api_key: str,
) -> CalcServices:
    return CalcServices(
        session_factory=session_factory,
        file_store=file_store,
        query_cards=lit.query_cards,
        llm_extractor=lit.llm_extractor,
        mp_client=MPClient(api_key=mp_api_key, cache_dir=data_root / "structures"),
        surface_builder=SurfaceBuilder(),
        vasp_generator=VASPGenerator(),
        rule_validator=lit.rule_validator,
    )


app = create_app()
