"""S6 文献工作流测试：上传/状态/审核端到端 + HitL + checkpointer 恢复。

- 确定性用例：FakeExtractor 替身（LLM 之外全是 S3/S5 真能力）
- 恢复用例：AsyncSqliteSaver 同一文件模拟重启
- 1 个用例真实调 DeepSeek 跑完整链路（不可用才 skip 并报告原因）
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import fitz
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.schemas import (
    CalculationType,
    DFTParameters,
    ExtractedCalculation,
    FunctionalFamily,
    KPoints,
    KPointsScheme,
)
from app.main import create_app
from app.services.llm_extractor import LLMExtractor
from app.services.pdf_parser import PDFParser
from app.services.query_cards import QueryCards
from app.services.rule_validator import RuleValidatorService
from app.services.vector_searcher import VectorSearcher
from app.storage.chroma import ChromaStore
from app.storage.db import Base, Card, Paper, create_engine
from app.storage.files import FileStore, ServiceError
from app.workflows.literature.graph import build_literature_graph
from app.workflows.literature.nodes import LiteratureServices

FIXTURES = Path(__file__).parent / "fixtures"
_PAPER_A = FIXTURES / "papers" / "Bulk and surface properties of spinel Co3O4 by density functional calculations.pdf"
_PAPER_B = next(FIXTURES.glob("papers/A DFT and microkinetic*"))

# ============ 构造器 ============


class FakeExtractor:
    """LLMExtractor 测试替身：按 paper_id 返回固定抽取结果。"""

    def __init__(self, results=None, by_paper: dict | None = None):
        self._results = list(results or [])
        self._by_paper = by_paper or {}
        self.calls: list[str] = []

    async def extract(self, paper_id, method_text, tables, captions):
        self.calls.append(paper_id)
        if paper_id in self._by_paper:
            return self._by_paper[paper_id]
        if self._results:
            return [self._results.pop(0)]  # 与真实 LLMExtractor 一致：返回列表
        return []


def _calc(*, encut: float | None = 520.0, family: FunctionalFamily = FunctionalFamily.PBE) -> ExtractedCalculation:
    return ExtractedCalculation(
        system_formula="Co3O4",
        parameters=DFTParameters(
            calculation_type=CalculationType.RELAX,
            functional="PBE+U" if family is FunctionalFamily.PBE_U else "PBE",
            functional_family=family,
            hubbard_u=[],
            encut_ev=encut,
            kpoints=KPoints(scheme=KPointsScheme.GAMMA, mesh=(3, 3, 1), gamma_centered=True),
        ),
        surface=None,
        confidence=0.9,
    )


def _make_services(tmp_path, extractor, *, chroma_available: bool = False) -> tuple[LiteratureServices, object]:
    data_root = tmp_path / "data"
    db_path = tmp_path / "db" / "chem.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite+aiosqlite:///{str(db_path).replace(chr(92), '/')}"
    engine = create_engine(url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    file_store = FileStore(data_root)
    file_store.ensure_base_dirs()
    chroma_store = ChromaStore(tmp_path / "chroma", "paraphrase-multilingual-MiniLM-L12-v2")
    if not chroma_available:
        chroma_store._available = False  # 预设降级：不触发模型加载（upsert 走 5002 真路径）
    services = LiteratureServices(
        pdf_parser=PDFParser(),
        llm_extractor=extractor,
        rule_validator=RuleValidatorService(),
        query_cards=QueryCards(session_factory),
        vector_searcher=VectorSearcher(chroma_store),
        session_factory=session_factory,
        file_store=file_store,
    )
    return services, engine


async def _create_tables(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _seed_paper(session_factory, paper_id: str, status: str = "queued") -> None:
    async with session_factory() as s:
        s.add(Paper(
            paper_id=paper_id, filename=f"{paper_id}.pdf",
            file_path=f"data/papers/{paper_id}.pdf", file_hash=paper_id,
            file_size=1, status=status, thread_id=f"lit:{paper_id}", meta_json="{}",
        ))
        await s.commit()


def _make_app(tmp_path, extractor, *, chroma_available: bool = False):
    services, engine = _make_services(tmp_path, extractor, chroma_available=chroma_available)
    asyncio.run(_create_tables(engine))
    db_path = str(tmp_path / "db" / "chem.db")
    app = create_app(
        services,
        db_path=db_path,
        data_dir=str(tmp_path / "data"),
        chroma_dir=str(tmp_path / "chroma"),
    )
    return app, services, engine


def _blank_pdf(path: Path, text: str | None = None) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    if text:
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


# ============ API 端到端（TestClient + FakeExtractor） ============


class TestUploadFlow:
    def test_clean_paper_saved_directly(self, tmp_path):
        app, services, engine = _make_app(tmp_path, FakeExtractor([_calc()]))
        with TestClient(app) as client:
            with open(_PAPER_A, "rb") as f:
                r = client.post("/documents/upload", files={"file": ("p.pdf", f.read(), "application/pdf")})
            assert r.status_code == 202
            paper_id = r.json()["data"]["paper_id"]
            status = client.get(f"/documents/{paper_id}").json()["data"]
            assert status["status"] == "saved"
            assert status["pending_action"] is None
            assert len(status["card_ids"]) == 1
        # cards 表落库（auto_saved：无 warning 不触发审核）
        cards = asyncio.run(_fetch_cards(engine))
        assert len(cards) == 1
        assert cards[0].review_status == "auto_saved"

    def test_warning_paper_needs_review(self, tmp_path):
        app, _, engine = _make_app(tmp_path, FakeExtractor([_calc(encut=None)]))  # 缺 encut → R5 warning
        with TestClient(app) as client:
            with open(_PAPER_A, "rb") as f:
                r = client.post("/documents/upload", files={"file": ("p.pdf", f.read(), "application/pdf")})
            paper_id = r.json()["data"]["paper_id"]
            status = client.get(f"/documents/{paper_id}").json()["data"]
            assert status["status"] == "needs_review"
            assert status["pending_action"] == "review"
            assert status["thread_id"] == f"lit:{paper_id}"

    def test_review_approve_flow(self, tmp_path):
        app, _, engine = _make_app(tmp_path, FakeExtractor([_calc(encut=None)]), chroma_available=True)
        with TestClient(app) as client:
            with open(_PAPER_A, "rb") as f:
                r = client.post("/documents/upload", files={"file": ("p.pdf", f.read(), "application/pdf")})
            paper_id = r.json()["data"]["paper_id"]
            review = client.post("/chat", json={
                "paper_id": paper_id, "action": "review",
                "payload": {"decisions": [{"index": 0, "decision": "approve"}]},
            })
            assert review.status_code == 200
            data = review.json()["data"]
            assert data["type"] == "cards"
            assert len(data["cards"]) == 1
            status = client.get(f"/documents/{paper_id}").json()["data"]
            assert status["status"] == "saved"
        cards = asyncio.run(_fetch_cards(engine))
        assert len(cards) == 1
        assert cards[0].review_status == "reviewed"
        assert cards[0].chroma_indexed == 1  # Chroma 可用 → 索引标记

    def test_review_edit_applied(self, tmp_path):
        app, _, engine = _make_app(tmp_path, FakeExtractor([_calc(encut=350.0)]))  # encut<400 → R7 warning
        with TestClient(app) as client:
            with open(_PAPER_A, "rb") as f:
                r = client.post("/documents/upload", files={"file": ("p.pdf", f.read(), "application/pdf")})
            paper_id = r.json()["data"]["paper_id"]
            review = client.post("/chat", json={
                "paper_id": paper_id, "action": "review",
                "payload": {"decisions": [{
                    "index": 0, "decision": "edit",
                    "parameters": {"encut_ev": 520.0},
                }]},
            })
            assert review.status_code == 200
            assert review.json()["data"]["type"] == "cards"
        cards = asyncio.run(_fetch_cards(engine))
        assert len(cards) == 1
        assert cards[0].parameters_json.find("520.0") != -1

    def test_approve_with_error_rejected_then_edit(self, tmp_path):
        # PBE+U 无 U → R1 error；approve 无效 → 再返回 review_request；edit 补 U 才能过
        pbeu_no_u = ExtractedCalculation(
            system_formula="Co3O4",
            parameters=DFTParameters(
                calculation_type=CalculationType.RELAX,
                functional="PBE+U", functional_family=FunctionalFamily.PBE_U,
                hubbard_u=[],
                encut_ev=520.0,
                kpoints=KPoints(scheme=KPointsScheme.GAMMA, mesh=(3, 3, 1), gamma_centered=True),
            ),
            confidence=0.9,
        )
        app, _, engine = _make_app(tmp_path, FakeExtractor([pbeu_no_u]))
        with TestClient(app) as client:
            with open(_PAPER_A, "rb") as f:
                r = client.post("/documents/upload", files={"file": ("p.pdf", f.read(), "application/pdf")})
            paper_id = r.json()["data"]["paper_id"]
            review = client.post("/chat", json={
                "paper_id": paper_id, "action": "review",
                "payload": {"decisions": [{"index": 0, "decision": "approve"}]},
            })
            data = review.json()["data"]
            assert data["type"] == "review_request"  # approve 被拒 → 再次暂停
            assert data["round"] == 2
            review2 = client.post("/chat", json={
                "paper_id": paper_id, "action": "review",
                "payload": {"decisions": [{
                    "index": 0, "decision": "edit",
                    "parameters": {"hubbard_u": [{"element": "Co", "orbital": "d", "u_ev": 3.5}]},
                }]},
            })
            assert review2.json()["data"]["type"] == "cards"
            status = client.get(f"/documents/{paper_id}").json()["data"]
            assert status["status"] == "saved"

    def test_three_round_cap(self, tmp_path):
        pbeu_no_u = ExtractedCalculation(
            system_formula="Co3O4",
            parameters=DFTParameters(
                calculation_type=CalculationType.RELAX,
                functional="PBE+U", functional_family=FunctionalFamily.PBE_U,
                hubbard_u=[],
                encut_ev=520.0,
                kpoints=KPoints(scheme=KPointsScheme.GAMMA, mesh=(3, 3, 1), gamma_centered=True),
            ),
            confidence=0.9,
        )
        app, _, _ = _make_app(tmp_path, FakeExtractor([pbeu_no_u]))
        with TestClient(app) as client:
            with open(_PAPER_A, "rb") as f:
                r = client.post("/documents/upload", files={"file": ("p.pdf", f.read(), "application/pdf")})
            paper_id = r.json()["data"]["paper_id"]
            rounds = 0
            review = None
            for _ in range(3):
                review = client.post("/chat", json={
                    "paper_id": paper_id, "action": "review",
                    "payload": {"decisions": [{"index": 0, "decision": "approve"}]},
                })
                rounds += 1
                if review.json()["data"]["type"] != "review_request":
                    break
            assert rounds == 3
            status = client.get(f"/documents/{paper_id}").json()["data"]
            assert status["status"] == "failed"
            assert status["error_code"] == 2005

    def test_scanned_pdf_failed(self, tmp_path):
        app, _, _ = _make_app(tmp_path, FakeExtractor([_calc()]))
        pdf = _blank_pdf(tmp_path / "scanned.pdf")
        with TestClient(app) as client:
            with open(pdf, "rb") as f:
                r = client.post("/documents/upload", files={"file": ("s.pdf", f.read(), "application/pdf")})
            paper_id = r.json()["data"]["paper_id"]
            status = client.get(f"/documents/{paper_id}").json()["data"]
            assert status["status"] == "failed"
            assert status["error_code"] == 2002

    def test_method_not_found(self, tmp_path):
        app, _, _ = _make_app(tmp_path, FakeExtractor([_calc()]))
        pdf = _blank_pdf(tmp_path / "plain.pdf", text="Introduction to materials science. "
                        "We study crystal growth and mechanical properties of alloys.")
        with TestClient(app) as client:
            with open(pdf, "rb") as f:
                r = client.post("/documents/upload", files={"file": ("m.pdf", f.read(), "application/pdf")})
            paper_id = r.json()["data"]["paper_id"]
            status = client.get(f"/documents/{paper_id}").json()["data"]
            assert status["status"] == "method_not_found"

    def test_duplicate_upload_returns_200(self, tmp_path):
        app, _, _ = _make_app(tmp_path, FakeExtractor([_calc()]))
        with open(_PAPER_B, "rb") as f:
            pdf_bytes = f.read()
        with TestClient(app) as client:
            r1 = client.post("/documents/upload", files={"file": ("p.pdf", pdf_bytes, "application/pdf")})
            assert r1.status_code == 202
            r2 = client.post("/documents/upload", files={"file": ("p.pdf", pdf_bytes, "application/pdf")})
            assert r2.status_code == 200
            assert r2.json()["data"]["status"] == "saved"

    def test_retry_rules(self, tmp_path):
        app, _, _ = _make_app(tmp_path, FakeExtractor([_calc()]))
        with open(_PAPER_B, "rb") as f:
            pdf_bytes = f.read()
        with TestClient(app) as client:
            r1 = client.post("/documents/upload", files={"file": ("p.pdf", pdf_bytes, "application/pdf")})
            paper_id = r1.json()["data"]["paper_id"]
            retry_saved = client.post(f"/documents/{paper_id}/retry")
            assert retry_saved.status_code == 409
        # failed 的 paper：用扫描版走 failed，然后 retry（fake 抽取干净 → 但扫描版仍 failed）
        with TestClient(app) as client:
            scanned = _blank_pdf(tmp_path / "scanned2.pdf")
            with open(scanned, "rb") as f:
                r = client.post("/documents/upload", files={"file": ("s.pdf", f.read(), "application/pdf")})
            failed_id = r.json()["data"]["paper_id"]
            status = client.get(f"/documents/{failed_id}").json()["data"]
            assert status["status"] == "failed"
            retry = client.post(f"/documents/{failed_id}/retry")
            assert retry.status_code == 202

    def test_chat_message_routed(self, tmp_path):
        # S8 起 action=message 已实现（原 501 占位被替换）：断言返回有效业务类型
        app, _, _ = _make_app(tmp_path, FakeExtractor([_calc()]))
        with TestClient(app) as client:
            r = client.post("/chat", json={"message": "提取这篇论文", "action": "message"})
            assert r.status_code == 200
            assert r.json()["data"]["type"] in ("clarify", "cards", "document_status", "results")


async def _fetch_cards(engine):
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        return (await s.execute(select(Card))).scalars().all()


# ============ 审核去重（方案 A：跨轮已决索引，重复 decisions 不二次保存） ============


class TwoCalcExtractor:
    """返回两条候选（均缺 encut → R5 warning → 触发审核）。"""

    async def extract(self, paper_id, method_text, tables, captions):
        return [_calc(encut=None), _calc(encut=None)]


class TestReviewDedup:
    @pytest.mark.asyncio
    async def test_duplicate_decisions_not_double_saved(self, tmp_path):
        services, engine = _make_services(tmp_path, TwoCalcExtractor())
        await _create_tables(engine)
        await _seed_paper(services.session_factory, "p1")

        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command as Cmd

        graph = build_literature_graph(InMemorySaver(), services)
        cfg = {"configurable": {"thread_id": "lit:p1"}}
        out1 = await graph.ainvoke(
            {"paper_id": "p1", "pdf_path": str(_PAPER_A)}, config=cfg
        )
        assert "__interrupt__" in out1

        # 第 1 轮：只 approve 候选 0
        out2 = await graph.ainvoke(
            Cmd(resume={"decisions": [{"index": 0, "decision": "approve"}]}), config=cfg
        )
        assert "__interrupt__" in out2  # 候选 1 仍 pending → 自环

        # 第 2 轮：payload 全量（含已决的 0）——重复决策应被服务端跳过
        out3 = await graph.ainvoke(
            Cmd(resume={"decisions": [
                {"index": 0, "decision": "approve"},
                {"index": 1, "decision": "approve"},
            ]}),
            config=cfg,
        )
        assert out3.get("status") == "approved"
        saved = out3.get("saved_card_ids", [])
        assert len(saved) == 2
        assert len(set(saved)) == 2  # 无重复 card_id
        async with services.session_factory() as s:
            cards = (await s.execute(select(Card))).scalars().all()
            assert len(cards) == 2
        await engine.dispose()


# ============ checkpointer 恢复（图级，AsyncSqliteSaver 同文件重启） ============


class TestCheckpointerRestart:
    @pytest.mark.asyncio
    async def test_restart_resume_with_same_sqlite_file(self, tmp_path):
        services, engine = _make_services(tmp_path, FakeExtractor([_calc(encut=None)]))
        await _create_tables(engine)
        await _seed_paper(services.session_factory, "p1")
        cp_path = str(tmp_path / "db" / "cp.db")

        async with AsyncSqliteSaver.from_conn_string(cp_path) as saver:
            await saver.setup()
            graph = build_literature_graph(saver, services)
            out1 = await graph.ainvoke(
                {"paper_id": "p1", "pdf_path": str(_PAPER_A)},
                config={"configurable": {"thread_id": "lit:p1"}},
            )
            assert "__interrupt__" in out1

        # 模拟重启：同一文件、全新 saver + 全新图
        async with AsyncSqliteSaver.from_conn_string(cp_path) as saver2:
            await saver2.setup()
            graph2 = build_literature_graph(saver2, services)
            out2 = await graph2.ainvoke(
                Command(resume={"decisions": [{"index": 0, "decision": "approve"}]}),
                config={"configurable": {"thread_id": "lit:p1"}},
            )
            assert out2.get("status") == "approved"
            assert out2.get("saved_card_ids")

        async with services.session_factory() as s:
            row = await s.get(Paper, "p1")
            assert row.status == "saved"
        await engine.dispose()


# ============ 真实 DeepSeek 端到端 ============


class TestRealDeepSeekE2E:
    @pytest.mark.asyncio
    async def test_full_chain_with_real_llm(self, tmp_path):
        services, engine = _make_services(tmp_path, LLMExtractor())
        await _create_tables(engine)
        await _seed_paper(services.session_factory, "p1")
        cp_path = str(tmp_path / "db" / "cp.db")
        try:
            async with AsyncSqliteSaver.from_conn_string(cp_path) as saver:
                await saver.setup()
                graph = build_literature_graph(saver, services)
                cfg = {"configurable": {"thread_id": "lit:p1"}}
                out = await graph.ainvoke(
                    {"paper_id": "p1", "pdf_path": str(_PAPER_B)},
                    config=cfg,
                )
                if "__interrupt__" in out:
                    payload = out["__interrupt__"][0].value
                    decisions = []
                    for c in payload["candidates"]:
                        has_error = any(f["severity"] == "error" for f in c["findings"])
                        decisions.append({
                            "index": c["index"],
                            "decision": "reject" if has_error else "approve",
                        })
                    out = await graph.ainvoke(Command(resume={"decisions": decisions}), config=cfg)
                # 最终状态以 papers 表为准（无审核路径不写 state.status）
        except ServiceError as exc:
            if exc.code == 5001:
                pytest.skip(f"DeepSeek 不可用，跳过真实端到端：{exc.details}")
            raise
        async with services.session_factory() as s:
            row = await s.get(Paper, "p1")
            assert row.status in ("saved", "rejected")
        await engine.dispose()
