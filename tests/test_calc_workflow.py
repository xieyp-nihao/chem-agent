"""S7 计算工作流测试：/calc/draft 端到端 + confirm/patch + 输出文件。

- 确定性用例：FakeLLM 替身（槽位解析之外全是 S3/S5 真能力；CIF 用例完全离线）
- 1 个用例真实调 DeepSeek 解析自然语言槽位（不可用才 skip 并报告原因）
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import AIMessage
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import settings
from app.domain.schemas import (
    CalcSlots,
    CalculationCard,
    CalculationType,
    DFTParameters,
    FunctionalFamily,
    KPoints,
    KPointsScheme,
    SurfaceModel,
    ValidationReport,
)
from app.main import create_app
from app.services.llm_extractor import LLMExtractor
from app.services.mp_client import MPClient
from app.services.pdf_parser import PDFParser
from app.services.query_cards import QueryCards
from app.services.rule_validator import RuleValidatorService
from app.services.surface_builder import SurfaceBuilder
from app.services.vasp_generator import VASPGenerator
from app.services.vector_searcher import VectorSearcher
from app.storage.chroma import ChromaStore
from app.storage.db import Base, Paper, create_engine
from app.storage.files import FileStore
from app.workflows.calc.graph import build_calc_graph
from app.workflows.calc.nodes import CalcServices
from app.workflows.literature.nodes import LiteratureServices

FIXTURES = Path(__file__).parent / "fixtures"
_CIF = FIXTURES / "structures" / "co3o4_bulk.cif"

_BASE_SLOTS = {
    "calculation_type": "relax",
    "functional": "PBE",
    "functional_family": "PBE",
    "encut_ev": 520.0,
    "kpoints": {"scheme": "gamma", "mesh": [3, 3, 1], "gamma_centered": True},
    "surface": {"miller": [1, 1, 1], "layers": 6, "vacuum_angstrom": 15.0},
}
_SLOTS_WITH_MP = {**_BASE_SLOTS, "formula": "Co3O4", "mp_id": "mp-18748"}


class FakeLLM:
    def __init__(self, response: str):
        self._response = AIMessage(content=response)

    async def ainvoke(self, messages):
        return self._response


def _make_calc_services(tmp_path, fake_llm: FakeLLM | None = None, real_llm: bool = False):
    data_root = tmp_path / "data"
    db_path = tmp_path / "db" / "chem.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite+aiosqlite:///{str(db_path).replace(chr(92), '/')}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    file_store = FileStore(data_root)
    file_store.ensure_base_dirs()
    llm_holder = SimpleNamespace(_model=fake_llm) if fake_llm else LLMExtractor()
    calc = CalcServices(
        session_factory=session_factory,
        file_store=file_store,
        query_cards=QueryCards(session_factory),
        llm_extractor=llm_holder,
        mp_client=MPClient(api_key=settings.MP_API_KEY, cache_dir=tmp_path / "mp_cache"),
        surface_builder=SurfaceBuilder(),
        vasp_generator=VASPGenerator(),
        rule_validator=RuleValidatorService(),
    )
    return calc, engine, session_factory, file_store


def _make_lit_services(tmp_path, session_factory, file_store):
    chroma = ChromaStore(tmp_path / "chroma", settings.EMBEDDING_MODEL)
    chroma._available = False
    return LiteratureServices(
        pdf_parser=PDFParser(),
        llm_extractor=LLMExtractor(),
        rule_validator=RuleValidatorService(),
        query_cards=QueryCards(session_factory),
        vector_searcher=VectorSearcher(chroma),
        session_factory=session_factory,
        file_store=file_store,
    )


def _make_app(tmp_path, fake_llm: FakeLLM | None = None, real_llm: bool = False):
    calc, engine, session_factory, file_store = _make_calc_services(
        tmp_path, fake_llm=fake_llm, real_llm=real_llm
    )
    lit = _make_lit_services(tmp_path, session_factory, file_store)
    asyncio.run(_create_tables(engine))
    app = create_app(
        lit, calc,
        db_path=str(tmp_path / "db" / "chem.db"),
        data_dir=str(tmp_path / "data"),
        chroma_dir=str(tmp_path / "chroma"),
        mp_api_key=settings.MP_API_KEY,
    )
    return app, engine, session_factory, file_store


async def _create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _draft(client, payload):
    return client.post("/calc/draft", json=payload)


def _draft_multipart(client, payload: dict, cif_path: Path):
    return client.post(
        "/calc/draft",
        data={"payload": json.dumps(payload)},
        files={"structure_file": (cif_path.name, open(cif_path, "rb"), "chemical/x-cif")},
    )


def _confirm(client, session_id, decision, patch=None):
    payload = {"decision": decision}
    if patch is not None:
        payload["patch"] = patch
    return client.post("/chat", json={"session_id": session_id, "action": "confirm", "payload": payload})


# ============ API 端到端 ============


class TestCalcDraftFlow:
    @pytest.mark.llm  # 真实 MP 调用（mp_id=mp-18748）
    def test_json_draft_with_mp_approve_writes_files(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        with TestClient(app) as client:
            r = _draft(client, {"slots": _SLOTS_WITH_MP})
            assert r.status_code == 200
            data = r.json()["data"]
            assert data["type"] == "confirm_request"
            session_id = data["session_id"]
            assert data["draft"]["incar"].find("ENCUT") != -1
            r2 = _confirm(client, session_id, "approve")
            assert r2.status_code == 200
            out = r2.json()["data"]
            assert out["type"] == "output_files"
            assert len(out["files"]) == 4
            # 文件真实存在
            for f in out["files"]:
                path = Path(f["path"])
                assert path.is_file(), f["name"]
            incar = (Path(tmp_path) / "data" / "outputs" / f"calc_{session_id}" / "INCAR").read_text()
            assert "ENCUT = 520" in incar
            assert "ISMEAR = 0" in incar

    def test_multipart_draft_with_cif(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        with TestClient(app) as client:
            r = _draft_multipart(client, {"slots": _BASE_SLOTS}, _CIF)
            assert r.status_code == 200
            data = r.json()["data"]
            assert data["type"] == "confirm_request"
            session_id = data["session_id"]
            r2 = _confirm(client, session_id, "approve")
            out = r2.json()["data"]
            assert out["type"] == "output_files"
            assert all(Path(f["path"]).is_file() for f in out["files"])

    def test_missing_slot_422(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        with TestClient(app) as client:
            slots = dict(_SLOTS_WITH_MP)
            slots.pop("functional")
            slots.pop("functional_family")
            r = _draft(client, {"slots": slots})
            assert r.status_code == 422
            body = r.json()
            assert body["code"] == 3001
            assert body["data"]["type"] == "clarify"
            assert "functional" in body["data"]["missing_slots"]

    def test_unsupported_keyword_422(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        with TestClient(app) as client:
            slots = dict(_SLOTS_WITH_MP)
            slots["notes"] = "帮我算氧空位形成能"
            r = _draft(client, {"slots": slots})
            assert r.status_code == 422
            body = r.json()
            assert body["code"] == 3002
            assert body["data"]["type"] == "clarify"
            assert body["data"]["suggestions"]

    def test_card_id_prefill(self, tmp_path):
        app, engine, session_factory, _ = _make_app(tmp_path)
        # 先写一张卡（encut 520 来自卡片）
        asyncio.run(_seed_paper(session_factory, "p1"))
        card = CalculationCard(
            card_id=uuid4(), paper_id="p1", system_formula="Co3O4", system_normalized="Co3O4",
            parameters=DFTParameters(
                calculation_type=CalculationType.RELAX, functional="PBE",
                functional_family=FunctionalFamily.PBE, hubbard_u=[],
                encut_ev=520.0,
                kpoints=KPoints(scheme=KPointsScheme.GAMMA, mesh=(3, 3, 1), gamma_centered=True),
            ),
            missing_fields=[], validation=ValidationReport(),
            created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            updated_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )
        asyncio.run(QueryCards(session_factory).save_card(card))
        with TestClient(app) as client:
            r = _draft_multipart(
                client,
                {"card_id": str(card.card_id), "slots": {"surface": _BASE_SLOTS["surface"]}},
                _CIF,
            )
            assert r.status_code == 200
            data = r.json()["data"]
            assert data["type"] == "confirm_request"
            assert "ENCUT = 520" in data["draft"]["incar"]  # 卡片预填

    def test_patch_forbidden_field(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        with TestClient(app) as client:
            r = _draft_multipart(client, {"slots": _BASE_SLOTS}, _CIF)
            session_id = r.json()["data"]["session_id"]
            r2 = _confirm(client, session_id, "patch", {"miller": [1, 1, 0]})
            body = r2.json()
            assert body["code"] == 4002

    def test_patch_allowed_field(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        with TestClient(app) as client:
            r = _draft_multipart(client, {"slots": _BASE_SLOTS}, _CIF)
            session_id = r.json()["data"]["session_id"]
            r2 = _confirm(client, session_id, "patch", {"encut_ev": 600.0})
            assert r2.status_code == 200
            assert r2.json()["data"]["type"] == "output_files"
            incar = (Path(tmp_path) / "data" / "outputs" / f"calc_{session_id}" / "INCAR").read_text()
            assert "ENCUT = 600" in incar

    def test_patch_leaves_error_re_interrupt(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        slots = dict(_BASE_SLOTS)
        slots.update({
            "functional": "PBE+U", "functional_family": "PBE+U",
            "hubbard_u": [{"element": "Co", "orbital": "d", "u_ev": 3.5, "j_ev": 0.0}],
        })
        with TestClient(app) as client:
            r = _draft_multipart(client, {"slots": slots}, _CIF)
            session_id = r.json()["data"]["session_id"]
            r2 = _confirm(client, session_id, "patch", {"hubbard_u": []})
            data = r2.json()["data"]
            assert data["type"] == "confirm_request"  # patch 后 R1 error → 再次暂停
            assert data["round"] == 2

    def test_three_round_cap(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        slots = dict(_BASE_SLOTS)
        slots.update({
            "functional": "PBE+U", "functional_family": "PBE+U",
            "hubbard_u": [{"element": "Co", "orbital": "d", "u_ev": 3.5, "j_ev": 0.0}],
        })
        with TestClient(app) as client:
            r = _draft_multipart(client, {"slots": slots}, _CIF)
            session_id = r.json()["data"]["session_id"]
            last = None
            for _ in range(3):
                last = _confirm(client, session_id, "patch", {"hubbard_u": []})
                body = last.json()
                # 错误响应 data 为 null；confirm_request 说明仍在轮次内
                if body["code"] != 0 or not body["data"] or body["data"]["type"] != "confirm_request":
                    break
            body = last.json()
            assert body["code"] == 3007

    def test_abort_writes_nothing(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        with TestClient(app) as client:
            r = _draft_multipart(client, {"slots": _BASE_SLOTS}, _CIF)
            session_id = r.json()["data"]["session_id"]
            r2 = _confirm(client, session_id, "abort")
            assert r2.json()["data"]["type"] == "ack"
            f = client.get(f"/files/calc_{session_id}/INCAR")
            assert f.status_code == 404

    def test_same_session_second_draft_409(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        with TestClient(app) as client:
            r = _draft_multipart(client, {"slots": _BASE_SLOTS}, _CIF)
            session_id = r.json()["data"]["session_id"]
            r2 = _draft_multipart(client, {"session_id": session_id, "slots": _BASE_SLOTS}, _CIF)
            assert r2.status_code == 409
            assert r2.json()["code"] == 4003

    def test_files_traversal_and_normal(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        with TestClient(app) as client:
            r = _draft_multipart(client, {"slots": _BASE_SLOTS}, _CIF)
            session_id = r.json()["data"]["session_id"]
            _confirm(client, session_id, "approve")
            normal = client.get(f"/files/calc_{session_id}/INCAR")
            assert normal.status_code == 200
            assert b"ENCUT" in normal.content
            traversal = client.get("/files/..%2F..%2Fetc%2Fpasswd")
            assert traversal.status_code in (403, 404)

    def test_confirm_missing_thread_4002(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path)
        with TestClient(app) as client:
            r = _confirm(client, "nonexistent-session", "approve")
            assert r.json()["code"] == 4002


async def _seed_paper(session_factory, paper_id):
    async with session_factory() as s:
        s.add(Paper(
            paper_id=paper_id, filename=f"{paper_id}.pdf",
            file_path=f"data/papers/{paper_id}.pdf", file_hash=paper_id,
            file_size=1, status="saved", thread_id=f"lit:{paper_id}", meta_json="{}",
        ))
        await s.commit()


# ============ 重启恢复（图级） ============


class TestCalcRestart:
    @pytest.mark.asyncio
    async def test_restart_resume(self, tmp_path):
        calc, engine, session_factory, file_store = _make_calc_services(tmp_path)
        await _create_tables(engine)
        file_store.save("structures", "s1__bulk.cif", _CIF.read_bytes())
        slots = dict(_BASE_SLOTS)
        slots["cif_path"] = "s1__bulk.cif"
        slots["structure_source"] = "cif_upload"
        cp_path = str(tmp_path / "db" / "cp.db")

        state = {"session_id": "s1", "thread_id": "calc:s1",
                 "raw_message": None, "user_slots": slots, "card_id": None}
        cfg = {"configurable": {"thread_id": "calc:s1"}}

        async with AsyncSqliteSaver.from_conn_string(cp_path) as saver:
            await saver.setup()
            graph = build_calc_graph(saver, calc)
            out1 = await graph.ainvoke(state, config=cfg)
            assert "__interrupt__" in out1

        async with AsyncSqliteSaver.from_conn_string(cp_path) as saver2:
            await saver2.setup()
            graph2 = build_calc_graph(saver2, calc)
            out2 = await graph2.ainvoke(
                Command(resume={"decision": "approve"}), config=cfg
            )
            assert out2.get("output_files")
        assert (tmp_path / "data" / "outputs" / "calc_s1" / "INCAR").is_file()
        await engine.dispose()


# ============ 真实 DeepSeek 槽位解析 ============


class TestRealSlotParsing:
    @pytest.mark.llm  # 真实 DeepSeek 槽位解析 + 真实 MP formula 调用
    @pytest.mark.asyncio
    async def test_natural_language_slots_with_real_llm(self, tmp_path):
        calc, engine, _, _ = _make_calc_services(tmp_path, real_llm=True)
        await _create_tables(engine)
        cp_path = str(tmp_path / "db" / "cp.db")
        cfg = {"configurable": {"thread_id": "calc:real1"}}
        state = {
            "session_id": "real1", "thread_id": "calc:real1",
            "raw_message": "用 PBE+U 对 Co3O4 (111) 做 6 层 slab 弛豫，真空 15 埃，U 取 3.5 eV",
            "user_slots": {}, "card_id": None,
        }
        try:
            async with AsyncSqliteSaver.from_conn_string(cp_path) as saver:
                await saver.setup()
                graph = build_calc_graph(saver, calc)
                out = await graph.ainvoke(state, config=cfg)
                assert "__interrupt__" in out, out.get("error")
                out2 = await graph.ainvoke(Command(resume={"decision": "approve"}), config=cfg)
                assert out2.get("output_files")
        except Exception as exc:
            # DeepSeek/MP 不可用才跳过（报告原因）
            if "LLM_UNAVAILABLE" in str(exc) or "MP_UNAVAILABLE" in str(exc):
                pytest.skip(f"外部依赖不可用，跳过真实端到端：{str(exc)[:120]}")
            raise
        await engine.dispose()
