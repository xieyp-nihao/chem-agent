"""S8 编排层测试：intent_split / rule_router / llm_fallback / cache / slot +
action=message / /health / CORS + 异常体系迁移。

- 确定性用例：FakeLLM 替身（llm_fallback / collect_slots）；结构生成用 CIF（离线）
  或真实 MP（mp-18748/formula，S5 已验证）
- 1 个用例真实调 DeepSeek 验证 llm_fallback（不可用才 skip）
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import settings
from app.core.exceptions import (
    AgentExecutionError,
    IntentRouteError,
    PipelineError,
    ServiceError,
)
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
from app.orchestrator.cache import cache_check
from app.orchestrator.intent_split import split_intents
from app.orchestrator.llm_fallback import llm_classify
from app.orchestrator.router import rule_route
from app.orchestrator.slot_manager import SlotManager
from app.services.llm_extractor import LLMExtractor
from app.services.mp_client import MPClient
from app.services.pdf_parser import PDFParser
from app.services.query_cards import QueryCards
from app.services.rule_validator import RuleValidatorService
from app.services.surface_builder import SurfaceBuilder
from app.services.vasp_generator import VASPGenerator
from app.services.vector_searcher import VectorSearcher
from app.storage.chroma import ChromaStore
from app.storage.db import Base, Paper, Session, create_engine
from app.storage.files import FileStore
from app.workflows.calc.nodes import CalcServices
from app.workflows.literature.nodes import LiteratureServices

FIXTURES = Path(__file__).parent / "fixtures"
_CIF = FIXTURES / "structures" / "co3o4_bulk.cif"

_SLOTS_JSON = json.dumps({
    "calculation_type": "relax",
    "functional": "PBE",
    "functional_family": "PBE",
    "formula": "Co3O4",
    "mp_id": "mp-18748",
    "encut_ev": 520.0,
    "kpoints": {"scheme": "gamma", "mesh": [3, 3, 1], "gamma_centered": True},
    "surface": {"miller": [1, 1, 1], "layers": 6, "vacuum_angstrom": 15.0},
})


class FakeLLM:
    def __init__(self, response: str):
        self._response = AIMessage(content=response)

    async def ainvoke(self, messages):
        return self._response

    def bind(self, **kwargs):
        return self  # 支持 llm_fallback 的 per-call bind


class SleepingLLM:
    async def ainvoke(self, messages):
        await asyncio.sleep(11)
        return AIMessage(content='{"intent": "calc", "reason": "x"}')

    def bind(self, **kwargs):
        return self


# ============ 单元：split / router / fallback ============


class TestIntentSplit:
    @pytest.mark.parametrize(
        ("message", "expected_len"),
        [
            ("提取参数，另外生成 INCAR", 2),
            ("帮我看看", 1),
            ("另外", 1),
            ("先算 bulk，另外切 slab，还有生成 KPOINTS", 2),  # 三段 → 两段
            ("", 0),
        ],
    )
    def test_split(self, message, expected_len):
        segments = split_intents(message)
        assert len(segments) == expected_len

    def test_segments_keep_original_text(self):
        segments = split_intents("提取参数，另外生成 INCAR")
        assert segments[0] == "提取参数，"  # 保留原始文本（含标点，不重写）
        assert segments[1].startswith("另外")


class TestRuleRouter:
    def test_literature_only(self):
        assert rule_route("提取这篇论文的 DFT 参数") == "literature"

    def test_calc_only(self):
        assert rule_route("帮我生成 INCAR") == "calc"

    def test_both_hit_fallback(self):
        assert rule_route("提取论文参数并生成 INCAR") == "fallback"

    def test_neither_hit_fallback(self):
        assert rule_route("你好呀") == "fallback"

    def test_paper_id_priority_on_ambiguous(self):
        # 同时命中两侧时 paper_id 优先（§2.4 "即使同时命中计算关键词"）
        assert rule_route("提取论文参数并生成 INCAR", paper_id="p1") == "literature"

    def test_card_id_priority_on_ambiguous(self):
        assert rule_route("提取论文参数并生成 INCAR", card_id="c1") == "calc"

    def test_paper_id_does_not_override_clear_calc(self):
        # 单侧明确命中的段不受 paper_id 影响（拆段后"另外生成 INCAR"仍走计算）
        assert rule_route("生成 INCAR", paper_id="p1") == "calc"


class TestLLMFallback:
    @pytest.mark.asyncio
    async def test_valid_intents(self):
        for intent in ("literature", "calc", "clarify"):
            fake = FakeLLM(json.dumps({"intent": intent, "reason": "r"}))
            assert await llm_classify(fake, "任意") == intent

    @pytest.mark.asyncio
    async def test_invalid_json_clarify(self):
        fake = FakeLLM("这不是 JSON")
        assert await llm_classify(fake, "任意") == "clarify"

    @pytest.mark.asyncio
    async def test_unknown_intent_clarify(self):
        fake = FakeLLM('{"intent": "dance", "reason": "r"}')
        assert await llm_classify(fake, "任意") == "clarify"

    @pytest.mark.asyncio
    async def test_timeout_clarify(self):
        assert await llm_classify(SleepingLLM(), "任意") == "clarify"  # 10s 超时

    @pytest.mark.llm  # 真实 DeepSeek（llm_fallback 真实分类）
    @pytest.mark.asyncio
    async def test_real_deepseek(self):
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            model = LLMExtractor()._model
            # 可用性预探测（llm_classify 失败会静默降级 clarify，无法区分）。
            # 注意：json_object 模式要求 prompt 含 "json" 字样，否则端点 400
            await model.ainvoke([
                SystemMessage(content="只输出一个 JSON 对象。"),
                HumanMessage(content="把这句话变成 JSON：你好"),
            ])
        except Exception as exc:
            pytest.skip(f"DeepSeek 不可用，跳过真实调用：{str(exc)[:120]}")
        intent = await llm_classify(model, "帮我提取这篇论文的 DFT 参数")
        assert intent in ("literature", "calc", "clarify")


# ============ 单元：cache / slot_manager ============


class TestCacheCheck:
    @pytest.mark.asyncio
    async def test_hit_returns_cache_cards(self, tmp_path):
        engine, session_factory = await _tmp_db(tmp_path)
        await _seed_paper(session_factory, "p1")
        q = QueryCards(session_factory)
        await q.save_card(_make_card("p1"))
        result = await cache_check(q, "p1")
        assert result is not None
        assert result["source"] == "cache"
        assert len(result["cards"]) == 1
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_miss_returns_none(self, tmp_path):
        engine, session_factory = await _tmp_db(tmp_path)
        await _seed_paper(session_factory, "p1")
        q = QueryCards(session_factory)
        assert await cache_check(q, "p1") is None
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_no_paper_id_returns_none(self, tmp_path):
        engine, session_factory = await _tmp_db(tmp_path)
        q = QueryCards(session_factory)
        assert await cache_check(q, None) is None
        await engine.dispose()


class TestSlotManager:
    @pytest.mark.asyncio
    async def test_write_read(self, tmp_path):
        engine, session_factory = await _tmp_db(tmp_path)
        mgr = SlotManager(session_factory)
        slots = CalcSlots(calculation_type=CalculationType.RELAX, encut_ev=400.0)
        await mgr.save_session_slots("s1", slots)
        got = await mgr.get_session_slots("s1")
        assert got is not None and got.encut_ev == 400.0
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_merge_priority(self, tmp_path):
        engine, session_factory = await _tmp_db(tmp_path)
        await _seed_paper(session_factory, "p1")
        q = QueryCards(session_factory)
        card = _make_card("p1", encut=520.0)
        await q.save_card(card)
        mgr = SlotManager(session_factory)
        await mgr.save_session_slots(
            "s1", CalcSlots(calculation_type=CalculationType.RELAX, encut_ev=400.0)
        )
        merged = await mgr.merge(
            user_slots=CalcSlots(calculation_type=CalculationType.RELAX, encut_ev=600.0),
            card=card, session_id="s1",
        )
        assert merged.encut_ev == 600.0  # user > card > session
        merged2 = await mgr.merge(user_slots=CalcSlots(), card=card, session_id="s1")
        assert merged2.encut_ev == 520.0  # card > session
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_ttl_expired(self, tmp_path):
        engine, session_factory = await _tmp_db(tmp_path)
        mgr = SlotManager(session_factory)
        await mgr.save_session_slots("s1", CalcSlots(encut_ev=400.0))
        async with session_factory() as s:
            row = await s.get(Session, "s1")
            row.expires_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
            await s.commit()
        assert await mgr.get_session_slots("s1") is None
        await engine.dispose()


# ============ 工具 ============


async def _tmp_db(tmp_path):
    db = tmp_path / "db" / "chem.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite+aiosqlite:///{str(db).replace(chr(92), '/')}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory


async def _seed_paper(session_factory, paper_id):
    async with session_factory() as s:
        s.add(Paper(
            paper_id=paper_id, filename=f"{paper_id}.pdf",
            file_path=f"data/papers/{paper_id}.pdf", file_hash=paper_id,
            file_size=1, status="saved", thread_id=f"lit:{paper_id}", meta_json="{}",
        ))
        await s.commit()


def _make_card(paper_id: str, encut: float = 520.0) -> CalculationCard:
    now = datetime.now(timezone.utc)
    return CalculationCard(
        card_id=uuid4(), paper_id=paper_id, system_formula="Co3O4",
        system_normalized="Co3O4",
        parameters=DFTParameters(
            calculation_type=CalculationType.RELAX, functional="PBE",
            functional_family=FunctionalFamily.PBE, hubbard_u=[],
            encut_ev=encut,
            kpoints=KPoints(scheme=KPointsScheme.GAMMA, mesh=(3, 3, 1), gamma_centered=True),
        ),
        missing_fields=[], validation=ValidationReport(),
        created_at=now, updated_at=now,
    )


def _make_app(tmp_path, fake_llm: FakeLLM):
    engine, session_factory = asyncio.run(_tmp_db(tmp_path))
    data_root = tmp_path / "data"
    file_store = FileStore(data_root)
    file_store.ensure_base_dirs()
    holder = SimpleNamespace(_model=fake_llm)
    chroma = ChromaStore(tmp_path / "chroma", settings.EMBEDDING_MODEL)
    chroma._available = False
    lit = LiteratureServices(
        pdf_parser=PDFParser(), llm_extractor=holder,
        rule_validator=RuleValidatorService(), query_cards=QueryCards(session_factory),
        vector_searcher=VectorSearcher(chroma),
        session_factory=session_factory, file_store=file_store,
    )
    calc = CalcServices(
        session_factory=session_factory, file_store=file_store,
        query_cards=lit.query_cards, llm_extractor=holder,
        mp_client=MPClient(api_key=settings.MP_API_KEY, cache_dir=tmp_path / "mp_cache"),
        surface_builder=SurfaceBuilder(), vasp_generator=VASPGenerator(),
        rule_validator=RuleValidatorService(),
    )
    app = create_app(
        lit, calc,
        db_path=str(tmp_path / "db" / "chem.db"),
        data_dir=str(data_root),
        chroma_dir=str(tmp_path / "chroma"),
        mp_api_key=settings.MP_API_KEY,
    )
    return app, engine, session_factory, file_store


# ============ action=message 端到端 ============


class TestActionMessage:
    def test_literature_cache_hit(self, tmp_path):
        app, engine, session_factory, _ = _make_app(tmp_path, FakeLLM(_SLOTS_JSON))
        asyncio.run(_seed_paper(session_factory, "p1"))
        asyncio.run(QueryCards(session_factory).save_card(_make_card("p1")))
        with TestClient(app) as client:
            r = client.post("/chat", json={
                "message": "提取这篇论文的参数", "paper_id": "p1", "action": "message",
            })
            data = r.json()["data"]
            assert data["type"] == "cards"
            assert data["source"] == "cache"

    def test_calc_segment_confirm_request(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path, FakeLLM(_SLOTS_JSON))
        with TestClient(app) as client:
            r = client.post("/chat", json={"message": "帮我生成 Co3O4 的 INCAR", "action": "message"})
            assert r.status_code == 200
            data = r.json()["data"]
            assert data["type"] == "confirm_request"
            assert data["draft"]["incar"].find("ENCUT") != -1

    def test_two_segments_results(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path, FakeLLM(_SLOTS_JSON))
        with TestClient(app) as client:
            r = client.post("/chat", json={
                "message": "提取论文参数，另外帮我生成 Co3O4 的 INCAR",
                "action": "message",
            })
            data = r.json()["data"]
            assert data["type"] == "results"
            assert len(data["results"]) == 2

    def test_one_segment_fails_other_succeeds(self, tmp_path):
        app, engine, session_factory, _ = _make_app(tmp_path, FakeLLM(_SLOTS_JSON))
        asyncio.run(_seed_paper(session_factory, "p1"))
        with TestClient(app) as client:
            # 进入 lifespan 后替换 find_by_paper 使其抛异常（文献段失败）
            query_cards = app.state.query_cards
            async def boom(paper_id):
                raise RuntimeError("db broken")
            query_cards.find_by_paper = boom
            r = client.post("/chat", json={
                "message": "提取这篇论文的参数，另外帮我生成 Co3O4 的 INCAR",
                "paper_id": "p1", "action": "message",
            })
            data = r.json()["data"]
            assert data["type"] == "results"
            assert data["results"][0]["type"] == "error"
            assert data["results"][1]["type"] == "confirm_request"

    def test_paused_session_message_4003(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path, FakeLLM(_SLOTS_JSON))
        with TestClient(app) as client:
            # 用 CIF 离线把 session 打进暂停态
            payload = {
                "slots": {
                    "calculation_type": "relax", "functional": "PBE",
                    "functional_family": "PBE", "encut_ev": 520.0,
                    "kpoints": {"scheme": "gamma", "mesh": [3, 3, 1], "gamma_centered": True},
                    "surface": {"miller": [1, 1, 1], "layers": 6, "vacuum_angstrom": 15.0},
                }
            }
            r = client.post(
                "/calc/draft",
                data={"payload": json.dumps(payload)},
                files={"structure_file": (_CIF.name, open(_CIF, "rb"), "chemical/x-cif")},
            )
            session_id = r.json()["data"]["session_id"]
            r2 = client.post("/chat", json={
                "session_id": session_id, "message": "换个参数", "action": "message",
            })
            assert r2.status_code == 409
            assert r2.json()["code"] == 4003

    def test_review_still_works(self, tmp_path):
        app, engine, session_factory, _ = _make_app(tmp_path, FakeLLM(_SLOTS_JSON))
        asyncio.run(_seed_paper(session_factory, "p1"))
        with TestClient(app) as client:
            r = client.post("/chat", json={
                "paper_id": "p1", "action": "review",
                "payload": {"decisions": [{"index": 0, "decision": "approve"}]},
            })
            # p1 不是 needs_review → 2006（S6 语义不变）
            assert r.status_code == 409
            assert r.json()["code"] == 2006


# ============ /health 与 CORS ============


class TestHealthAndCors:
    def test_health_ok(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path, FakeLLM(_SLOTS_JSON))
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            data = r.json()["data"]
            assert data["status"] in ("ok", "degraded")
            assert data["components"]["llm"] == "ok"
            assert data["components"]["mp"] == "ok"

    def test_health_down_when_sqlite_broken(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path, FakeLLM(_SLOTS_JSON))
        with TestClient(app) as client:
            class BrokenEngine:
                def connect(self):
                    raise RuntimeError("db down")
            app.state.engine = BrokenEngine()
            r = client.get("/health")
            assert r.status_code == 503
            assert r.json()["data"]["status"] == "down"

    def test_cors_preflight(self, tmp_path):
        app, _, _, _ = _make_app(tmp_path, FakeLLM(_SLOTS_JSON))
        with TestClient(app) as client:
            r = client.options(
                "/chat",
                headers={"Origin": "http://example.com",
                         "Access-Control-Request-Method": "POST"},
            )
            assert "access-control-allow-origin" in r.headers


# ============ 异常体系迁移 ============


class TestExceptions:
    def test_four_classes_importable(self):
        assert issubclass(ServiceError, Exception)
        for cls in (AgentExecutionError, PipelineError, IntentRouteError):
            e = cls(4002, "x", {"k": "v"})
            assert e.code == 4002 and e.message == "x" and e.details == {"k": "v"}

    def test_reexport_identity(self):
        from app.storage.files import ServiceError as FromFiles
        from app.core.exceptions import ServiceError as FromCore
        assert FromFiles is FromCore
