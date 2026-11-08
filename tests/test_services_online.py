"""S5 在线/离线混合测试：MPClient（真实 MP 调用）、QueryCards、VectorSearcher。

skip 政策：MP 网络用例在 MP 不可用时 runtime skip（原因写入 skip 消息）；
QueryCards / VectorSearcher 降级路径不依赖网络，不允许 skip；
VectorSearcher 真实用例在模型不可用时走降级分支（5002 断言），不 skip。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from pymatgen.core import Structure
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import settings
from app.domain.schemas import (
    CalculationCard,
    CalculationType,
    DFTParameters,
    FunctionalFamily,
    HubbardU,
    KPoints,
    KPointsScheme,
    Orbital,
    SurfaceModel,
    ValidationReport,
)
from app.services.mp_client import MPClient
from app.services.query_cards import QueryCards
from app.services.vector_searcher import VectorSearcher
from app.storage.chroma import ChromaStore
from app.storage.db import Base, create_engine
from app.storage.files import ServiceError

FIXTURES = Path(__file__).parent / "fixtures"


# ============ fixtures ============


@pytest_asyncio.fixture
async def engine(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    eng = create_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture(scope="module")
def mp_cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("mp_cache")


def _make_card(
    *,
    paper_id: str = "p1",
    system_normalized: str = "Co3O4",
    family: FunctionalFamily = FunctionalFamily.PBE_U,
    hubbard_u: list[HubbardU] | None = None,
    with_surface: bool = True,
    missing_fields: list[str] | None = None,
) -> CalculationCard:
    if hubbard_u is None:
        hubbard_u = (
            [HubbardU(element="Co", orbital=Orbital.D, u_ev=3.5)]
            if family is FunctionalFamily.PBE_U
            else []
        )
    now = datetime.now(timezone.utc)
    return CalculationCard(
        card_id=uuid4(),
        paper_id=paper_id,
        system_formula=system_normalized,
        system_normalized=system_normalized,
        parameters=DFTParameters(
            calculation_type=CalculationType.RELAX,
            functional="PBE+U" if family is FunctionalFamily.PBE_U else "PBE",
            functional_family=family,
            hubbard_u=hubbard_u,
            encut_ev=520.0,
            kpoints=KPoints(scheme=KPointsScheme.GAMMA, mesh=(3, 3, 1), gamma_centered=True),
        ),
        surface=(
            SurfaceModel(miller=(1, 1, 1), layers=6, vacuum_angstrom=15.0)
            if with_surface
            else None
        ),
        missing_fields=missing_fields or [],
        validation=ValidationReport(),
        created_at=now,
        updated_at=now,
    )


# ============ MPClient ============


class TestMPClient:
    @pytest.mark.asyncio
    async def test_get_bulk_by_mp_id(self, mp_cache_dir):
        client = MPClient(api_key=settings.MP_API_KEY, cache_dir=mp_cache_dir)
        try:
            record = await client.get_bulk(mp_id="mp-18748")
        except ServiceError as exc:
            if exc.code == 3004:
                pytest.skip(f"MP 不可用，跳过真实调用：{exc.details}")
            raise
        assert record.source == "mp_id"
        assert record.formula_normalized == "Co3O4"
        # mp-18748 为原胞（14 原子），fixtures 的 CIF 为常规胞（56 原子），1:4 关系
        cif = Structure.from_file(FIXTURES / "structures" / "co3o4_bulk.cif")
        assert len(record.structure) * 4 == len(cif), (
            f"mp-18748 原胞 {len(record.structure)} 原子 × 4 ≠ CIF 常规胞 {len(cif)} 原子"
        )

    @pytest.mark.asyncio
    async def test_get_bulk_by_formula(self, mp_cache_dir):
        client = MPClient(api_key=settings.MP_API_KEY, cache_dir=mp_cache_dir)
        try:
            record = await client.get_bulk(formula="Co3O4")
        except ServiceError as exc:
            if exc.code == 3004:
                pytest.skip(f"MP 不可用，跳过真实调用：{exc.details}")
            raise
        assert record.source == "formula"
        assert record.formula_normalized == "Co3O4"
        assert hasattr(record, "energy_above_hull")  # 字段存在；本环境 legacy API 恒为 None
        assert record.num_sites > 0

    @pytest.mark.asyncio
    async def test_cache_hit_no_network(self, mp_cache_dir):
        client = MPClient(api_key=settings.MP_API_KEY, cache_dir=mp_cache_dir)
        try:
            await client.get_bulk(mp_id="mp-18748")
        except ServiceError as exc:
            if exc.code == 3004:
                pytest.skip(f"MP 不可用，跳过真实调用：{exc.details}")
            raise
        client._api_key = "invalid-key-after-cache"  # 第二次必须走缓存，不触网
        record = await client.get_bulk(mp_id="mp-18748")
        assert record.formula_normalized == "Co3O4"
        assert any("缓存命中" in w for w in record.warnings)

    @pytest.mark.asyncio
    async def test_invalid_key_raises_3004(self, tmp_path):
        # 独立空缓存目录：共享 mp_cache_dir 里已有 mp-18748 缓存，会绕过网络
        client = MPClient(api_key="invalid-key-xxxx", cache_dir=tmp_path / "empty_cache")
        with pytest.raises(ServiceError) as exc:
            await client.get_bulk(mp_id="mp-18748")
        assert exc.value.code == 3004

    def test_missing_key_raises_at_construction(self, mp_cache_dir):
        with pytest.raises(ServiceError) as exc:
            MPClient(api_key="", cache_dir=mp_cache_dir)
        assert exc.value.code == 3004
        assert exc.value.message == "MP_API_KEY 未配置"


# ============ QueryCards ============


async def _seed_paper(session_factory, paper_id: str) -> None:
    """cards.paper_id 有 FK → papers，保存卡片前先建 paper 行。"""
    from app.storage.db import Paper

    async with session_factory() as s:
        s.add(Paper(
            paper_id=paper_id, filename=f"{paper_id}.pdf",
            file_path=f"data/papers/{paper_id}.pdf", file_hash=paper_id,
            file_size=1, status="saved", thread_id=f"lit:{paper_id}", meta_json="{}",
        ))
        await s.commit()


class TestQueryCards:
    @pytest.mark.asyncio
    async def test_crud_and_filter(self, session_factory):
        await _seed_paper(session_factory, "p1")
        q = QueryCards(session_factory)
        card = _make_card(paper_id="p1", system_normalized="Co3O4")
        await q.save_card(card)

        got = await q.get_by_id(str(card.card_id))
        assert got is not None
        assert got.system_normalized == "Co3O4"
        assert got.parameters.encut_ev == 520.0

        by_paper = await q.find_by_paper("p1")
        assert len(by_paper) == 1

        by_sys = await q.filter(system_normalized="Co3O4")
        assert len(by_sys) == 1

        # functional 匹配 functional 列（展示名）或 functional_family 列
        by_family = await q.filter(functional="PBE+U")
        assert len(by_family) == 1
        by_miss = await q.filter(system_normalized="Fe2O3")
        assert by_miss == []

        await q.mark_chroma_indexed(str(card.card_id), True)
        got2 = await q.get_by_id(str(card.card_id))
        assert got2 is not None

    @pytest.mark.asyncio
    async def test_aggregate_by_system(self, session_factory):
        await _seed_paper(session_factory, "p1")
        await _seed_paper(session_factory, "p2")
        q = QueryCards(session_factory)
        # 一张带 U（PBE+U）、一张不带 U（PBE），同体系
        await q.save_card(_make_card(paper_id="p1", family=FunctionalFamily.PBE_U))
        await q.save_card(
            _make_card(paper_id="p2", family=FunctionalFamily.PBE, missing_fields=["encut_ev"])
        )

        rows = await q.aggregate(group_by="system")
        assert len(rows) == 1
        row = rows[0]
        assert row.key == "Co3O4"
        assert row.card_count == 2
        assert row.paper_count == 2
        assert row.avg_u_ev == pytest.approx(3.5)  # 只平均有 U 的卡片
        assert row.avg_encut_ev == pytest.approx(520.0)
        assert row.missing_count == 1

    @pytest.mark.asyncio
    async def test_resave_resets_chroma_indexed(self, session_factory):
        await _seed_paper(session_factory, "p1")
        q = QueryCards(session_factory)
        card = _make_card(paper_id="p1")
        await q.save_card(card)
        await q.mark_chroma_indexed(str(card.card_id), True)
        await q.save_card(card)  # 重审后重新保存
        async with session_factory() as s:
            from app.storage.db import Card as CardRow

            row = await s.get(CardRow, str(card.card_id))
            assert row.chroma_indexed == 0


# ============ VectorSearcher ============


def _make_store(tmp_path) -> ChromaStore:
    return ChromaStore(tmp_path / "chroma", settings.EMBEDDING_MODEL)


class TestVectorSearcher:
    @pytest.mark.asyncio
    async def test_upsert_and_search_real(self, tmp_path):
        store = _make_store(tmp_path)
        searcher = VectorSearcher(store)
        if not searcher.is_available():
            # 模型不可用 → 走降级分支断言（不 skip）
            with pytest.raises(ServiceError) as exc:
                await searcher.upsert(_make_card())
            assert exc.value.code == 5002
            return
        dimension = len(store._embedder(["Co3O4"])[0])
        assert dimension == 384

        card = _make_card(paper_id="p1", system_normalized="Co3O4")
        await searcher.upsert(card)
        hits = await searcher.search("Co3O4 表面 CO2 吸附", top_k=1)
        assert len(hits) >= 1
        assert hits[0].card_id == str(card.card_id)
        meta = hits[0].metadata
        assert meta["system_normalized"] == "Co3O4"
        assert meta["functional_family"] == "PBE+U"
        assert meta["has_surface"] is True
        assert meta["missing_count"] == 0

    @pytest.mark.asyncio
    async def test_degraded_store_raises_5002(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)

        def broken():
            raise RuntimeError("model download failed")

        monkeypatch.setattr(store, "_build_embedder", broken)
        searcher = VectorSearcher(store)
        assert searcher.is_available() is False  # 不抛异常
        with pytest.raises(ServiceError) as exc:
            await searcher.search("Co3O4")
        assert exc.value.code == 5002
        with pytest.raises(ServiceError) as exc:
            await searcher.upsert(_make_card())
        assert exc.value.code == 5002

    @pytest.mark.asyncio
    async def test_querycards_chroma_integration(self, session_factory, tmp_path):
        await _seed_paper(session_factory, "p1")
        q = QueryCards(session_factory)
        card = _make_card(paper_id="p1")
        await q.save_card(card)

        store = _make_store(tmp_path)
        searcher = VectorSearcher(store)
        if searcher.is_available():
            await searcher.upsert(card)
            await q.mark_chroma_indexed(str(card.card_id), True)
            hits = await searcher.search("Co3O4 slab relax", top_k=1)
            assert len(hits) >= 1
            assert hits[0].card_id == str(card.card_id)
        else:
            with pytest.raises(ServiceError) as exc:
                await searcher.upsert(card)
            assert exc.value.code == 5002
            await q.mark_chroma_indexed(str(card.card_id), False)  # 可正常标记
