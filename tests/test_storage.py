"""S2 存储层测试。

覆盖（docs/technical-design.md §8.2 S2 退出条件）：
- papers/cards/sessions 建表 + PRAGMA（WAL / foreign_keys / busy_timeout）连接级生效
- checkpoints/writes 表由 AsyncSqliteSaver.setup() 创建
- 三表 CRUD、cards 按 paper_id / system_normalized 查询、FK 级联删除
- 文件名 sanitize 与路径穿越防线（含符号链接/junction 逃逸）
- Chroma 降级路径（monkeypatch 模型失败）与真实 upsert/search（模型未下载时 skip）

skip 政策：仅两处允许 skip——符号链接（系统无创建权限时）与 Chroma 真实用例
（嵌入模型未下载/网络不可用），报告需说明原因。
"""
import functools
import os
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.schemas import (
    CalculationCard,
    CalculationType,
    DFTParameters,
    FunctionalFamily,
    KPoints,
    KPointsScheme,
    SurfaceModel,
    ValidationReport,
)
from app.storage.chroma import ChromaStore
from app.storage.db import Base, Card, Paper, Session, create_engine, utcnow
from app.storage.files import FileStore, ServiceError, sanitize_filename

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ============ fixtures ============


@pytest_asyncio.fixture
async def engine(tmp_path):
    """每个测试独立的临时 db 文件，不污染 data/chem-agent.db。"""
    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    eng = create_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


def _paper(paper_id: str = "p1", status: str = "queued") -> Paper:
    return Paper(
        paper_id=paper_id,
        filename="a.pdf",
        file_path=f"data/papers/{paper_id}.pdf",
        file_hash=paper_id,
        file_size=123,
        status=status,
        thread_id=f"lit:{paper_id}",
        meta_json="{}",
    )


def _card(
    paper_id: str = "p1",
    system_normalized: str = "Co3O4",
    functional: str = "PBE",
    family: str = "PBE",
    calc_type: str = "relax",
) -> Card:
    card_id = str(uuid4())
    return Card(
        card_id=card_id,
        paper_id=paper_id,
        system_formula=system_normalized,
        system_normalized=system_normalized,
        calculation_type=calc_type,
        functional=functional,
        functional_family=family,
        parameters_json="{}",
        surface_json=None,
        missing_fields="[]",
        validation_json="{}",
        provenance_json="[]",
        review_json=None,
        review_status="auto_saved",
        chroma_id=card_id,
        chroma_indexed=0,
        schema_version="1.0",
        rules_version="1.0",
    )


# ============ 建表与 PRAGMA ============


class TestTables:
    @pytest.mark.asyncio
    async def test_business_tables_created(self, engine, tmp_path):
        con = sqlite3.connect(tmp_path / "test.db")
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        con.close()
        assert {"papers", "cards", "sessions"} <= tables

    @pytest.mark.asyncio
    async def test_pragma_per_connection(self, engine):
        async with engine.connect() as conn:
            fk = (await conn.execute(text("PRAGMA foreign_keys"))).scalar()
            busy_timeout = (await conn.execute(text("PRAGMA busy_timeout"))).scalar()
        assert fk == 1
        assert busy_timeout == 5000

    @pytest.mark.asyncio
    async def test_journal_mode_wal(self, engine):
        async with engine.connect() as conn:
            mode = (await conn.execute(text("PRAGMA journal_mode"))).scalar()
        assert mode == "wal"

    @pytest.mark.asyncio
    async def test_checkpoint_tables_created(self, tmp_path):
        """init_db.py 同款流程在 tmp db 上复验：checkpoints/writes 由包创建。"""
        db_path = str(tmp_path / "cp.db")
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            await saver.setup()
        con = sqlite3.connect(db_path)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        con.close()
        assert {"checkpoints", "writes"} <= tables


# ============ CRUD ============


class TestPaperCrud:
    @pytest.mark.asyncio
    async def test_insert_update_delete(self, session_factory):
        async with session_factory() as s:
            s.add(_paper())
            await s.commit()

            row = (await s.execute(select(Paper).where(Paper.paper_id == "p1"))).scalar_one()
            assert row.status == "queued"
            assert row.file_hash == "p1"
            assert row.created_at is not None and row.updated_at is not None

            row.status = "saved"
            row.error_msg = "ok"
            await s.commit()
            row = (await s.execute(select(Paper).where(Paper.paper_id == "p1"))).scalar_one()
            assert row.status == "saved"

            await s.delete(row)
            await s.commit()
            assert (
                await s.execute(select(Paper).where(Paper.paper_id == "p1"))
            ).scalar_one_or_none() is None


class TestCardCrud:
    @pytest.mark.asyncio
    async def test_query_by_paper_and_system_normalized(self, session_factory):
        async with session_factory() as s:
            s.add(_paper("p1"))
            s.add(_card(paper_id="p1", system_normalized="Co3O4"))
            s.add(_card(paper_id="p1", system_normalized="Fe2O3"))
            await s.commit()

            by_paper = (await s.execute(select(Card).where(Card.paper_id == "p1"))).scalars().all()
            assert len(by_paper) == 2

            by_system = (
                await s.execute(select(Card).where(Card.system_normalized == "Co3O4"))
            ).scalars().all()
            assert len(by_system) == 1
            assert by_system[0].system_normalized == "Co3O4"
            assert by_system[0].chroma_id == by_system[0].card_id

    @pytest.mark.asyncio
    async def test_cascade_delete_on_paper(self, session_factory):
        async with session_factory() as s:
            s.add(_paper("p1"))
            await s.commit()
            s.add(_card(paper_id="p1"))
            await s.commit()
        async with session_factory() as s:
            paper = (await s.execute(select(Paper).where(Paper.paper_id == "p1"))).scalar_one()
            await s.delete(paper)
            await s.commit()
            remaining = (await s.execute(select(Card))).scalars().all()
            assert remaining == []


class TestSessionCrud:
    @pytest.mark.asyncio
    async def test_insert_update_query(self, session_factory):
        async with session_factory() as s:
            # expires_at 设为 1 小时后，用于"1 天内过期"的索引查询
            row = Session(
                session_id="s1", slots_json="{}", expires_at=utcnow() + timedelta(hours=1)
            )
            s.add(row)
            await s.commit()

            got = (await s.execute(select(Session).where(Session.session_id == "s1"))).scalar_one()
            assert got.slots_json == "{}"

            got.active_workflow = "calc"
            got.slots_json = '{"calculation_type": "relax"}'
            await s.commit()

            got = (await s.execute(select(Session).where(Session.session_id == "s1"))).scalar_one()
            assert got.active_workflow == "calc"
            assert got.slots_json == '{"calculation_type": "relax"}'

            expiring = (
                await s.execute(
                    select(Session).where(Session.expires_at < utcnow() + timedelta(days=1))
                )
            ).scalars().all()
            assert len(expiring) == 1


# ============ 文件名 sanitize ============


class TestSanitize:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("a/b/../c.pdf", "c.pdf"),
            ("..\\..\\etc\\passwd", "passwd"),
            ("/etc/passwd", "passwd"),
            ("C:\\Windows\\system32", "system32"),
            (".hidden", "hidden"),
        ],
    )
    def test_sanitize(self, name, expected):
        assert sanitize_filename(name) == expected

    def test_sanitize_max_length(self):
        long_name = "a" * 200 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) == 128
        assert result.endswith(".pdf")

    def test_sanitize_empty_raises(self):
        with pytest.raises(ServiceError) as exc:
            sanitize_filename("...")
        assert exc.value.code == 6002


# ============ 路径穿越防线 ============


class TestPathTraversal:
    @pytest.fixture
    def store(self, tmp_path):
        fs = FileStore(tmp_path)
        fs.ensure_base_dirs()
        return fs

    @pytest.mark.parametrize(
        "name",
        ["../../etc/passwd", "..\\..\\etc\\passwd", "/etc/passwd", "C:\\Windows\\system32"],
    )
    def test_traversal_rejected_on_save(self, store, name):
        with pytest.raises(ServiceError) as exc:
            store.save("papers", name, b"x")
        assert exc.value.code == 6002

    @pytest.mark.parametrize(
        "name",
        ["../../etc/passwd", "..\\..\\etc\\passwd", "/etc/passwd", "C:\\Windows\\system32"],
    )
    def test_traversal_rejected_on_read(self, store, name):
        with pytest.raises(ServiceError) as exc:
            store.read("papers", name)
        assert exc.value.code == 6002

    def test_bad_subroot_rejected(self, store):
        with pytest.raises(ServiceError) as exc:
            store.save("etc", "a.txt", b"x")
        assert exc.value.code == 6002

    def test_save_read_delete_roundtrip(self, store):
        path = store.save("papers", "a.pdf", b"data")
        assert path.name == "a.pdf"
        assert store.read("papers", "a.pdf") == b"data"
        assert store.delete("papers", "a.pdf") is True
        assert store.delete("papers", "a.pdf") is False
        with pytest.raises(ServiceError) as exc:
            store.read("papers", "a.pdf")
        assert exc.value.code == 6001

    def _make_link(self, papers_dir, target):
        """创建指向根目录的符号链接；Windows 无权限时回退 mklink /J junction。"""
        link = papers_dir / "link"
        try:
            os.symlink(str(target), str(link), target_is_directory=True)
            return link
        except OSError as symlink_exc:
            if os.name != "nt":
                raise
            try:
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                    check=True,
                    capture_output=True,
                )
                return link
            except (OSError, subprocess.CalledProcessError) as junction_exc:
                pytest.skip(
                    f"无法创建符号链接/junction（os.symlink: {symlink_exc}; "
                    f"mklink /J: {junction_exc}），跳过符号链接逃逸用例"
                )

    def test_symlink_escape_rejected(self, store, tmp_path):
        # 链接指向根目录（papers 子根之外）→ resolve 后越界 → 6002
        self._make_link(store.root / "papers", tmp_path)
        with pytest.raises(ServiceError) as exc:
            store.save("papers", "link", b"x")
        assert exc.value.code == 6002


# ============ Chroma ============


def _make_card() -> CalculationCard:
    return CalculationCard(
        card_id=uuid4(),
        paper_id="p1",
        system_formula="Co3O4",
        system_normalized="Co3O4",
        parameters=DFTParameters(
            calculation_type=CalculationType.RELAX,
            functional="PBE+U",
            functional_family=FunctionalFamily.PBE_U,
            encut_ev=520.0,
            kpoints=KPoints(scheme=KPointsScheme.GAMMA, mesh=(3, 3, 1), gamma_centered=True),
        ),
        surface=SurfaceModel(miller=(1, 1, 1), layers=6, vacuum_angstrom=15.0),
        missing_fields=[],
        validation=ValidationReport(),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@functools.lru_cache(maxsize=1)
def _chroma_available() -> bool:
    """探测嵌入模型是否本地可用（已下载且能编码 384 维）。"""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        embedder = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL, device="cpu", normalize_embeddings=True
        )
        vector = embedder(["ping"])
        return len(vector[0]) == 384
    except Exception:
        return False


class TestChromaDegradation:
    @pytest.mark.asyncio
    async def test_constructor_does_not_touch_network(self, tmp_path, monkeypatch):
        store = ChromaStore(tmp_path / "chroma", EMBEDDING_MODEL)

        def broken():
            raise AssertionError("构造阶段不得加载模型")

        monkeypatch.setattr(store, "_build_embedder", broken)
        assert store._available is None  # 未探测

    @pytest.mark.asyncio
    async def test_unavailable_when_model_fails(self, tmp_path, monkeypatch):
        store = ChromaStore(tmp_path / "chroma", EMBEDDING_MODEL)

        def broken():
            raise RuntimeError("model download failed")

        monkeypatch.setattr(store, "_build_embedder", broken)
        assert store.is_available() is False  # 不抛异常
        assert store.is_available() is False  # 结果缓存，不重复尝试

        with pytest.raises(ServiceError) as exc:
            await store.upsert(_make_card())
        assert exc.value.code == 5002

        with pytest.raises(ServiceError) as exc:
            await store.search("Co3O4")
        assert exc.value.code == 5002


@pytest.mark.skipif(
    not _chroma_available(),
    reason="embedding model not downloaded / huggingface network unavailable",
)
class TestChromaReal:
    @pytest.mark.asyncio
    async def test_upsert_and_search(self, tmp_path):
        store = ChromaStore(tmp_path / "chroma", EMBEDDING_MODEL)
        assert store.is_available() is True

        dimension = len(store._embedder(["Co3O4"])[0])
        assert dimension == 384

        card = _make_card()
        await store.upsert(card)

        hits = await store.search("Co3O4 (111) surface calculation", top_k=1)
        assert len(hits) == 1
        assert hits[0].card_id == str(card.card_id)
        meta = hits[0].metadata
        assert meta["system_normalized"] == "Co3O4"
        assert meta["functional_family"] == "PBE+U"
        assert meta["has_surface"] is True
        assert meta["missing_count"] == 0
