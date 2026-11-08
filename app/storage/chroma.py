"""ChromaDB 持久层：param_cards collection + 嵌入函数 + 降级。

对应 docs/technical-design.md §6.3 / §4.8：
- collection param_cards，cosine（hnsw:space=cosine），id=card_id，维度 384
- 显式接入 sentence-transformers（EMBEDDING_MODEL，cpu，normalize_embeddings）
- 模型不可用（下载失败/网络不通）时 is_available() 返回 False，调用方据此跳过语义
  检索；upsert/search/delete 抛 ServiceError(5002, VECTOR_STORE_UNAVAILABLE)。
- 本模块 import 与构造均不触发模型加载：chromadb 的 SentenceTransformerEmbeddingFunction
  在 __init__ 即时加载/下载模型，因此必须懒探测并缓存结果。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pydantic import BaseModel, ConfigDict

from app.domain.schemas import CalculationCard
from app.storage.files import ServiceError

COLLECTION_NAME = "param_cards"


class VectorHit(BaseModel):
    """一条向量检索命中（§4.8 引用；定义暂存此处，S3 迁共享位置）。"""

    model_config = ConfigDict(extra="forbid")

    card_id: str
    distance: float
    metadata: dict[str, str | int | float | bool]


def build_document(card: CalculationCard) -> str:
    """§4.8 固定嵌入模板。"""
    p = card.parameters
    u_parts = [f"{u.element} {u.orbital.value} {u.u_ev}" for u in p.hubbard_u]
    if p.kpoints is not None:
        kp = p.kpoints
        mesh = " ".join(str(n) for n in kp.mesh) if kp.mesh else ""
        kp_text = f"{kp.scheme.value} {mesh}".strip()
    else:
        kp_text = ""
    if card.surface is not None:
        s = card.surface
        miller = "".join(str(i) for i in s.miller)
        surface_text = f"({miller}) {s.layers} layers, vacuum {s.vacuum_angstrom} Å"
    else:
        surface_text = ""
    lines = [
        f"system: {card.system_normalized}",
        f"calculation: {p.calculation_type.value}",
        f"functional: {p.functional}",
        f"hubbard_u: {'; '.join(u_parts) if u_parts else ''}",
        f"encut: {p.encut_ev if p.encut_ev is not None else ''} eV",
        f"kpoints: {kp_text}",
        f"surface: {surface_text}",
        f"notes: {p.notes or ''}",
    ]
    return "\n".join(lines)


def build_metadata(card: CalculationCard) -> dict[str, str | int | float | bool]:
    """§6.3 metadata schema：仅标量（str/int/float/bool）。"""
    p = card.parameters
    return {
        "card_id": str(card.card_id),
        "paper_id": card.paper_id,
        "system_normalized": card.system_normalized,
        "calculation_type": p.calculation_type.value,
        "functional_family": p.functional_family.value,
        "has_surface": card.surface is not None,
        "review_status": card.review_status,
        "missing_count": len(card.missing_fields),
        "created_at": card.created_at.isoformat(),
    }


class ChromaStore:
    """懒探测可用性的 Chroma 持久层；不可用时抛 5002。"""

    def __init__(self, persist_dir: Path, embedding_model: str):
        self._persist_dir = Path(persist_dir)
        self._embedding_model = embedding_model
        self._client: chromadb.PersistentClient | None = None
        self._collection = None
        self._embedder = None
        self._available: bool | None = None
        self._error: str | None = None

    def _build_embedder(self):
        """测试 monkeypatch 点：chromadb 的 EF 在构造时即时加载/下载模型。"""
        return SentenceTransformerEmbeddingFunction(
            model_name=self._embedding_model,
            device="cpu",
            normalize_embeddings=True,
        )

    def _ensure_available(self) -> bool:
        if self._available is None:
            try:
                embedder = self._build_embedder()
                embedder(["ping"])  # 强制加载模型并验证可用
                client = chromadb.PersistentClient(
                    path=str(self._persist_dir),
                    settings=chromadb.Settings(anonymized_telemetry=False),
                )
                collection = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=embedder,
                    metadata={"hnsw:space": "cosine"},
                )
                self._embedder = embedder
                self._client = client
                self._collection = collection
                self._available = True
            except Exception as exc:  # 下载失败/网络不通等一律降级
                self._error = str(exc)[:200]
                self._available = False
        return self._available

    def is_available(self) -> bool:
        return self._ensure_available()

    def _require(self) -> None:
        if not self._ensure_available():
            raise ServiceError(5002, "VECTOR_STORE_UNAVAILABLE", {"reason": self._error})

    async def upsert(self, card: CalculationCard) -> None:
        self._require()
        document = build_document(card)
        metadata = build_metadata(card)
        await asyncio.to_thread(
            self._collection.upsert,
            ids=[str(card.card_id)],
            documents=[document],
            metadatas=[metadata],
        )

    async def search(
        self, query: str, top_k: int = 5, where: dict | None = None
    ) -> list[VectorHit]:
        self._require()
        result = await asyncio.to_thread(
            self._collection.query,
            query_texts=[query],
            n_results=top_k,
            where=where,
            include=["distances", "metadatas"],
        )
        hits: list[VectorHit] = []
        for card_id, distance, meta in zip(
            result["ids"][0], result["distances"][0], result["metadatas"][0]
        ):
            hits.append(VectorHit(card_id=card_id, distance=distance, metadata=meta or {}))
        return hits

    async def delete(self, card_id: str) -> None:
        self._require()
        await asyncio.to_thread(self._collection.delete, ids=[card_id])
