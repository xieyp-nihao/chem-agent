"""向量检索 Service：封装 ChromaDB 查询（§4.8）。

S2 的 app/storage/chroma.ChromaStore 已实现 §6.3/§4.8 的全部语义：
- 显式接入 sentence-transformers（paraphrase-multilingual-MiniLM-L12-v2，
  cpu、normalize_embeddings=True）、param_cards collection、cosine、id=card_id
- §4.8 document 模板、§6.3 标量 metadata
- is_available() 降级（嵌入模型加载失败返回 False，不抛异常）
- 不可用时 upsert/search/delete 抛 ServiceError(5002, VECTOR_STORE_UNAVAILABLE)

本 Service 是 ChromaStore 的组合薄包装（依赖注入），不重复实现嵌入/降级逻辑；
VectorHit 沿用 storage.chroma 的既有定义（card_id/distance/metadata）。
"""
from __future__ import annotations

from app.domain.schemas import CalculationCard
from app.storage.chroma import ChromaStore, VectorHit


class VectorSearcher:
    """语义检索（异步、无状态；store 注入）。"""

    def __init__(self, store: ChromaStore):
        self._store = store

    def is_available(self) -> bool:
        """嵌入模型可用性（懒探测并缓存；加载失败返回 False，不抛异常）。"""
        return self._store.is_available()

    async def upsert(self, card: CalculationCard) -> None:
        await self._store.upsert(card)

    async def search(
        self, query: str, top_k: int = 5, where: dict | None = None
    ) -> list[VectorHit]:
        return await self._store.search(query, top_k=top_k, where=where)

    async def delete(self, card_id: str) -> None:
        await self._store.delete(card_id)
