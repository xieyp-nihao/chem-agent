"""文献精确缓存检查（§2.4 编排层约定）。

- 只对"文献 + 有 paper_id"生效：QueryCards.find_by_paper 命中直接返回
  cards(source=cache)，不跑图、不查 Chroma
- 语义检索（VectorSearcher）是另一条路径，source="semantic" 区分，不冒充精确缓存
"""
from __future__ import annotations


async def cache_check(query_cards, paper_id: str | None) -> dict | None:
    if not paper_id:
        return None
    cards = await query_cards.find_by_paper(paper_id)
    if not cards:
        return None
    return {
        "type": "cards",
        "source": "cache",
        "cards": [
            {
                "card_id": str(c.card_id),
                "paper_id": c.paper_id,
                "system_normalized": c.system_normalized,
                "calculation_type": c.parameters.calculation_type.value,
                "functional": c.parameters.functional,
                "missing_fields": c.missing_fields,
                "review_status": c.review_status,
            }
            for c in cards
        ],
    }
