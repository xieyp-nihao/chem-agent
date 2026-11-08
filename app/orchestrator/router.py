"""规则路由（§2.4 编排层约定）。

- 关键词子串匹配、大小写不敏感
- 只命中一侧 → 该意图；两侧都命中或都不命中 → fallback（交给 llm_fallback）
- 优先级：带 paper_id → 文献；带 card_id 或显式 slots → 计算
"""
from __future__ import annotations

_LITERATURE_KEYWORDS = ("论文", "文献", "pdf", "提取", "方法段", "paper", "literature", "extract")
_CALC_KEYWORDS = ("生成", "计算", "incar", "poscar", "kpoints", "弛豫", "静态",
                  "slab", "vasp", "relax", "static")


def rule_route(
    message: str,
    *,
    paper_id: str | None = None,
    card_id: str | None = None,
    has_slots: bool = False,
) -> str:
    """返回 "literature" | "calc" | "fallback"。

    paper_id / card_id 优先级只用于消歧：段同时命中两侧（或都不命中）时生效；
    单侧明确命中的段不受影响（"另外帮我生成 INCAR" 在带 paper_id 的整体请求中
    仍走计算）。
    """
    lowered = message.lower()
    hit_lit = any(k in lowered for k in _LITERATURE_KEYWORDS)
    hit_calc = any(k in lowered for k in _CALC_KEYWORDS)
    if hit_lit and hit_calc:
        if paper_id:
            return "literature"
        if card_id or has_slots:
            return "calc"
        return "fallback"
    if hit_lit:
        return "literature"
    if hit_calc:
        return "calc"
    if paper_id:
        return "literature"
    if card_id or has_slots:
        return "calc"
    return "fallback"
