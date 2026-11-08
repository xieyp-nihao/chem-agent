"""参数卡片查询：cards 表的读写、过滤、聚合。

对应 docs/technical-design.md §4.9：
- save_card 只写 cards 表，不越权改 papers.status；不调 VectorSearcher/LLM
- filter 多条件 AND；system_normalized 精确匹配；functional 匹配 functional 列
  （展示名）或 functional_family 列
- aggregate 在 Python 层聚合（先取行再分组，避免 SQLite JSON1 复杂度）
- AggregateRow 定义在本文件（§4.9 定位；与 storage.chroma.VectorHit 同理）

序列化口径（与 S2 §6.2.2 一致）：
- JSON 列用 domain 模型的 model_dump_json()；missing_fields 单独存 JSON 数组
- datetime 存 naive UTC（SQLite 无时区类型）；读取时补 UTC tzinfo
"""
from __future__ import annotations

import json
from datetime import timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.schemas import (
    CalculationCard,
    DFTParameters,
    Evidence,
    ReviewRecord,
    SurfaceModel,
    ValidationReport,
)
from app.storage.db import Card as CardRow


def _naive_utc(dt) -> object:
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _aware_utc(dt) -> object:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _to_domain(row: CardRow) -> CalculationCard:
    return CalculationCard(
        card_id=row.card_id,
        paper_id=row.paper_id,
        system_formula=row.system_formula,
        system_normalized=row.system_normalized,
        parameters=DFTParameters.model_validate_json(row.parameters_json),
        surface=SurfaceModel.model_validate_json(row.surface_json) if row.surface_json else None,
        missing_fields=json.loads(row.missing_fields),
        validation=ValidationReport.model_validate_json(row.validation_json),
        provenance=[Evidence.model_validate(x) for x in json.loads(row.provenance_json)],
        review=ReviewRecord.model_validate_json(row.review_json) if row.review_json else None,
        review_status=row.review_status,
        schema_version=row.schema_version,
        rules_version=row.rules_version,
        created_at=_aware_utc(row.created_at),
        updated_at=_aware_utc(row.updated_at),
    )


class AggregateRow(BaseModel):
    """聚合结果（§4.9）。

    missing_count 口径：组内卡片 missing_fields 条目数之和。
    avg_u_ev 口径：组内所有卡片的 HubbardU.u_ev 取平均；组内无任何 U 值为 None。
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    card_count: int
    paper_count: int
    avg_encut_ev: float | None
    avg_u_ev: float | None
    missing_count: int


class QueryCards:
    """卡片查询服务（异步、无状态；session factory 注入，不建 engine）。"""

    def __init__(self, session_factory: async_sessionmaker):
        self._session_factory = session_factory

    # ---- 写 ----

    async def save_card(self, card: CalculationCard) -> None:
        """upsert 一张卡片；重审/修改后重新保存会把 chroma_indexed 置 0（触发 S6 重索引）。"""
        data = dict(
            card_id=str(card.card_id),
            paper_id=card.paper_id,
            system_formula=card.system_formula,
            system_normalized=card.system_normalized,
            calculation_type=card.parameters.calculation_type.value,
            functional=card.parameters.functional,
            functional_family=card.parameters.functional_family.value,
            parameters_json=card.parameters.model_dump_json(),
            surface_json=card.surface.model_dump_json() if card.surface else None,
            missing_fields=json.dumps(card.missing_fields),
            validation_json=card.validation.model_dump_json(),
            provenance_json=json.dumps(
                [e.model_dump(mode="json") for e in card.provenance]
            ),
            review_json=card.review.model_dump_json() if card.review else None,
            review_status=card.review_status,
            chroma_id=str(card.card_id),
            chroma_indexed=0,
            schema_version=card.schema_version,
            rules_version=card.rules_version,
            created_at=_naive_utc(card.created_at),
            updated_at=_naive_utc(card.updated_at),
        )
        async with self._session_factory() as s:
            row = await s.get(CardRow, str(card.card_id))
            if row is None:
                s.add(CardRow(**data))
            else:
                for key, value in data.items():
                    setattr(row, key, value)
                row.chroma_indexed = 0  # 重审/修改后需重新建立向量索引
            await s.commit()

    async def mark_chroma_indexed(self, card_id: str, indexed: bool) -> None:
        async with self._session_factory() as s:
            row = await s.get(CardRow, card_id)
            if row is not None:
                row.chroma_indexed = 1 if indexed else 0
                await s.commit()

    # ---- 读 ----

    async def get_by_id(self, card_id: str) -> CalculationCard | None:
        async with self._session_factory() as s:
            row = await s.get(CardRow, card_id)
            return _to_domain(row) if row else None

    async def find_by_paper(self, paper_id: str) -> list[CalculationCard]:
        """按 paper_id 查询，created_at 升序。"""
        async with self._session_factory() as s:
            rows = (
                await s.execute(
                    select(CardRow).where(CardRow.paper_id == paper_id).order_by(CardRow.created_at)
                )
            ).scalars().all()
            return [_to_domain(r) for r in rows]

    async def filter(
        self,
        *,
        system_normalized: str | None = None,
        functional: str | None = None,
        paper_id: str | None = None,
        card_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[CalculationCard]:
        """多条件 AND；functional 匹配 functional（展示名）或 functional_family。"""
        stmt = select(CardRow)
        if system_normalized is not None:
            stmt = stmt.where(CardRow.system_normalized == system_normalized)
        if functional is not None:
            stmt = stmt.where(
                or_(
                    CardRow.functional == functional,
                    CardRow.functional_family == functional,
                )
            )
        if paper_id is not None:
            stmt = stmt.where(CardRow.paper_id == paper_id)
        if card_id is not None:
            stmt = stmt.where(CardRow.card_id == card_id)
        stmt = stmt.order_by(CardRow.created_at).limit(limit).offset(offset)
        async with self._session_factory() as s:
            rows = (await s.execute(stmt)).scalars().all()
            return [_to_domain(r) for r in rows]

    async def aggregate(
        self, *, group_by: Literal["system", "functional", "paper"]
    ) -> list[AggregateRow]:
        """全表取行后 Python 层聚合（第一版数据量小）。"""
        async with self._session_factory() as s:
            rows = (await s.execute(select(CardRow))).scalars().all()
        cards = [_to_domain(r) for r in rows]
        groups: dict[str, list[CalculationCard]] = {}
        for c in cards:
            if group_by == "system":
                key = c.system_normalized
            elif group_by == "functional":
                key = c.parameters.functional_family.value
            else:
                key = c.paper_id
            groups.setdefault(key, []).append(c)
        result: list[AggregateRow] = []
        for key, group in groups.items():
            encuts = [c.parameters.encut_ev for c in group if c.parameters.encut_ev is not None]
            u_values = [u.u_ev for c in group for u in c.parameters.hubbard_u]
            result.append(AggregateRow(
                key=key,
                card_count=len(group),
                paper_count=len({c.paper_id for c in group}),
                avg_encut_ev=sum(encuts) / len(encuts) if encuts else None,
                avg_u_ev=sum(u_values) / len(u_values) if u_values else None,
                missing_count=sum(len(c.missing_fields) for c in group),
            ))
        result.sort(key=lambda r: r.key)
        return result
