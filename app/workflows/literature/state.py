"""文献图 State（§5.2.1）。

约定：State 只放 JSON 可序列化数据——ParsedPaper / ExtractedCalculation /
ValidationReport / CalculationCard 一律 model_dump(mode="json") 后存入，
节点内部重建模型。saved_card_ids 与 messages 使用 operator.add reducer，
其余字段后写覆盖。
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class LiteratureState(TypedDict, total=False):
    paper_id: str
    pdf_path: str

    parsed: dict | None          # ParsedPaper.model_dump(mode="json")
    method_found: bool
    extracted: list[dict]        # list[ExtractedCalculation.model_dump(...)]
    reports: list[dict]          # list[ValidationReport.model_dump(...)]，与 extracted 按索引对应
    review_round: int
    review_payload: dict | None  # 最近一次 interrupt 的 resume 值
    approved: list[dict]         # 审核通过的计算（含 decision 元信息，见 nodes.human_review）
    rejected: list[dict]

    saved_card_ids: Annotated[list[str], operator.add]

    status: str                  # 与 PaperStatus 对齐
    error: dict | None           # {code, message, details}
    messages: Annotated[list[str], operator.add]
