"""计算图 State（§5.3.1）。

约定：
- thread_id = calc:{session_id}，只用于 checkpoint（不用作目录名）
- 只放 JSON 可序列化数据：CalcSlots/VASPDraft/ValidationReport 用 model_dump(mode="json")；
  pymatgen Structure 用 as_dict() 存，节点内 Structure.from_dict() 重建
- output_files / messages 用 operator.add reducer，其余覆盖
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class CalcState(TypedDict, total=False):
    session_id: str
    thread_id: str
    raw_message: str | None

    user_slots: dict                 # 当前请求显式槽位（原始，审计用）
    card_id: str | None
    card_snapshot: dict | None       # CalculationCard.model_dump(...)
    slots: dict                      # 合并后的 CalcSlots.model_dump(...)

    capability: dict | None          # {supported, reason, suggestions}

    structure_dict: dict | None      # Structure.as_dict()
    structure_meta: dict | None      # StructureRecord 的可序列化部分
    slab_dict: dict | None           # SlabResult.structure.as_dict()
    slab_meta: dict | None           # SlabResult 的可序列化部分

    draft: dict | None               # VASPDraft.model_dump(...)
    validation: dict | None          # ValidationReport.model_dump(...)
    confirm_round: int
    confirm_payload: dict | None     # 最近一次 interrupt 的 resume 值
    patch_history: list[dict]

    output_files: Annotated[list[dict], operator.add]

    status: str
    error: dict | None               # {code, message, details}
    messages: Annotated[list[str], operator.add]
