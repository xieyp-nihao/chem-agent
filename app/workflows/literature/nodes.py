"""文献图节点（§5.2.3）。

- 节点失败写 status + error，条件边进入 notify_* 后 END，不把异常抛到调用方
- human_review 单次执行只 interrupt 一次；多轮由条件边自环驱动（§5.2.2 L6→L6）
- edit 只允许合并 parameters / surface，不允许改 paper_id / system_formula 来源
- error 未清零不允许 approve；最多 3 轮，第 3 轮仍有 error → failed（2005）
- save_cards：一次计算一张卡；SQLite 卡片保存不因 Chroma 失败而阻断
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.types import interrupt

from app.domain.normalizer import normalize_formula
from app.domain.schemas import (
    CalculationCard,
    ExtractedCalculation,
    ReviewDecision,
    ReviewRecord,
    SurfaceModel,
    ValidationReport,
)
from app.services.pdf_parser import PDFParser, ParsedPaper
from app.storage.files import FileStore, ServiceError

logger = logging.getLogger(__name__)

MAX_REVIEW_ROUNDS = 3


@dataclass
class LiteratureServices:
    """文献图依赖（注入，无模块级状态）。"""

    pdf_parser: PDFParser
    llm_extractor: Any
    rule_validator: Any
    query_cards: Any
    vector_searcher: Any
    session_factory: Any
    file_store: FileStore


# ---- papers 状态更新（工作流层直接使用存储层；QueryCards 不越权） ----


async def _update_paper(
    services: LiteratureServices,
    paper_id: str,
    status: str,
    *,
    error_code: int | None = None,
    error_msg: str | None = None,
) -> None:
    from app.storage.db import Paper

    async with services.session_factory() as s:
        row = await s.get(Paper, paper_id)
        if row is None:
            logger.warning("paper %s 不存在，跳过状态更新", paper_id)
            return
        row.status = status
        if error_code is not None:
            row.error_code = error_code
        if error_msg is not None:
            row.error_msg = error_msg
        await s.commit()


def _error(code: int, message: str, details: dict | None = None) -> dict:
    return {"code": code, "message": message, "details": details or {}}


# ---- 节点 ----


async def parse_pdf(state: dict, services: LiteratureServices) -> dict:
    paper_id = state["paper_id"]
    await _update_paper(services, paper_id, "parsing")
    try:
        parsed = services.pdf_parser.parse(Path(state["pdf_path"]))
    except ServiceError as exc:
        return {
            "status": "failed",
            "error": _error(exc.code, exc.message, exc.details),
        }
    if parsed.is_scanned:
        reason = parsed.warnings[0] if parsed.warnings else "扫描版 PDF"
        return {
            "status": "failed",
            "error": _error(2002, "SCANNED_PDF_UNSUPPORTED", {"reason": reason}),
        }
    return {
        "parsed": parsed.model_dump(mode="json"),
        "method_found": parsed.method_found,
    }


async def extract_params(state: dict, services: LiteratureServices) -> dict:
    await _update_paper(services, state["paper_id"], "extracting")
    parsed = ParsedPaper.model_validate(state["parsed"])
    try:
        calcs = await services.llm_extractor.extract(
            paper_id=state["paper_id"],
            method_text=parsed.method_text,
            tables=parsed.tables,
            captions=parsed.captions,
        )
    except ServiceError as exc:
        return {
            "status": "failed",
            "error": _error(exc.code, exc.message, exc.details),
        }
    if not calcs:
        return {
            "status": "failed",
            "error": _error(2004, "EXTRACTION_FAILED", {"reason": "未抽取到任何计算"}),
        }
    return {"extracted": [c.model_dump(mode="json") for c in calcs]}


async def validate_params(state: dict, services: LiteratureServices) -> dict:
    await _update_paper(services, state["paper_id"], "validating")
    reports: list[dict] = []
    for item in state.get("extracted", []):
        calc = ExtractedCalculation.model_validate(item)
        report = services.rule_validator.validate(
            parameters=calc.parameters,
            surface=calc.surface,
            calculation_type=calc.parameters.calculation_type,
            structure=None,
        )
        reports.append(report.model_dump(mode="json"))
    return {"reports": reports}


# ---- human_review ----


def _build_review_request(state: dict, round_num: int) -> dict:
    """§5.2.5 的 review_request payload。"""
    calcs = [ExtractedCalculation.model_validate(x) for x in state.get("extracted", [])]
    reports = [ValidationReport.model_validate(r) for r in state.get("reports", [])]
    candidates = []
    for i, calc in enumerate(calcs):
        report = reports[i] if i < len(reports) else ValidationReport()
        candidates.append({
            "index": i,
            "system_formula": calc.system_formula,
            "system_normalized": normalize_formula(calc.system_formula)[0],
            "parameters": calc.parameters.model_dump(mode="json"),
            "surface": calc.surface.model_dump(mode="json") if calc.surface else None,
            "findings": [f.model_dump(mode="json") for f in report.findings],
            "missing_fields": report.missing_fields,
            "evidence": [e.model_dump(mode="json") for e in calc.evidence],
        })
    return {
        "type": "review_request",
        "paper_id": state["paper_id"],
        "thread_id": f"lit:{state['paper_id']}",
        "round": round_num,
        "candidates": candidates,
    }


def _apply_edit(calc: ExtractedCalculation, decision: dict) -> tuple[ExtractedCalculation, list[str]]:
    """edit 只合并 parameters / surface 字段；返回 (新计算, 被编辑字段列表)。

    用 model_dump 合并 + model_validate 重建（model_copy(update=...) 不做嵌套校验，
    hubbard_u 等嵌套 dict 不会被强转为模型）。
    """
    from app.domain.schemas import DFTParameters

    edited_fields: list[str] = []
    updates: dict[str, Any] = {}
    params_update = decision.get("parameters")
    if isinstance(params_update, dict):
        merged = calc.parameters.model_dump()
        merged.update(params_update)
        updates["parameters"] = DFTParameters.model_validate(merged)
        edited_fields.extend(f"parameters.{k}" for k in params_update)
    surface_update = decision.get("surface")
    if isinstance(surface_update, dict):
        if calc.surface is None:
            updates["surface"] = SurfaceModel.model_validate(surface_update)
        else:
            merged_surface = calc.surface.model_dump()
            merged_surface.update(surface_update)
            updates["surface"] = SurfaceModel.model_validate(merged_surface)
        edited_fields.extend(f"surface.{k}" for k in surface_update)
    return calc.model_copy(update=updates), edited_fields


async def human_review(state: dict, services: LiteratureServices) -> dict:
    paper_id = state["paper_id"]
    round_num = int(state.get("review_round", 0)) + 1
    await _update_paper(services, paper_id, "needs_review")

    # 暂停点：resume 后 interrupt() 返回客户端提交的 payload
    decision: dict = interrupt(_build_review_request(state, round_num))
    decisions = decision.get("decisions", []) if isinstance(decision, dict) else []

    calcs = [ExtractedCalculation.model_validate(x) for x in state.get("extracted", [])]
    reports = [ValidationReport.model_validate(r) for r in state.get("reports", [])]
    messages: list[str] = list(state.get("messages", []))
    approved: list[dict] = list(state.get("approved", []))
    rejected: list[dict] = list(state.get("rejected", []))

    decided: dict[int, str] = {}  # index -> approve/edit/reject（本轮有效决定）
    edited_fields_by_idx: dict[int, list[str]] = {}
    comment_by_idx: dict[int, str] = {}
    # 跨轮已决索引（方案 A）：客户端按 payload 全量决策时，重复决策被跳过
    prior_decided = {
        e.get("index")
        for e in state.get("approved", []) + state.get("rejected", [])
        if isinstance(e, dict) and e.get("index") is not None
    }

    for d in decisions:
        idx = d.get("index")
        choice = d.get("decision")
        if not isinstance(idx, int) or idx < 0 or idx >= len(calcs) or idx in decided:
            messages.append(f"忽略无效 decision：{d}")
            continue
        if idx in prior_decided:
            logger.warning("忽略重复决策：index=%s 已决", idx)
            messages.append(f"忽略重复决策：候选 {idx} 在之前的轮次已决")
            continue
        if choice == "approve":
            if reports[idx].errors:
                messages.append(
                    f"候选 {idx} 存在 error，approve 被拒（只能 edit 或 reject），需重新决定"
                )
                continue  # 不记入 decided → 保持 pending
            decided[idx] = "approve"
        elif choice == "edit":
            calcs[idx], edited = _apply_edit(calcs[idx], d)
            new_report = services.rule_validator.validate(
                parameters=calcs[idx].parameters,
                surface=calcs[idx].surface,
                calculation_type=calcs[idx].parameters.calculation_type,
            )
            reports[idx] = new_report
            if new_report.errors:
                messages.append(f"候选 {idx} edit 后仍有 error，需继续修正")
                continue  # 保持 pending，下一轮 payload 显示编辑后的状态
            decided[idx] = "edit"
            edited_fields_by_idx[idx] = edited
            comment_by_idx[idx] = d.get("comment") or ""
        elif choice == "reject":
            decided[idx] = "reject"
        else:
            messages.append(f"未知 decision：{choice}")

    # 结算本轮
    for idx, choice in decided.items():
        if choice in ("approve", "edit"):
            approved.append({
                "index": idx,
                "calculation": calcs[idx].model_dump(mode="json"),
                "decision": choice,
                "edited_fields": edited_fields_by_idx.get(idx, []),
                "comment": comment_by_idx.get(idx, ""),
            })
        else:
            rejected.append({"index": idx, "system_formula": calcs[idx].system_formula})

    # pending = 未决候选：排除本轮 decided ∪ 跨轮已决（否则跳过的重复决策会永久 pending）
    settled = prior_decided | set(decided.keys())
    pending = [i for i in range(len(calcs)) if i not in settled]
    if not pending and not approved:
        status = "rejected"
        error = None
    elif pending and round_num < MAX_REVIEW_ROUNDS:
        status = "needs_review"  # 自环：再次 interrupt
        error = None
    elif pending:
        status = "failed"
        error = _error(
            2005, "REVIEW_REJECTED",
            {"reason": f"第 {round_num} 轮审核后仍存在未解决的 error 候选（上限 {MAX_REVIEW_ROUNDS} 轮）"},
        )
    else:
        status = "approved"
        error = None

    return {
        "extracted": [c.model_dump(mode="json") for c in calcs],
        "reports": [r.model_dump(mode="json") for r in reports],
        "review_round": round_num,
        "review_payload": decision,
        "approved": approved,
        "rejected": rejected,
        "status": status,
        "error": error,
        "messages": messages,
    }


# ---- save_cards ----


async def save_cards(state: dict, services: LiteratureServices) -> dict:
    saved: list[str] = []
    messages: list[str] = list(state.get("messages", []))
    # 无审核路径（validate_params 直通）：approved 为空，直接保存 extracted（auto_saved）
    entries = state.get("approved") or [
        {"calculation": x, "decision": "approve", "edited_fields": [], "comment": ""}
        for x in state.get("extracted", [])
    ]
    for entry in entries:
        calc = ExtractedCalculation.model_validate(entry["calculation"])
        # 重跑校验取 report（approved 与 reports 不再按索引对齐，避免错位）
        report = services.rule_validator.validate(
            parameters=calc.parameters,
            surface=calc.surface,
            calculation_type=calc.parameters.calculation_type,
        )
        normalized, _w = normalize_formula(calc.system_formula)
        now = datetime.now(timezone.utc)
        reviewed = int(state.get("review_round", 0)) > 0
        card = CalculationCard(
            card_id=uuid4(),
            paper_id=state["paper_id"],
            system_formula=calc.system_formula,
            system_normalized=normalized,
            parameters=calc.parameters,
            surface=calc.surface,
            missing_fields=report.missing_fields,
            validation=report,
            provenance=calc.evidence,
            review=(
                ReviewRecord(
                    reviewer=None,
                    decision=(
                        ReviewDecision.EDIT
                        if entry.get("decision") == "edit"
                        else ReviewDecision.APPROVE
                    ),
                    edited_fields=entry.get("edited_fields", []),
                    comment=entry.get("comment") or None,
                    reviewed_at=now,
                    round=int(state.get("review_round", 0)),
                )
                if reviewed
                else None
            ),
            review_status="reviewed" if reviewed else "auto_saved",
            created_at=now,
            updated_at=now,
        )
        await services.query_cards.save_card(card)
        try:
            await services.vector_searcher.upsert(card)
            await services.query_cards.mark_chroma_indexed(str(card.card_id), True)
        except ServiceError as exc:
            # Chroma 失败不阻断 SQLite 卡片保存（§6.3）
            await services.query_cards.mark_chroma_indexed(str(card.card_id), False)
            messages.append(f"卡片 {card.card_id} 的 Chroma 索引失败（{exc.message}），SQLite 已保存")
        saved.append(str(card.card_id))
    await _update_paper(services, state["paper_id"], "saved")
    return {"saved_card_ids": saved, "messages": messages}


# ---- notify 节点 ----


async def notify_missing(state: dict, services: LiteratureServices) -> dict:
    await _update_paper(services, state["paper_id"], "method_not_found")
    return {}


async def notify_failure(state: dict, services: LiteratureServices) -> dict:
    error = state.get("error") or _error(5005, "INTERNAL_ERROR")
    await _update_paper(
        services,
        state["paper_id"],
        "failed",
        error_code=error.get("code"),
        error_msg=error.get("message"),
    )
    return {}


async def notify_rejected(state: dict, services: LiteratureServices) -> dict:
    await _update_paper(services, state["paper_id"], "rejected")
    return {}
