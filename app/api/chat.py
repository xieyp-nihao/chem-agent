"""聊天入口（§7.2）。

S6 只实现 action=review（文献审核恢复）；action=message 由 S8 实现、
action=confirm 由 S7 实现，当前返回 HTTP 501 占位。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict

from app.api.documents import error_response, ok_response, make_envelope, _request_id
from app.core.exceptions import ServiceError
from app.storage.db import Paper

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    message: str | None = None
    paper_id: str | None = None
    card_id: str | None = None
    action: str = "message"
    payload: dict | None = None


def _card_brief(card) -> dict:
    return {
        "card_id": str(card.card_id),
        "paper_id": card.paper_id,
        "system_normalized": card.system_normalized,
        "calculation_type": card.parameters.calculation_type.value,
        "functional": card.parameters.functional,
        "missing_fields": card.missing_fields,
        "review_status": card.review_status,
    }


async def _handle_message(request: Request, body: ChatRequest):
    """action=message（S8）：intent_split → 逐段路由 → 文献/计算；两段串行互不阻断。"""
    from uuid import uuid4

    from app.orchestrator.cache import cache_check
    from app.orchestrator.intent_split import split_intents
    from app.orchestrator.llm_fallback import llm_classify
    from app.orchestrator.router import rule_route
    from app.orchestrator.slot_manager import SlotManager

    if not body.message:
        return error_response(request, 1000, "INVALID_REQUEST",
                              {"reason": "action=message 需要 message"})
    session_id = body.session_id or str(uuid4())

    # 同一 session 已有暂停的计算图 → 4003（文献审核按 paper 状态走，不在此检查）
    calc_graph = request.app.state.calc_graph
    calc_state = await calc_graph.aget_state(
        {"configurable": {"thread_id": f"calc:{session_id}"}}
    )
    if calc_state.tasks:
        return error_response(
            request, 4003, "WORKFLOW_STATE_CONFLICT",
            {"reason": "该 session 已有暂停的确认图，请先 confirm/abort"},
        )

    slot_manager = SlotManager(request.app.state.session_factory)
    segments = split_intents(body.message)
    results = []
    for seg in segments:
        try:
            results.append(await _route_segment(
                request, seg, body, session_id, slot_manager,
                cache_check, rule_route, llm_classify,
            ))
        except ServiceError as exc:
            results.append({"type": "error", "code": exc.code, "message": exc.message})
        except Exception:
            logger.exception("segment routing failed: %r", seg[:80])
            results.append({"type": "error", "code": 5005, "message": "INTERNAL_ERROR"})

    if len(results) == 1:
        return ok_response(request, results[0])
    # 两段都失败 → code 非 0；否则 code 0（results 内各自带成功/错误对象）
    all_failed = all(r.get("type") == "error" for r in results)
    rid = _request_id(request)
    if all_failed:
        return JSONResponse(
            status_code=200,
            content=make_envelope(5005, "两段请求均失败", {"type": "results", "results": results}, rid),
            headers={"X-Request-ID": rid},
        )
    return ok_response(request, {"type": "results", "results": results})


async def _route_segment(
    request: Request, seg: str, body: ChatRequest, session_id: str,
    slot_manager, cache_check, rule_route, llm_classify,
):
    from app.domain.schemas import CalcSlots
    from app.storage.db import Paper

    intent = rule_route(
        seg, paper_id=body.paper_id, card_id=body.card_id,
        has_slots=bool(body.payload),
    )
    if intent == "fallback":
        # 复用共享客户端（thinking disabled，app.state 注入，测试可替身）
        intent = await llm_classify(request.app.state.llm_model, seg)

    if intent == "literature":
        if body.paper_id:
            cached = await cache_check(request.app.state.query_cards, body.paper_id)
            if cached is not None:
                return cached
            session_factory = request.app.state.session_factory
            async with session_factory() as s:
                row = await s.get(Paper, body.paper_id)
            if row is None:
                return {"type": "clarify", "message": "paper_id 不存在，请先上传 PDF"}
            return {
                "type": "document_status",
                "paper_id": body.paper_id,
                "status": row.status,
                "thread_id": row.thread_id,
                "pending_action": "review" if row.status == "needs_review" else None,
            }
        # 无 paper_id：语义检索（与精确缓存区分 source）
        searcher = request.app.state.vector_searcher
        if searcher.is_available():
            hits = await searcher.search(seg, top_k=5)
            if hits:
                return {
                    "type": "cards",
                    "source": "semantic",
                    "cards": [{"card_id": h.card_id, "score": h.distance} for h in hits],
                }
        return {"type": "clarify", "message": "请提供 paper_id 或先上传论文 PDF"}

    if intent == "calc":
        user_slots = CalcSlots()
        if isinstance(body.payload, dict) and body.payload.get("slots"):
            try:
                user_slots = CalcSlots.model_validate(body.payload["slots"])
            except Exception:
                user_slots = CalcSlots()
        card = None
        if body.card_id:
            card = await request.app.state.query_cards.get_by_id(body.card_id)
        merged = await slot_manager.merge(
            user_slots=user_slots, card=card, session_id=session_id
        )
        result = await request.app.state.calc_graph.ainvoke(
            {
                "session_id": session_id,
                "thread_id": f"calc:{session_id}",
                "raw_message": seg,
                "user_slots": merged.model_dump(mode="json"),
                "card_id": None,  # 已合并，避免 collect_slots 二次叠加
            },
            config={"configurable": {"thread_id": f"calc:{session_id}"}},
        )
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            return {
                "type": "confirm_request",
                "session_id": payload.get("session_id"),
                "thread_id": payload.get("thread_id"),
                "round": payload.get("round"),
                "draft": payload.get("draft"),
                "validation": payload.get("validation"),
            }
        if result.get("status") in ("unsupported", "missing_slots", "failed"):
            error = result.get("error") or {}
            return {"type": "error", "code": error.get("code", 5005),
                    "message": error.get("message", "INTERNAL_ERROR")}
        return {"type": "clarify", "message": "计算请求未产生草案"}

    return {"type": "clarify", "message": "无法判断意图，请补充说明"}


async def _handle_confirm(request: Request, body: ChatRequest):
    """action=confirm（§7.2 的 confirm 部分，S7）：恢复计算图的确认暂停。"""
    if not body.session_id or not body.payload:
        return error_response(
            request, 1000, "INVALID_REQUEST",
            {"reason": "action=confirm 需要 session_id 与非空 payload"},
        )
    graph = request.app.state.calc_graph
    config = {"configurable": {"thread_id": f"calc:{body.session_id}"}}
    current = await graph.aget_state(config)
    if not current.tasks:
        return error_response(
            request, 4002, "GRAPH_RESUME_FAILED",
            {"reason": "thread 不存在或已完成，无法恢复确认"},
        )
    result = await graph.ainvoke(Command(resume=body.payload), config=config)
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return ok_response(
            request,
            {"type": "confirm_request",
             "session_id": payload.get("session_id"),
             "thread_id": payload.get("thread_id"),
             "round": payload.get("round"),
             "draft": payload.get("draft"),
             "validation": payload.get("validation")},
        )
    if result.get("status") == "aborted":
        return ok_response(
            request,
            {"type": "ack", "message": "已放弃该草案，未写输出文件",
             "session_id": body.session_id},
        )
    if result.get("status") == "failed":
        error = result.get("error") or {}
        return error_response(
            request, error.get("code", 5005), error.get("message", "INTERNAL_ERROR"),
        )
    if result.get("output_files"):
        return ok_response(
            request,
            {"type": "output_files", "session_id": body.session_id,
             "files": result["output_files"]},
        )
    return error_response(request, 5005, "INTERNAL_ERROR", {"reason": "确认后无输出"})


@router.post("/chat")
async def chat(request: Request, body: ChatRequest):
    if body.action == "message":
        return await _handle_message(request, body)
    if body.action == "confirm":
        return await _handle_confirm(request, body)
    if body.action != "review":
        return error_response(request, 1000, "INVALID_REQUEST", {"reason": f"未知 action: {body.action}"})

    if not body.paper_id or not body.payload or not body.payload.get("decisions"):
        return error_response(
            request, 1000, "INVALID_REQUEST",
            {"reason": "action=review 需要 paper_id 与非空 payload.decisions"},
        )

    session_factory = request.app.state.session_factory
    async with session_factory() as s:
        row = await s.get(Paper, body.paper_id)
        if row is None:
            return error_response(request, 2001, "PAPER_NOT_FOUND")
        if row.status != "needs_review":
            return error_response(
                request, 2006, "DOCUMENT_STATE_CONFLICT",
                {"reason": f"status={row.status}，当前不允许 review"},
            )

    result = await request.app.state.literature_graph.ainvoke(
        Command(resume=body.payload),
        config={"configurable": {"thread_id": f"lit:{body.paper_id}"}},
    )
    if "__interrupt__" in result:
        # 仍有 error / 无效 approve：再次暂停，返回新一轮 review_request
        payload = result["__interrupt__"][0].value
        return ok_response(
            request,
            {"type": "review_request", "paper_id": body.paper_id,
             "thread_id": f"lit:{body.paper_id}",
             "round": payload.get("round"), "candidates": payload.get("candidates")},
        )
    cards = await request.app.state.query_cards.find_by_paper(body.paper_id)
    return ok_response(
        request,
        {"type": "cards", "source": "extracted",
         "cards": [_card_brief(c) for c in cards]},
    )
