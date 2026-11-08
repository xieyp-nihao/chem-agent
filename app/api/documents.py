"""文献文档 API（§7.3 / §7.4 / §7.5）。

- envelope 与错误映射定义在本文件，chat.py 复用导入
- 上传：multipart、PDF 校验（扩展名 + %PDF- 魔数 + MIME + 大小）、SHA-256 paper_id、
  重复上传返回 200、202 + BackgroundTasks 跑文献图（不同步等）
- 状态查询 / retry（仅 failed / method_not_found 可 retry，不删已保存卡片）
"""
from __future__ import annotations

import hashlib
import logging
from datetime import timezone
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.storage.db import Paper

logger = logging.getLogger(__name__)

router = APIRouter()

_HTTP_STATUS = {
    1000: 400, 1001: 415, 1002: 400, 1003: 413,
    2001: 404, 2002: 422, 2003: 422, 2004: 422, 2005: 200, 2006: 409,
    3001: 422, 3002: 422, 3003: 404, 3004: 503, 3005: 422, 3006: 422,
    3007: 409, 3008: 500,
    4002: 409, 4003: 409, 4004: 409,
    5001: 503, 5002: 503, 5003: 500, 5004: 503, 5005: 500,
    6001: 404, 6002: 403,
}


def _request_id(request: Request) -> str:
    rid = request.headers.get("X-Request-ID")
    return rid or str(uuid4())


def make_envelope(code: int, message: str, data, request_id: str) -> dict:
    return {"code": code, "message": message, "data": data, "request_id": request_id}


def ok_response(request: Request, data, *, message: str = "ok", status: int = 200) -> JSONResponse:
    rid = _request_id(request)
    return JSONResponse(
        status_code=status,
        content=make_envelope(0, message, data, rid),
        headers={"X-Request-ID": rid},
    )


def error_response(request: Request, code: int, message: str, data=None) -> JSONResponse:
    rid = _request_id(request)
    return JSONResponse(
        status_code=_HTTP_STATUS.get(code, 500),
        content=make_envelope(code, message, data, rid),
        headers={"X-Request-ID": rid},
    )


# ---- 后台任务 ----


async def _run_graph(app, paper_id: str, pdf_path: str) -> None:
    graph = app.state.literature_graph
    try:
        await graph.ainvoke(
            {"paper_id": paper_id, "pdf_path": pdf_path},
            config={"configurable": {"thread_id": f"lit:{paper_id}"}},
        )
        # 返回值含 __interrupt__ 即正常暂停（needs_review 已由节点落库），无需处理
    except Exception:  # 图内异常兜底：落 failed，不抛出
        logger.exception("literature graph unexpected failure: %s", paper_id)
        from app.workflows.literature.nodes import _update_paper  # 复用状态更新

        async with app.state.session_factory() as s:
            row = await s.get(Paper, paper_id)
            if row is not None:
                row.status = "failed"
                row.error_code = 5005
                row.error_msg = "INTERNAL_ERROR"
                await s.commit()


# ---- 路由 ----


@router.post("/documents/upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    data = await file.read()
    filename = file.filename or ""
    content_type = file.content_type or ""
    if not filename.lower().endswith(".pdf") or not data.startswith(b"%PDF-"):
        return error_response(request, 1002, "INVALID_FILE", {"reason": "非 PDF 文件"})
    if content_type and content_type not in ("application/pdf", "application/octet-stream", "application/x-pdf"):
        return error_response(request, 1002, "INVALID_FILE", {"reason": f"MIME 类型不符: {content_type}"})
    if len(data) > settings.MAX_PDF_MB * 1024 * 1024:
        return error_response(request, 1003, "PAYLOAD_TOO_LARGE")

    paper_id = hashlib.sha256(data).hexdigest()
    store = request.app.state.file_store
    try:
        store.save("papers", f"{paper_id}.pdf", data)
    except Exception as exc:
        return error_response(request, 6002, "PATH_FORBIDDEN", {"reason": str(exc)[:200]})

    session_factory = request.app.state.session_factory
    async with session_factory() as s:
        existing = await s.get(Paper, paper_id)
        if existing is not None:
            return ok_response(
                request,
                {"type": "document_status", "paper_id": paper_id, "status": existing.status,
                 "status_url": f"/documents/{paper_id}"},
                message="already exists",
            )
        s.add(Paper(
            paper_id=paper_id,
            filename=filename,
            file_path=str(store.root / "papers" / f"{paper_id}.pdf"),
            file_hash=paper_id,
            file_size=len(data),
            status="queued",
            thread_id=f"lit:{paper_id}",
            meta_json="{}",
        ))
        await s.commit()

    pdf_path = str(store.root / "papers" / f"{paper_id}.pdf")
    background_tasks.add_task(_run_graph, request.app, paper_id, pdf_path)
    return ok_response(
        request,
        {"type": "document_status", "paper_id": paper_id, "status": "queued",
         "status_url": f"/documents/{paper_id}"},
        message="accepted",
        status=202,
    )


def _iso(dt) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat() if dt.tzinfo is None else dt.isoformat()


async def _document_status(request: Request, paper_id: str):
    cards = await request.app.state.query_cards.find_by_paper(paper_id)
    return {
        "type": "document_status",
        "paper_id": paper_id,
        "status": None,  # 由调用方填充（占位避免重复查询）
        "method_found": None,
        "page_count": None,
        "error_msg": None,
        "error_code": None,
        "card_ids": [str(c.card_id) for c in cards],
        "pending_action": None,
        "thread_id": f"lit:{paper_id}",
        "created_at": None,
        "updated_at": None,
    }


@router.get("/documents/{paper_id}")
async def get_document(request: Request, paper_id: str):
    session_factory = request.app.state.session_factory
    async with session_factory() as s:
        row = await s.get(Paper, paper_id)
        if row is None:
            return error_response(request, 2001, "PAPER_NOT_FOUND")
        data = await _document_status(request, paper_id)
        data.update({
            "status": row.status,
            "method_found": row.method_found,
            "page_count": row.page_count,
            "error_msg": row.error_msg,
            "error_code": row.error_code,
            "pending_action": "review" if row.status == "needs_review" else None,
            "created_at": _iso(row.created_at),
            "updated_at": _iso(row.updated_at),
        })
        return ok_response(request, data)


@router.post("/documents/{paper_id}/retry")
async def retry_document(request: Request, paper_id: str, background_tasks: BackgroundTasks):
    session_factory = request.app.state.session_factory
    async with session_factory() as s:
        row = await s.get(Paper, paper_id)
        if row is None:
            return error_response(request, 2001, "PAPER_NOT_FOUND")
        if row.status not in ("failed", "method_not_found"):
            return error_response(
                request, 2006, "DOCUMENT_STATE_CONFLICT",
                {"reason": f"status={row.status} 不允许 retry（仅 failed/method_not_found）"},
            )
        row.status = "queued"
        row.error_code = None
        row.error_msg = None
        await s.commit()
        pdf_path = row.file_path
    background_tasks.add_task(_run_graph, request.app, paper_id, pdf_path)
    return ok_response(
        request,
        {"type": "document_status", "paper_id": paper_id, "status": "queued",
         "status_url": f"/documents/{paper_id}"},
        message="accepted",
        status=202,
    )
