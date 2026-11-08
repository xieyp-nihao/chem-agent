"""计算草案 API（§7.6）：POST /calc/draft（JSON + multipart，同一张计算图）。

- 暂停检测：graph.aget_state 的 tasks 非空 → 409 + 4003 WORKFLOW_STATE_CONFLICT
- multipart 的 structure_file 存 data/structures/uploads/{session_id}/（FileStore 严格
  API 不支持子目录，此处 API 层内联防护：文件名 sanitize + resolve 越界检查）
- 结果映射：__interrupt__ → confirm_request；3001 → 422 clarify.missing_slots；
  3002 → 422 clarify.suggestions；其余 failed 按 §7.9
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from app.api.documents import error_response, make_envelope, ok_response, _request_id
from app.domain.schemas import CalcSlots
from app.storage.files import sanitize_filename

logger = logging.getLogger(__name__)

router = APIRouter()


class DraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    card_id: str | None = None
    slots: dict = {}


def _confirm_data(result: dict) -> dict:
    payload = result["__interrupt__"][0].value
    return {
        "type": "confirm_request",
        "session_id": payload.get("session_id"),
        "thread_id": payload.get("thread_id"),
        "round": payload.get("round"),
        "draft": payload.get("draft"),
        "validation": payload.get("validation"),
    }


def _map_error(request: Request, result: dict):
    error = result.get("error") or {}
    code = error.get("code", 5005)
    if code == 3001:
        return error_response(
            request, 3001, "缺少必填槽位",
            {"type": "clarify", "missing_slots": error.get("details", {}).get("missing_slots", [])},
        )
    if code == 3002:
        return error_response(
            request, 3002, error.get("message", "第一版不支持"),
            {"type": "clarify", "suggestions": error.get("details", {}).get("suggestions", [])},
        )
    return error_response(request, code, error.get("message", "INTERNAL_ERROR"))


async def _invoke_calc(request: Request, session_id: str, state: dict):
    graph = request.app.state.calc_graph
    config = {"configurable": {"thread_id": f"calc:{session_id}"}}
    # 暂停检测（§5.4：同一 session 已有暂停图 → WORKFLOW_STATE_CONFLICT）
    current = await graph.aget_state(config)
    if current.tasks:
        return None, error_response(
            request, 4003, "WORKFLOW_STATE_CONFLICT",
            {"reason": "该 session 已有暂停的确认图，请先 confirm/abort"},
        )
    result = await graph.ainvoke(state, config=config)
    return result, None


def _save_structure_upload(request: Request, session_id: str, upload: UploadFile) -> str:
    """落盘用户上传的结构文件；返回存储相对名（structures 子根下）。"""
    safe_name = sanitize_filename(upload.filename or "structure.cif")
    root = request.app.state.file_store.root / "structures"
    upload_dir = (root / "uploads" / session_id).resolve()
    if not upload_dir.is_relative_to(root.resolve()):
        raise ValueError("upload 目录越界")
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = (upload_dir / safe_name).resolve()
    if not target.is_relative_to(upload_dir):
        raise ValueError("upload 文件越界")
    target.write_bytes(upload.file.read())
    # FileStore.read 只接受单段文件名：扁平化存储名为 {session_id}__{safe_name}
    flat_name = f"{session_id}__{safe_name}"
    (root / flat_name).write_bytes(target.read_bytes())
    return flat_name


@router.post("/calc/draft")
async def create_draft(request: Request):
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    session_id: str | None = None
    card_id: str | None = None
    slots: dict = {}
    cif_name: str | None = None

    if content_type == "application/json":
        body = await request.json()
        try:
            req = DraftRequest.model_validate(body)
        except Exception:
            return error_response(request, 1000, "INVALID_REQUEST")
        session_id = req.session_id
        card_id = req.card_id
        slots = req.slots
    elif content_type == "multipart/form-data":
        form = await request.form()
        payload_raw = form.get("payload")
        try:
            payload = json.loads(payload_raw) if isinstance(payload_raw, str) else {}
            req = DraftRequest.model_validate(payload)
        except Exception:
            return error_response(request, 1000, "INVALID_REQUEST")
        session_id = req.session_id
        card_id = req.card_id
        slots = req.slots
        upload = form.get("structure_file")
        if upload is not None and hasattr(upload, "filename"):
            try:
                cif_name = _save_structure_upload(request, session_id or "anon", upload)
            except ValueError as exc:
                return error_response(request, 6002, "PATH_FORBIDDEN", {"reason": str(exc)})
    else:
        return error_response(request, 1001, "UNSUPPORTED_CONTENT_TYPE",
                              {"reason": "仅支持 application/json 或 multipart/form-data"})

    session_id = session_id or str(uuid4())
    try:
        slots_model = CalcSlots.model_validate(slots)
    except Exception:
        return error_response(request, 1000, "INVALID_REQUEST", {"reason": "slots 校验失败"})
    if cif_name:
        slots_model = slots_model.model_copy(
            update={"cif_path": cif_name, "structure_source": "cif_upload"}
        )

    result, conflict = await _invoke_calc(request, session_id, {
        "session_id": session_id,
        "thread_id": f"calc:{session_id}",
        "raw_message": None,
        "user_slots": slots_model.model_dump(mode="json"),
        "card_id": card_id,
    })
    if conflict is not None:
        return conflict
    if "__interrupt__" in result:
        return ok_response(request, _confirm_data(result))
    if result.get("status") in ("unsupported", "missing_slots", "failed"):
        return _map_error(request, result)
    if result.get("output_files"):
        return ok_response(request, {"type": "output_files",
                                     "session_id": session_id,
                                     "files": result["output_files"]})
    # 理论上到不了这里（无 interrupt 且无输出 = notify 路径）
    rid = _request_id(request)
    return JSONResponse(
        status_code=500,
        content=make_envelope(5005, "INTERNAL_ERROR", None, rid),
        headers={"X-Request-ID": rid},
    )
