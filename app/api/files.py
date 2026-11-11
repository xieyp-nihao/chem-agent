"""文件下载 API（§7.7）。

- path 相对 data/outputs/（如 calc_6b1d.../INCAR）
- 路径安全内联实现（S7 禁改 storage 层；FileStore 严格 API 不支持多段子目录路径）：
  resolve() 后必须位于 outputs 根之下，越界 403/404
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.api.documents import error_response

router = APIRouter()


@router.get("/files/{path:path}")
async def get_file(request: Request, path: str):
    root = request.app.state.file_store.root / "outputs"
    root_resolved = root.resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root_resolved):
        return error_response(request, 6002, "PATH_FORBIDDEN")
    if not candidate.is_file():
        return error_response(request, 6001, "FILE_NOT_FOUND")
    return FileResponse(candidate)
