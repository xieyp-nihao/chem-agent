"""文件系统存储：路径安全、读写、清理。

对应 docs/technical-design.md §6.4：
- sanitize：只保留 [A-Za-z0-9._-]，长度 ≤128，禁止以 . 开头
- 写入/读取前 resolve() 必须位于允许子根（papers/structures/outputs）之下
- 越界抛 ServiceError(code=6002, message="PATH_FORBIDDEN")

ServiceError 为最小占位定义（S3 阶段迁移到 app.core.exceptions）。
"""
from __future__ import annotations

import re
from pathlib import Path

# ServiceError 已迁移至 app.core.exceptions（S8）；此处 re-export 保持旧 import 路径兼容
from app.core.exceptions import ServiceError  # noqa: F401


_ALLOWED_SUBROOTS = ("papers", "structures", "outputs")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_MAX_NAME_LEN = 128


def sanitize_filename(name: str) -> str:
    """用户文件名归一化：取最后一段、只保留安全字符、截断至 128、剥离开头点。

    "a/b/../c.pdf" → "c.pdf"；结果为空抛 ServiceError(6002)。
    """
    last = name.replace("\\", "/").rsplit("/", 1)[-1]
    safe = _UNSAFE_CHARS.sub("", last)
    if len(safe) > _MAX_NAME_LEN:
        safe = safe[-_MAX_NAME_LEN:]
    safe = safe.lstrip(".")
    if not safe:
        raise ServiceError(6002, "PATH_FORBIDDEN", {"reason": "sanitized filename is empty"})
    return safe


class FileStore:
    """限定在 root/papers、root/structures、root/outputs 之下的文件读写。

    契约：调用方先 sanitize_filename 再 save/read/delete；本类内部只做防线，
    不做归一化（归一化会吞掉穿越输入）。
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    # ---- 目录骨架 ----

    def ensure_base_dirs(self) -> None:
        """创建 §6.4 目录骨架（init_db 复用）。"""
        dirs = [
            self.root / "papers",
            self.root / "structures" / "mp",
            self.root / "structures" / "uploads",
            self.root / "outputs",
            self.root / "chroma",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ---- 防线 ----

    def resolve_within(self, subroot: str, filename: str) -> Path:
        """双防线解析：① 严格名检查 ② subroot 白名单 + resolve 后必须位于子根之下。"""
        if subroot not in _ALLOWED_SUBROOTS:
            raise ServiceError(
                6002, "PATH_FORBIDDEN", {"reason": f"subroot not allowed: {subroot}"}
            )
        # 防线 ①：含分隔符/绝对路径/穿越的输入在此被拒
        if filename != sanitize_filename(filename):
            raise ServiceError(6002, "PATH_FORBIDDEN", {"reason": f"unsafe filename: {filename!r}"})
        # 防线 ②：resolve() 跟随符号链接/junction，逃逸即拒
        base = (self.root / subroot).resolve()
        candidate = (self.root / subroot / filename).resolve()
        if not candidate.is_relative_to(base):
            raise ServiceError(6002, "PATH_FORBIDDEN", {"reason": f"path escapes root: {filename!r}"})
        return candidate

    # ---- 读写/清理 ----

    def save(self, subroot: str, filename: str, data: bytes) -> Path:
        path = self.resolve_within(subroot, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def read(self, subroot: str, filename: str) -> bytes:
        path = self.resolve_within(subroot, filename)
        if not path.is_file():
            raise ServiceError(6001, "FILE_NOT_FOUND", {"path": str(path)})
        return path.read_bytes()

    def delete(self, subroot: str, filename: str) -> bool:
        """删除文件；返回是否确实删除了文件。"""
        path = self.resolve_within(subroot, filename)
        if path.exists():
            path.unlink()
            return True
        return False
