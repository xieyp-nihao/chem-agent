"""意图拆分（§2.4 编排层约定）。

- 只识别四个连接词：另外 / 还有 / 同时 / 顺便
- 最多拆两段：取最早出现的连接词位置拆开，其余内容全部并入第二段
- 空段丢弃；每段保留原始文本（不重写、不摘要）
- 先拆分、再逐段路由（路由在 router.py）
"""
from __future__ import annotations

_SPLITTERS = ("另外", "还有", "同时", "顺便")


def split_intents(message: str) -> list[str]:
    """拆分意图；无连接词 → 单段。"""
    positions = [(message.find(w), w) for w in _SPLITTERS if message.find(w) >= 0]
    if not positions:
        return [message.strip()] if message.strip() else []
    pos, _word = min(positions)
    first = message[:pos].strip()
    rest = message[pos:].strip()  # 第二段保留原始文本（含连接词，不重写）
    segments = [s for s in (first, rest) if s]
    if not segments:
        return []
    return segments
