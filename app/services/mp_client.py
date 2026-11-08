"""Materials Project 客户端：按 mp_id / 化学式获取体相结构。

对应 docs/technical-design.md §4.5：
- 只返回 bulk Structure，不生成 slab、不写 cards/sessions
- 缓存 data/structures/mp/{mp_id}.cif，TTL 30 天
- 同步 MPRester 调用放 asyncio.to_thread，超时 30s，失败重试 2 次（1s/2s 退避）
- MP_API_KEY 缺失：构造时即抛 3004（不静默回退）
- 请求失败 → 3004 MP_UNAVAILABLE（提示可上传 CIF 继续）；材料不存在 → 3003

环境事实（2025.4.16 实测，见 S5 计划）：
- 使用 legacy pymatgen.ext.matproj.MPRester（mp-api 未安装；legacy 接受现有 MP_API_KEY）
- legacy 的 get_entries / get_entry_by_material_id / get_summary / search 在本版本
  因 API 返回 dict 而损坏（AttributeError/TypeError）；可用：get_structure_by_material_id、
  get_structures、get_materials_ids。因此 energy_above_hull 在本环境恒为 None + warning
  ——文档允许（StructureRecord.energy_above_hull 可空；§4.5 有"无 hull 数据选第一个"路径）。
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from pymatgen.core import Structure

from app.domain.normalizer import normalize_formula
from app.domain.schemas import StructureRecord
from app.storage.files import ServiceError

_MP_ID_PATTERN = re.compile(r"^mp-[\w-]+$")
_CACHE_TTL_SECONDS = 30 * 24 * 3600
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 2.0)


def _is_not_found(exc: Exception) -> bool:
    """实测 legacy MPRester 对不存在的 mp_id 抛 IndexError。"""
    text = str(exc).lower()
    return isinstance(exc, IndexError) or "not found" in text or "404" in text


class MPClient:
    """Materials Project bulk 结构获取（构造同步、调用异步、无状态）。"""

    def __init__(self, api_key: str, cache_dir: Path, *, timeout: float = 30.0):
        if not api_key:
            raise ServiceError(3004, "MP_API_KEY 未配置", {"reason": "Materials Project 功能不可用"})
        self._api_key = api_key
        self._cache_dir = Path(cache_dir)
        self._timeout = timeout

    async def get_bulk(
        self, *, mp_id: str | None = None, formula: str | None = None
    ) -> StructureRecord:
        if mp_id is not None:
            return await self._by_mp_id(mp_id)
        if formula is not None:
            return await self._by_formula(formula)
        raise ServiceError(1000, "INVALID_REQUEST", {"reason": "mp_id 与 formula 至少提供一个"})

    # ---- mp_id 路径 ----

    async def _by_mp_id(self, mp_id: str) -> StructureRecord:
        if not _MP_ID_PATTERN.fullmatch(mp_id):
            raise ServiceError(1000, "INVALID_REQUEST", {"reason": f"非法 mp_id: {mp_id}"})
        cached = self._read_cache(mp_id)
        if cached is not None:
            return self._record(cached, source="mp_id", mp_id=mp_id, warnings=["缓存命中"])

        def fetch() -> Structure:
            from pymatgen.ext.matproj import MPRester

            with MPRester(api_key=self._api_key) as m:
                return m.get_structure_by_material_id(mp_id)

        structure = await self._with_retry(fetch, not_found_hint=f"mp_id 不存在: {mp_id}")
        self._write_cache(mp_id, structure)
        return self._record(structure, source="mp_id", mp_id=mp_id, warnings=[])

    # ---- formula 路径 ----

    async def _by_formula(self, formula: str) -> StructureRecord:
        normalized, _warnings = normalize_formula(formula)

        def fetch() -> list[Structure]:
            from pymatgen.ext.matproj import MPRester

            with MPRester(api_key=self._api_key) as m:
                return m.get_structures(normalized)

        candidates = await self._with_retry(fetch, not_found_hint=f"化学式无匹配: {normalized}")
        if not candidates:
            raise ServiceError(3003, "STRUCTURE_NOT_FOUND", {"reason": f"化学式无匹配: {normalized}"})
        structure = candidates[0]
        warnings = [
            f"energy_above_hull 在本环境不可用（legacy MP API 的 hull 方法已损坏），取候选第一个"
        ]
        if len(candidates) > 1:
            warnings.append(f"化学式 {normalized} 有 {len(candidates)} 个候选结构，取第一个")
        return self._record(structure, source="formula", mp_id=None, warnings=warnings)

    # ---- 网络调用（超时 + 重试） ----

    async def _with_retry(self, fetch, *, not_found_hint: str):
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return await asyncio.wait_for(asyncio.to_thread(fetch), self._timeout)
            except asyncio.TimeoutError as exc:
                last_exc = exc
            except ServiceError:
                raise
            except Exception as exc:
                if _is_not_found(exc):
                    raise ServiceError(3003, "STRUCTURE_NOT_FOUND", {"reason": not_found_hint}) from exc
                last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(_BACKOFF_SECONDS[attempt])
        raise ServiceError(
            3004, "MP_UNAVAILABLE",
            {"reason": "可上传 CIF 继续", "detail": str(last_exc)[:200]},
        ) from last_exc

    # ---- 缓存 ----

    def _cache_path(self, mp_id: str) -> Path:
        return self._cache_dir / "mp" / f"{mp_id}.cif"

    def _read_cache(self, mp_id: str) -> Structure | None:
        path = self._cache_path(mp_id)
        if not path.is_file():
            return None
        if time.time() - path.stat().st_mtime > _CACHE_TTL_SECONDS:
            return None
        try:
            return Structure.from_file(path)
        except Exception as exc:
            raise ServiceError(1002, "INVALID_FILE", {"reason": f"缓存损坏: {exc}"}) from exc

    def _write_cache(self, mp_id: str, structure: Structure) -> None:
        from pymatgen.io.cif import CifWriter

        path = self._cache_path(mp_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        CifWriter(structure).write_file(path)

    # ---- 记录组装 ----

    @staticmethod
    def _record(
        structure: Structure,
        *,
        source: str,
        mp_id: str | None,
        warnings: list[str],
    ) -> StructureRecord:
        formula_normalized, _w = normalize_formula(structure.composition.formula)
        return StructureRecord(
            structure=structure,
            source=source,
            mp_id=mp_id,
            formula=structure.composition.formula,
            formula_normalized=formula_normalized,
            num_sites=len(structure),
            energy_above_hull=None,
            warnings=warnings,
        )
