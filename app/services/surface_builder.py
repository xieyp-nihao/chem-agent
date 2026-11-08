"""Slab 构建：体相结构 + SurfaceModel → SlabResult（终止面、层数、真空层、超胞、固定层）。

对应 docs/technical-design.md §4.6。同步、无状态。

关键语义（§3.3.5）：
- SurfaceModel.layers 是"沿表面法向的目标 hkl 平面数"，**不是原子层数**；
  Co3O4(111) 每个 hkl 平面内还含 Co/O 亚层。
- vacuum_angstrom 是 slab 上下真空区的合计厚度。

实现要点（实测校准）：
- SlabGenerator.get_slab() 没有 unit_layers 参数，min_slab_size 单位是 Å（§2.5）。
- 不设 max_normal_search：实测 Co3O4(111) 默认（None）get_slabs ≈ 2.9s，
  设 1 → 22s、设 2 → 70s，默认最快。
- lll_reduce=True 后 c 轴不垂直于 slab 面，层数统计与真空层计算都投影到
  表面法向 n̂ = normalize(a × b)；层数 = floor(span_n / d_hkl) + 1
  （实测 6 层 slab：span 27.08 Å、d111 4.68 Å → 6 层）。
"""
from __future__ import annotations

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

from app.domain.schemas import SlabResult, SurfaceModel
from app.storage.files import ServiceError

_MIN_INTERATOMIC_DISTANCE = 0.5
_VACUUM_LIKE_THRESHOLD = 5.0  # 已带真空的输入判定阈值（Å）


def _surface_normal(structure: Structure) -> np.ndarray:
    a = np.array(structure.lattice.matrix[0], dtype=float)
    b = np.array(structure.lattice.matrix[1], dtype=float)
    n = np.cross(a, b)
    norm = float(np.linalg.norm(n))
    if norm == 0.0:
        raise ServiceError(3005, "SURFACE_GENERATION_FAILED", {"reason": "lattice a、b 平行"})
    return n / norm


def _plane_bins(z_n: np.ndarray, d_hkl: float) -> np.ndarray:
    """按 hkl 平面分 bin：bin = floor((z_n - z_n.min()) / d_hkl)（clamp 到 ≥0）。"""
    return np.floor((z_n - z_n.min()) / d_hkl).astype(int)


def _species_symbol(site) -> str:
    """site.specie 可能是 Element 或 Specie（Element 没有 .element 属性）。"""
    specie = site.specie
    return specie.symbol if hasattr(specie, "symbol") else str(specie)


def _top_plane_formula(structure: Structure, bins: np.ndarray) -> str:
    """最顶层 hkl 平面的组成（用于终止面描述）。"""
    from collections import Counter
    top = int(bins.max())
    counts = Counter(
        _species_symbol(site) for site, b in zip(structure, bins) if b == top
    )
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(counts.items()))


def _top_plane_elements(structure: Structure, bins: np.ndarray) -> set[str]:
    top = int(bins.max())
    return {_species_symbol(site) for site, b in zip(structure, bins) if b == top}


class SurfaceBuilder:
    """bulk → slab 构建器（同步、无状态）。"""

    def build(self, bulk: Structure, surface: SurfaceModel) -> SlabResult:
        warnings: list[str] = []

        # 0) 前置校验
        if all(i == 0 for i in surface.miller):
            raise ServiceError(3005, "SURFACE_GENERATION_FAILED", {"reason": "Miller 指数全为 0"})
        if surface.vacuum_angstrom is None or surface.vacuum_angstrom <= 0:
            raise ServiceError(
                3005, "SURFACE_GENERATION_FAILED", {"reason": "生成 slab 必须提供真空层厚度（> 0）"}
            )

        # 1) 已带真空的输入提示（只警告，不静默）
        n0 = _surface_normal(bulk)
        z_n0 = bulk.cart_coords @ n0
        c_proj0 = float(np.array(bulk.lattice.matrix[2], dtype=float) @ n0)
        span0 = float(z_n0.max() - z_n0.min())
        if c_proj0 - span0 > _VACUUM_LIKE_THRESHOLD:
            warnings.append("输入结构沿表面法向已存在真空区，可能已是表面结构；仍按 bulk 处理切面")

        # 2) d_hkl → 最小 slab 厚度（Å，不是层数）
        d_hkl = float(bulk.lattice.d_hkl(surface.miller))
        min_slab = max(surface.layers * d_hkl, 1.0)

        # 3) SlabGenerator（不设 max_normal_search——实测默认最快）
        gen = SlabGenerator(
            bulk,
            surface.miller,
            min_slab_size=min_slab,
            min_vacuum_size=surface.vacuum_angstrom,
            lll_reduce=True,
            center_slab=True,
            primitive=False,
            reorient_lattice=True,
        )

        # 4) 候选终止面
        try:
            slabs = gen.get_slabs()
        except Exception as exc:  # pymatgen 内部失败统一映射
            raise ServiceError(
                3005, "SURFACE_GENERATION_FAILED", {"reason": f"get_slabs 失败: {exc}"}
            ) from exc
        if not slabs:
            raise ServiceError(3005, "SURFACE_GENERATION_FAILED", {"reason": "无候选终止面"})

        available: list[str] = []
        for i, s in enumerate(slabs):
            n_i = _surface_normal(s)
            bins_i = _plane_bins(s.cart_coords @ n_i, d_hkl)
            available.append(f"termination-{i}: top={_top_plane_formula(s, bins_i)}")

        selected: Structure | None = None
        if surface.termination:
            wanted = surface.termination.lower()
            for s in slabs:
                n_i = _surface_normal(s)
                bins_i = _plane_bins(s.cart_coords @ n_i, d_hkl)
                top_elements = _top_plane_elements(s, bins_i)
                if any(el.lower() in wanted for el in top_elements):
                    selected = s
                    break
            if selected is None:
                selected = slabs[0]
                warnings.append(
                    f"未找到与 termination='{surface.termination}' 匹配的终止面，取第一个候选"
                )
        else:
            if len(slabs) > 1:
                warnings.append(f"存在 {len(slabs)} 个候选终止面，未指定 termination，取第一个")
            selected = slabs[0]

        slab = selected

        # 5) 层数校验：平面数 = floor(span_n / d_hkl) + 1，与 layers 偏差 > 1 → warning
        n_hat = _surface_normal(slab)
        z_n = slab.cart_coords @ n_hat
        span_n = float(z_n.max() - z_n.min())
        n_planes = int(np.floor(span_n / d_hkl)) + 1
        if abs(n_planes - surface.layers) > 1:
            warnings.append(
                f"实际 hkl 平面数 {n_planes} 与目标 {surface.layers} 偏差 > 1（min_slab_size 被晶胞高度量化）"
            )

        # 6) 真空层精确化：c' = c × (span + vacuum) / c_proj；沿 n̂ 平移居中
        c_vec = np.array(slab.lattice.matrix[2], dtype=float)
        c_proj = float(c_vec @ n_hat)
        new_c_proj = span_n + surface.vacuum_angstrom
        new_c_vec = c_vec * (new_c_proj / c_proj)
        shift = (new_c_proj / 2.0) - (float(z_n.min()) + span_n / 2.0)
        cart_new = slab.cart_coords.copy()
        cart_new += n_hat * shift
        new_matrix = np.array(slab.lattice.matrix, dtype=float)
        new_matrix[2] = new_c_vec
        slab = Structure(new_matrix, slab.species, cart_new, coords_are_cartesian=True)

        # 7) 超胞
        if surface.supercell != (1, 1, 1):
            slab = slab.make_supercell(surface.supercell)

        # 8) 固定层：按 hkl 平面 bin 取底部 N 层
        n_hat = _surface_normal(slab)
        z_n = slab.cart_coords @ n_hat
        span_n = float(z_n.max() - z_n.min())
        bins = _plane_bins(z_n, d_hkl)
        fixed_indices: list[int] = []
        if surface.fixed_layers is not None:
            n_distinct = len(set(bins.tolist()))
            take = surface.fixed_layers
            if take > n_distinct:
                warnings.append(f"fixed_layers={take} 超过实际平面数 {n_distinct}，按全部平面处理")
                take = n_distinct
            bottom = sorted(set(bins.tolist()))[:take]
            fixed_indices = [i for i, b in enumerate(bins) if b in bottom]

        # 9) 最终校验：无重叠原子、c > 0、真空 > 0
        c_proj_final = float(np.array(slab.lattice.matrix[2], dtype=float) @ n_hat)
        vacuum_final = float(c_proj_final - span_n)
        if c_proj_final <= 0 or vacuum_final <= 0:
            raise ServiceError(
                3005, "SURFACE_GENERATION_FAILED",
                {"reason": f"c_proj={c_proj_final:.3f}, vacuum={vacuum_final:.3f} 必须 > 0"},
            )
        if len(slab) > 1:
            dm = slab.distance_matrix
            np.fill_diagonal(dm, np.inf)  # 排除自距离（对角线为 0）
            min_dist = float(dm.min())
            if min_dist < _MIN_INTERATOMIC_DISTANCE:
                raise ServiceError(
                    3005, "SURFACE_GENERATION_FAILED",
                    {"reason": f"存在重叠原子（最近邻 {min_dist:.3f} Å < {_MIN_INTERATOMIC_DISTANCE} Å）"},
                )

        return SlabResult(
            structure=slab,
            miller=surface.miller,
            layers=surface.layers,
            vacuum_angstrom=vacuum_final,
            slab_thickness_angstrom=span_n,
            termination=surface.termination,
            available_terminations=available,
            fixed_indices=fixed_indices,
            warnings=warnings,
        )
