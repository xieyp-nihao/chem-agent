"""VASP 输入生成：INCAR / POSCAR / KPOINTS 文本 + POTCAR 元素顺序。

对应 docs/technical-design.md §4.7。同步、无状态；不读卡片/session、不调 LLM。

覆盖优先级（高 → 低）：合并槽位 > 预设（MPRelaxSet/MPStaticSet）> 规则默认。
每个覆盖记录到 overrides_applied（INCAR 标签 → "slot"/"preset"/"rule"）。

说明（S3 计划已确认）：
- ISMEAR=0 为规则默认（实测 pymatgen 预设对 Co3O4 给出 -5；§4.7 表格未列
  ISMEAR，测试要求 0，故以 "rule" 来源覆盖）。
- 表面强制 Gamma-centered k 点（monkhorst-pack → gamma 转换 + warning）。
- 选择性动力学：generate() 签名不含 fixed_indices，本类按 surface.fixed_layers
  独立重算底部 N 个 hkl 平面（与 SurfaceBuilder 同一 bin 算法；结构已 reorient，
  用 span/layers 估计面间距）。
"""
from __future__ import annotations

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet

from app.domain.schemas import (
    CalcSlots,
    CalculationType,
    FunctionalFamily,
    HubbardU,
    KPointsScheme,
    Orbital,
    SurfaceModel,
    VASPDraft,
)
from app.storage.files import ServiceError

_ORBITAL_LDAUL = {Orbital.S: 0, Orbital.P: 1, Orbital.D: 2, Orbital.F: 3}
_VDW_IVDW = {"D3(BJ)": 12, "D3(zero)": 11}
_LOW_ENCUT_EV = 400.0
_MIN_INTERATOMIC_DISTANCE = 0.5


def _kpoints_text(scheme: KPointsScheme, mesh: tuple[int, int, int]) -> str:
    label = "Gamma" if scheme is KPointsScheme.GAMMA else "Monkhorst-Pack"
    return (
        "Automatic generation\n"
        "0\n"
        f"{label}\n"
        f"{mesh[0]} {mesh[1]} {mesh[2]}\n"
        "0 0 0\n"
    )


def _bottom_plane_indices(structure: Structure, surface: SurfaceModel) -> list[int]:
    """底部 fixed_layers 个 hkl 平面的原子下标（与 SurfaceBuilder 同语义的 bin 算法）。

    slab 结构已 reorient：面间距无法直接从（bulk 坐标系下的）miller 得到，
    用 span/layers 估计：bin = floor((z_n - z_min) / d_est)，clamp 到 layers-1。
    """
    if surface.layers is None:
        raise ServiceError(
            3006, "DRAFT_VALIDATION_FAILED", {"reason": "缺少 surface.layers，无法生成固定层"}
        )
    a = np.array(structure.lattice.matrix[0], dtype=float)
    b = np.array(structure.lattice.matrix[1], dtype=float)
    n = np.cross(a, b)
    n = n / np.linalg.norm(n)
    z_n = structure.cart_coords @ n
    d_est = float(z_n.max() - z_n.min()) / max(surface.layers, 1)
    bins = np.minimum(
        np.floor((z_n - z_n.min()) / d_est).astype(int), surface.layers - 1
    )
    take = min(surface.fixed_layers or 0, surface.layers)
    bottom = sorted(set(bins.tolist()))[:take]
    return [i for i, b in enumerate(bins) if b in bottom]


class VASPGenerator:
    """VASP 输入草案生成器（同步、无状态）。"""

    def generate(
        self,
        structure: Structure,
        slots: CalcSlots,
        surface: SurfaceModel | None = None,
    ) -> VASPDraft:
        if slots.calculation_type not in (CalculationType.RELAX, CalculationType.STATIC):
            raise ServiceError(
                3006, "DRAFT_VALIDATION_FAILED",
                {"reason": f"calculation_type={slots.calculation_type} 不支持（仅 relax/static）"},
            )

        is_slab = surface is not None
        relax = slots.calculation_type is CalculationType.RELAX
        preset_cls = MPRelaxSet if relax else MPStaticSet
        try:
            preset = preset_cls(structure)
        except Exception as exc:
            raise ServiceError(
                3006, "DRAFT_VALIDATION_FAILED", {"reason": f"pymatgen 预设生成失败: {exc}"}
            ) from exc

        incar: dict = dict(preset.incar)
        overrides: dict[str, str] = {}
        warnings: list[str] = []

        # ---- 纯 PBE 剥除预设 +U 标签 ----
        # MPRelaxSet/MPStaticSet 会带入 Materials Project 兼容参数（如 Co3O4 的
        # LDAU=3.32）；用户请求纯 PBE 时不继承 +U（剥除而非设 False，避免残留空标签）
        if slots.functional_family is not FunctionalFamily.PBE_U:
            for tag in ("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "LMAXMIX", "LDAUPRINT"):
                if tag in incar:
                    incar.pop(tag)
                    if tag == "LDAU":
                        warnings.append(
                            "已剥除预设的 +U 标签（MP 兼容参数，如 Co3O4 的 LDAU=3.32）；"
                            "如需 +U 请将 functional 设为 PBE+U 并提供 hubbard_u"
                        )

        # ---- ENCUT：槽位 > 预设 ----
        if slots.encut_ev is not None:
            incar["ENCUT"] = slots.encut_ev
            overrides["ENCUT"] = "slot"
        if float(incar.get("ENCUT", 0.0)) < _LOW_ENCUT_EV:
            warnings.append(
                f"ENCUT = {incar.get('ENCUT')} eV 低于 {_LOW_ENCUT_EV} eV，可能欠收敛（同 R7）"
            )

        # ---- ISMEAR=0（规则默认；预设实测为 -5）----
        incar["ISMEAR"] = 0
        overrides["ISMEAR"] = "rule"

        # ---- 结构弛豫/静态核心标签 ----
        if relax:
            incar["NSW"] = 100
            incar["IBRION"] = 2
            incar["EDIFF"] = slots.ediff if slots.ediff is not None else 1e-5
            incar["EDIFFG"] = slots.ediffg if slots.ediffg is not None else -0.02
            overrides["NSW"] = "rule"
            overrides["IBRION"] = "rule"
            overrides["EDIFF"] = "slot" if slots.ediff is not None else "rule"
            overrides["EDIFFG"] = "slot" if slots.ediffg is not None else "rule"
        else:
            incar["NSW"] = 0
            incar["IBRION"] = -1
            incar["ISIF"] = 2
            overrides["NSW"] = "rule"
            overrides["IBRION"] = "rule"
            overrides["ISIF"] = "rule"
            if slots.ediff is not None:
                incar["EDIFF"] = slots.ediff
                overrides["EDIFF"] = "slot"
            if slots.ediffg is not None:
                incar["EDIFFG"] = slots.ediffg
                overrides["EDIFFG"] = "slot"
        if relax:
            incar["ISIF"] = 2 if is_slab else 3
            overrides["ISIF"] = "rule"

        # ---- ISPIN：显式 True 才覆盖，否则沿用预设 ----
        if slots.spin_polarized is True:
            incar["ISPIN"] = 2
            overrides["ISPIN"] = "slot"
        # MAGMOM：沿用预设（不覆盖）

        # ---- +U（LDAU/LDAUL/LDAUU/LDAUJ）----
        if slots.functional_family is FunctionalFamily.PBE_U:
            self._apply_u(incar, overrides, structure, slots)

        # ---- vdW ----
        if slots.vdw:
            if slots.vdw in _VDW_IVDW:
                incar["IVDW"] = _VDW_IVDW[slots.vdw]
                overrides["IVDW"] = "slot"
            else:
                warnings.append(
                    f"vdW 方案 {slots.vdw!r} 不在自动映射范围（D3(BJ)/D3(zero)），"
                    "未写 IVDW，请自行确认"
                )

        # ---- 偶极校正 ----
        if is_slab and surface.dipole_correction:
            incar["LDIPOL"] = True
            incar["IDIPOL"] = 3
            overrides["LDIPOL"] = "slot"
            overrides["IDIPOL"] = "slot"

        # ---- KPOINTS：槽位显式网格 > 预设；表面强制 Gamma ----
        kp_source = "preset"
        if slots.kpoints is not None:
            scheme = slots.kpoints.scheme
            mesh = slots.kpoints.mesh
            kp_source = "slot"
        else:
            preset_kpts = preset.kpoints
            if preset_kpts.kpts:
                mesh = tuple(preset_kpts.kpts[0])
            else:
                from pymatgen.io.vasp.inputs import Kpoints
                mesh = tuple(
                    Kpoints.automatic_density(structure, 1000, force_gamma=is_slab).kpts[0]
                )
            scheme = (
                KPointsScheme.GAMMA
                if preset_kpts.style is preset_kpts.supported_modes.Gamma
                else KPointsScheme.MONKHORST_PACK
            )
        if is_slab and scheme is not KPointsScheme.GAMMA:
            warnings.append("表面计算强制 Gamma-centered k 点")
            scheme = KPointsScheme.GAMMA
            kp_source = "rule"
        if mesh is None or any(n < 1 for n in mesh):
            raise ServiceError(
                3006, "DRAFT_VALIDATION_FAILED", {"reason": f"k 点网格无效: {mesh}"}
            )
        kpoints_text = _kpoints_text(scheme, mesh)
        overrides["KPOINTS"] = kp_source

        # ---- POSCAR（选择性动力学）----
        poscar_structure = structure.copy()
        if is_slab and surface.fixed_layers is not None:
            fixed = _bottom_plane_indices(poscar_structure, surface)
            sd = [[True, True, True]] * len(poscar_structure)
            for i in fixed:
                sd[i] = [False, False, False]
            poscar_structure.add_site_property("selective_dynamics", sd)

        # ---- 输出 ----
        incar_obj = Incar(incar)
        incar_text = str(incar_obj)
        poscar_text = Poscar(poscar_structure).get_string()
        potcar_order = list(preset.potcar_symbols)

        return VASPDraft(
            incar=incar_text,
            poscar=poscar_text,
            kpoints=kpoints_text,
            potcar_order=potcar_order,
            overrides_applied=overrides,
            warnings=warnings,
            generator_version="1.0",
        )

    def _apply_u(
        self,
        incar: dict,
        overrides: dict[str, str],
        structure: Structure,
        slots: CalcSlots,
    ) -> None:
        # 用元素符号比较：CIF 可能带氧化态（composition.elements 会给出 Co2+/Co3+ 等 Species）
        elements_in_structure = [el.symbol for el in structure.composition.elements]
        u_map: dict[str, HubbardU] = {}
        for u in slots.hubbard_u:
            if u.element not in elements_in_structure:
                raise ServiceError(
                    3006, "DRAFT_VALIDATION_FAILED",
                    {"reason": f"U 值元素 {u.element} 不在结构中（结构元素：{elements_in_structure}）"},
                )
            if u.element in u_map and u_map[u.element].orbital is not u.orbital:
                raise ServiceError(
                    3006, "DRAFT_VALIDATION_FAILED",
                    {"reason": f"元素 {u.element} 存在多个轨道的 U 值，暂不支持生成（LDAUL 每元素单一值）"},
                )
            u_map[u.element] = u

        ldaul: list[int] = []
        ldauu: list[float] = []
        ldauj: list[float] = []
        for el in elements_in_structure:
            u = u_map.get(el)
            if u is None:
                ldaul.append(-1)
                ldauu.append(0.0)
                ldauj.append(0.0)
            else:
                ldaul.append(_ORBITAL_LDAUL[u.orbital])
                ldauu.append(u.u_ev if u.j_ev == 0 else u.u_ev + u.j_ev)
                ldauj.append(u.j_ev)

        incar["LDAU"] = True
        incar["LDAUTYPE"] = 2
        incar["LDAUL"] = ldaul
        incar["LDAUU"] = ldauu
        incar["LDAUJ"] = ldauj
        for key in ("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ"):
            overrides[key] = "slot"
