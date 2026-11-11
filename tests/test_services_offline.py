"""S3 离线 Service 测试。

覆盖（docs/technical-design.md §8.2 S3 退出条件）：
- PDFParser：9 篇真实论文逐篇解析（method_found、method 块、bbox）；
  扫描版判定；加密 PDF 拒绝
- SurfaceBuilder：co3o4_bulk.cif → (111) 6 层 15 Å slab
- VASPGenerator：PBE/PBE+U 的 INCAR 断言、potcar_order、选择性动力学、
  U 元素不在结构报错、overrides_applied
- RuleValidatorService：与 domain 规则引擎的集成

fixture 缺失时直接 fail（不 mock 核心能力）。
"""
from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
import pytest
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Poscar

from app.domain.schemas import (
    CalcSlots,
    CalculationType,
    DFTParameters,
    FunctionalFamily,
    HubbardU,
    KPoints,
    KPointsScheme,
    Orbital,
    SurfaceModel,
)
from app.services.pdf_parser import PDFParser
from app.services.rule_validator import RuleValidatorService
from app.services.surface_builder import SurfaceBuilder
from app.services.vasp_generator import VASPGenerator
from app.storage.files import ServiceError

FIXTURES = Path(__file__).parent / "fixtures"
_PAPERS = sorted((FIXTURES / "papers").glob("*.pdf"))
_BULK_CIF = FIXTURES / "structures" / "co3o4_bulk.cif"


def _require_fixtures() -> None:
    if not _BULK_CIF.exists():
        pytest.fail(f"fixtures 缺失：{_BULK_CIF}")
    if len(_PAPERS) != 9:
        pytest.fail(f"fixtures 缺失：tests/fixtures/papers 应有 9 篇 PDF，实际 {len(_PAPERS)}")


_require_fixtures()


# ============ fixtures ============


@pytest.fixture(scope="module")
def parsed_papers() -> dict[str, object]:
    """9 篇论文只解析一次。"""
    return {p.name: PDFParser().parse(p) for p in _PAPERS}


@pytest.fixture(scope="module")
def bulk() -> Structure:
    return Structure.from_file(_BULK_CIF)


@pytest.fixture(scope="module")
def surface_model() -> SurfaceModel:
    return SurfaceModel(miller=(1, 1, 1), layers=6, vacuum_angstrom=15.0)


@pytest.fixture(scope="module")
def slab_result(bulk, surface_model) -> object:
    return SurfaceBuilder().build(bulk, surface_model)


@pytest.fixture(scope="module")
def slab_result_fixed(bulk) -> object:
    surface = SurfaceModel(miller=(1, 1, 1), layers=6, vacuum_angstrom=15.0, fixed_layers=2)
    return SurfaceBuilder().build(bulk, surface)


def _slots(**kw) -> CalcSlots:
    base = dict(
        calculation_type=CalculationType.RELAX,
        functional="PBE",
        functional_family=FunctionalFamily.PBE,
    )
    base.update(kw)
    return CalcSlots(**base)


def _gamma_kpoints() -> KPoints:
    return KPoints(scheme=KPointsScheme.GAMMA, mesh=(3, 3, 1), gamma_centered=True)


# ============ PDFParser ============


class TestPDFParser:
    def test_papers_have_method_sections(self, parsed_papers):
        found = {n: r for n, r in parsed_papers.items() if r.method_found}
        missing = [n for n, r in parsed_papers.items() if not r.method_found]
        assert len(found) >= 7, (
            f"method_found=True 仅 {len(found)}/9，少于 7 篇；"
            f"False 的篇目：{missing}"
        )
        for name, r in found.items():
            assert r.method_text.strip(), f"method_text 为空：{name}"
            assert any(b.kind == "method" for b in r.blocks), f"无 method 块：{name}"
            assert not r.is_scanned, f"误判为扫描版：{name}"

    def test_blocks_have_valid_bbox(self, parsed_papers):
        for name, r in parsed_papers.items():
            assert r.page_count >= 1, name
            for b in r.blocks:
                assert len(b.bbox) == 4, (name, b.kind)
                x0, y0, x1, y1 = b.bbox
                assert x1 > x0 and y1 > y0, (name, b.kind, b.bbox)

    def test_scanned_pdf_detected(self, tmp_path):
        doc = fitz.open()
        doc.new_page(width=595, height=842)  # 空白页：0 字符
        out = tmp_path / "scanned.pdf"
        doc.save(out)
        doc.close()
        parsed = PDFParser().parse(out)
        assert parsed.is_scanned is True
        assert parsed.method_found is False

    def test_encrypted_pdf_rejected(self, tmp_path):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "computational details")
        out = tmp_path / "encrypted.pdf"
        doc.save(out, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
        doc.close()
        with pytest.raises(ServiceError) as exc:
            PDFParser().parse(out)
        assert exc.value.code == 1002


# ============ SurfaceBuilder ============


class TestSurfaceBuilder:
    def test_build_co3o4_111_slab(self, slab_result):
        assert len(slab_result.structure) > 0
        assert slab_result.slab_thickness_angstrom > 0
        assert abs(slab_result.vacuum_angstrom - 15.0) <= 1.0
        assert slab_result.available_terminations, "available_terminations 为空"
        assert slab_result.miller == (1, 1, 1)
        assert slab_result.layers == 6

    def test_build_requires_layers(self, bulk):
        """文献侧允许 layers=None（R5 提示）；计算侧生成 slab 必须拒绝。"""
        surface = SurfaceModel(miller=(1, 1, 1), layers=None, vacuum_angstrom=15.0)
        with pytest.raises(ServiceError) as exc:
            SurfaceBuilder().build(bulk, surface)
        assert exc.value.code == 3005


# ============ VASPGenerator ============


class TestVASPGenerator:
    def test_pbe_slab_relax(self, slab_result, surface_model):
        draft = VASPGenerator().generate(
            slab_result.structure,
            _slots(encut_ev=520.0, kpoints=_gamma_kpoints()),
            surface_model,
        )
        incar = Incar.from_str(draft.incar)
        assert incar["ENCUT"] == 520.0
        assert incar["ISMEAR"] == 0
        assert incar["IBRION"] == 2
        assert incar["ISIF"] == 2
        assert incar["NSW"] == 100
        assert "Gamma" in draft.kpoints
        assert "3 3 1" in draft.kpoints
        poscar = Poscar.from_str(draft.poscar)
        poscar_els = [str(el) for el in poscar.structure.composition.elements]
        assert [s.split("_")[0] for s in draft.potcar_order] == poscar_els
        assert draft.overrides_applied["ENCUT"] == "slot"
        assert draft.overrides_applied["KPOINTS"] == "slot"

    def test_pbe_strips_preset_ldau(self, slab_result, surface_model):
        """纯 PBE 请求不得继承 MPRelaxSet 的 MP 兼容 +U 标签（demo 暴露的 bug）。"""
        draft = VASPGenerator().generate(
            slab_result.structure,
            _slots(encut_ev=520.0, kpoints=_gamma_kpoints()),
            surface_model,
        )
        incar = Incar.from_str(draft.incar)
        for tag in ("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "LMAXMIX", "LDAUPRINT"):
            assert tag not in incar, f"纯 PBE 的 INCAR 不应包含 {tag}"
        assert any("剥除预设的 +U" in w for w in draft.warnings)

    def test_pbe_plus_u(self, slab_result, surface_model):
        slots = _slots(
            functional="PBE+U",
            functional_family=FunctionalFamily.PBE_U,
            encut_ev=520.0,
            kpoints=_gamma_kpoints(),
            hubbard_u=[HubbardU(element="Co", orbital=Orbital.D, u_ev=3.5)],
        )
        draft = VASPGenerator().generate(slab_result.structure, slots, surface_model)
        incar = Incar.from_str(draft.incar)
        assert incar["LDAU"] is True
        assert incar["LDAUTYPE"] == 2
        # 用元素符号（CIF 可能带氧化态，composition.elements 会给出 Co2+/Co3+ 等 Species）
        els = [el.symbol for el in slab_result.structure.composition.elements]
        i_co, i_o = els.index("Co"), els.index("O")
        assert incar["LDAUL"][i_co] == 2
        assert incar["LDAUL"][i_o] == -1
        assert incar["LDAUU"][i_co] == 3.5
        assert incar["LDAUU"][i_o] == 0.0
        assert incar["LDAUJ"][i_o] == 0.0

    def test_u_element_not_in_structure(self, slab_result):
        slots = _slots(
            functional="PBE+U",
            functional_family=FunctionalFamily.PBE_U,
            hubbard_u=[HubbardU(element="Ni", orbital=Orbital.D, u_ev=3.0)],
        )
        with pytest.raises(ServiceError) as exc:
            VASPGenerator().generate(slab_result.structure, slots, None)
        assert exc.value.code == 3006

    def test_selective_dynamics_fixed_layers(self, slab_result_fixed):
        surface_fixed = SurfaceModel(
            miller=(1, 1, 1), layers=6, vacuum_angstrom=15.0, fixed_layers=2
        )
        draft = VASPGenerator().generate(
            slab_result_fixed.structure,
            _slots(encut_ev=520.0, kpoints=_gamma_kpoints()),
            surface_fixed,
        )
        assert "Selective dynamics" in draft.poscar
        poscar = Poscar.from_str(draft.poscar)
        struct = poscar.structure
        sd = struct.site_properties["selective_dynamics"]
        # 独立重算底部 2 个 hkl 平面的期望集合（与实现同语义、独立书写）
        a = np.array(struct.lattice.matrix[0], dtype=float)
        b = np.array(struct.lattice.matrix[1], dtype=float)
        n = np.cross(a, b)
        n = n / np.linalg.norm(n)
        z = struct.cart_coords @ n
        d_est = float(z.max() - z.min()) / 6
        expected = {int(i) for i in np.where((z - z.min()) < 2 * d_est)[0]}
        fixed = {i for i, flags in enumerate(sd) if not all(flags)}
        assert len(fixed) > 0
        assert fixed == expected


# ============ RuleValidatorService 集成 ============


class TestRuleValidatorIntegration:
    def test_r3_none_vacuum_does_not_crash(self):
        """vacuum_angstrom=None 的表面不触发 R3 比较（S4 Optional 后的守卫回归）。"""
        params = DFTParameters(
            calculation_type=CalculationType.RELAX,
            functional="PBE",
            functional_family=FunctionalFamily.PBE,
            encut_ev=520.0,
            kpoints=_gamma_kpoints(),
        )
        surface = SurfaceModel(miller=(1, 1, 1), layers=6, vacuum_angstrom=None)
        report = RuleValidatorService().validate(
            parameters=params,
            surface=surface,
            calculation_type=CalculationType.RELAX,
        )
        assert all(f.rule_id != "R3" for f in report.warnings)
        assert "surface.vacuum_angstrom" in report.missing_fields  # R5 归入缺失

    def test_clean_context_no_findings(self, surface_model):
        params = DFTParameters(
            calculation_type=CalculationType.RELAX,
            functional="PBE",
            functional_family=FunctionalFamily.PBE,
            encut_ev=520.0,
            kpoints=_gamma_kpoints(),
        )
        report = RuleValidatorService().validate(
            parameters=params,
            surface=surface_model,
            calculation_type=CalculationType.RELAX,
        )
        assert report.errors == []
        assert all(f.rule_id != "R7" for f in report.warnings), report.warnings
        assert all(f.rule_id != "R3" for f in report.warnings), report.warnings
        assert report.rules_version == "1.0"
        assert report.missing_fields == []
