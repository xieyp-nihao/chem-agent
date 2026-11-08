"""S1 领域层测试。

覆盖（docs/technical-design.md §8.2 S1 退出条件）：
- R1–R7 每条规则至少 2 个正例 + 2 个反例
- ValidationReport 的 errors / warnings / infos / has_blocking_issues 四个 property
- R5 的 missing_fields 与 warning 一一对应
- normalizer 幂等性与归一化一致性
- DFTParameters / SurfaceModel / CalculationCard 的 JSON roundtrip
- Schema validator 负向测试
"""
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.domain.normalizer import functional_family, normalize_formula, normalize_functional
from app.domain.rules import RuleValidator
from app.domain.schemas import (
    CalculationCard,
    CalculationType,
    DFTParameters,
    Evidence,
    FunctionalFamily,
    HubbardU,
    KPoints,
    KPointsScheme,
    Orbital,
    ReviewDecision,
    ReviewRecord,
    Severity,
    SourceKind,
    SurfaceModel,
    ValidationContext,
    ValidationReport,
)

_DEFAULT_KP = KPoints(scheme=KPointsScheme.GAMMA, mesh=(3, 3, 1), gamma_centered=True)


def make_ctx(
    *,
    family: FunctionalFamily = FunctionalFamily.PBE,
    functional: str = "PBE",
    hubbard_u: list[HubbardU] | None = None,
    encut: float | None = 520.0,
    kpoints: KPoints | None = _DEFAULT_KP,
    surface: SurfaceModel | None = None,
) -> ValidationContext:
    """构造一个"干净"上下文（PBE、ENCUT 520、gamma k 点、无表面），测试按需覆盖。"""
    params = DFTParameters(
        calculation_type=CalculationType.RELAX,
        functional=functional,
        functional_family=family,
        hubbard_u=list(hubbard_u or []),
        encut_ev=encut,
        kpoints=kpoints,
    )
    return ValidationContext(
        parameters=params,
        surface=surface,
        structure=None,
        calculation_type=CalculationType.RELAX,
    )


def make_surface(vacuum: float = 15.0) -> SurfaceModel:
    return SurfaceModel(miller=(1, 1, 1), layers=6, vacuum_angstrom=vacuum)


def findings_of(report: ValidationReport, rule_id: str) -> list:
    return [f for f in report.findings if f.rule_id == rule_id]


# ============ R1 ============


class TestR1:
    def test_pbe_plus_u_without_u(self):
        report = RuleValidator().validate(
            make_ctx(family=FunctionalFamily.PBE_U, functional="PBE+U", hubbard_u=[])
        )
        hits = findings_of(report, "R1")
        assert len(hits) == 1
        f = hits[0]
        assert f.severity is Severity.ERROR
        assert f.field == "parameters.hubbard_u"
        assert f.message == "泛函为 PBE+U，但未提供任何 U 值"
        assert f.suggestion == "补充元素/轨道/U_eff，或改用 PBE"
        assert f.context == {"functional": "PBE+U"}

    def test_pbe_plus_u_without_u_with_surface(self):
        report = RuleValidator().validate(
            make_ctx(
                family=FunctionalFamily.PBE_U,
                functional="PBE+U",
                hubbard_u=[],
                surface=make_surface(),
            )
        )
        assert len(findings_of(report, "R1")) == 1

    def test_pbe_plus_u_with_u(self):
        report = RuleValidator().validate(
            make_ctx(
                family=FunctionalFamily.PBE_U,
                functional="PBE+U",
                hubbard_u=[HubbardU(element="Co", orbital=Orbital.D, u_ev=3.5)],
            )
        )
        assert findings_of(report, "R1") == []

    def test_pbe_without_u(self):
        report = RuleValidator().validate(make_ctx(hubbard_u=[]))
        assert findings_of(report, "R1") == []


# ============ R2 ============


class TestR2:
    def test_pbe_with_u(self):
        report = RuleValidator().validate(
            make_ctx(hubbard_u=[HubbardU(element="Co", orbital=Orbital.D, u_ev=3.5)])
        )
        hits = findings_of(report, "R2")
        assert len(hits) == 1
        f = hits[0]
        assert f.severity is Severity.WARNING
        assert f.field == "parameters.functional"
        assert f.message == "已提供 U 值，但泛函不是 PBE+U"
        assert f.suggestion == "将泛函改为 PBE+U，或确认生成时忽略 U 值"

    def test_other_with_u(self):
        report = RuleValidator().validate(
            make_ctx(
                family=FunctionalFamily.OTHER,
                functional="SCAN",
                hubbard_u=[HubbardU(element="Co", orbital=Orbital.D, u_ev=3.5)],
            )
        )
        assert len(findings_of(report, "R2")) == 1

    def test_pbe_plus_u_with_u(self):
        report = RuleValidator().validate(
            make_ctx(
                family=FunctionalFamily.PBE_U,
                functional="PBE+U",
                hubbard_u=[HubbardU(element="Co", orbital=Orbital.D, u_ev=3.5)],
            )
        )
        assert findings_of(report, "R2") == []

    def test_pbe_without_u(self):
        report = RuleValidator().validate(make_ctx(hubbard_u=[]))
        assert findings_of(report, "R2") == []


# ============ R3 ============


class TestR3:
    @pytest.mark.parametrize("vacuum", [12.0, 14.99])
    def test_vacuum_below_15(self, vacuum):
        report = RuleValidator().validate(make_ctx(surface=make_surface(vacuum)))
        hits = findings_of(report, "R3")
        assert len(hits) == 1
        f = hits[0]
        assert f.severity is Severity.WARNING
        assert f.field == "surface.vacuum_angstrom"
        assert f.message == f"表面真空层 {vacuum} Å 小于 15 Å"
        assert f.suggestion == "增大真空层至 ≥15 Å，避免周期像相互作用"
        assert f.context == {"miller": "(111)"}

    @pytest.mark.parametrize("vacuum", [15.0, 20.0])
    def test_vacuum_at_least_15(self, vacuum):
        report = RuleValidator().validate(make_ctx(surface=make_surface(vacuum)))
        assert findings_of(report, "R3") == []

    def test_no_surface(self):
        report = RuleValidator().validate(make_ctx(surface=None))
        assert findings_of(report, "R3") == []


# ============ R4 ============


class TestR4:
    @pytest.mark.parametrize("u_ev", [1.9, 5.5])
    def test_co_d_u_out_of_range(self, u_ev):
        report = RuleValidator().validate(
            make_ctx(hubbard_u=[HubbardU(element="Co", orbital=Orbital.D, u_ev=u_ev)])
        )
        hits = findings_of(report, "R4")
        assert len(hits) == 1
        f = hits[0]
        assert f.severity is Severity.WARNING
        assert f.field == "parameters.hubbard_u[0].u_ev"
        assert f.message == f"Co 3d 的 U_eff = {u_ev} eV 超出常用区间 [2.0, 5.0] eV"
        assert f.suggestion == "核对文献；常见 Co3O4 取 3.0–4.0 eV"

    @pytest.mark.parametrize("u_ev", [2.0, 5.0, 3.5])
    def test_co_d_u_in_range(self, u_ev):
        report = RuleValidator().validate(
            make_ctx(hubbard_u=[HubbardU(element="Co", orbital=Orbital.D, u_ev=u_ev)])
        )
        assert findings_of(report, "R4") == []

    def test_other_element_d(self):
        report = RuleValidator().validate(
            make_ctx(hubbard_u=[HubbardU(element="Mn", orbital=Orbital.D, u_ev=8.0)])
        )
        assert findings_of(report, "R4") == []


# ============ R5 ============


class TestR5:
    def test_missing_encut(self):
        report = RuleValidator().validate(make_ctx(encut=None))
        hits = findings_of(report, "R5")
        assert len(hits) == 1
        f = hits[0]
        assert f.severity is Severity.WARNING
        assert f.field == "parameters.encut_ev"
        assert f.message == "缺少必填参数：parameters.encut_ev"
        assert f.suggestion == "补充参数，或在审核中标注“论文未报告”"
        assert report.missing_fields == ["parameters.encut_ev"]

    def test_missing_kpoints(self):
        report = RuleValidator().validate(make_ctx(kpoints=None))
        assert report.missing_fields == ["parameters.kpoints"]
        assert [f.field for f in findings_of(report, "R5")] == ["parameters.kpoints"]

    def test_missing_both(self):
        report = RuleValidator().validate(make_ctx(encut=None, kpoints=None))
        assert len(findings_of(report, "R5")) == 2
        assert report.missing_fields == ["parameters.encut_ev", "parameters.kpoints"]

    def test_all_present(self):
        report = RuleValidator().validate(make_ctx())
        assert findings_of(report, "R5") == []
        assert report.missing_fields == []

    def test_all_present_with_surface(self):
        report = RuleValidator().validate(make_ctx(surface=make_surface()))
        assert findings_of(report, "R5") == []
        assert report.missing_fields == []

    def test_missing_fields_correspond_to_r5_warnings(self):
        report = RuleValidator().validate(make_ctx(encut=None, kpoints=None))
        r5_hits = findings_of(report, "R5")
        assert len(r5_hits) == 2
        assert all(f.severity is Severity.WARNING for f in r5_hits)
        fields = [f.field for f in r5_hits]
        assert report.missing_fields == fields
        assert len(report.missing_fields) == len(r5_hits)


# ============ R6 ============


class TestR6:
    def test_other_functional(self):
        report = RuleValidator().validate(
            make_ctx(family=FunctionalFamily.OTHER, functional="HSE06")
        )
        hits = findings_of(report, "R6")
        assert len(hits) == 1
        f = hits[0]
        assert f.severity is Severity.INFO
        assert f.field == "parameters.functional"
        assert f.message == "泛函 HSE06 不在第一版 PBE/PBE+U 生成能力内"
        assert f.suggestion == "卡片可保存；计算生成请改用 PBE/PBE+U"

    def test_other_with_surface_and_u(self):
        report = RuleValidator().validate(
            make_ctx(
                family=FunctionalFamily.OTHER,
                functional="SCAN",
                hubbard_u=[HubbardU(element="Co", orbital=Orbital.D, u_ev=3.5)],
                surface=make_surface(),
            )
        )
        assert len(findings_of(report, "R6")) == 1

    def test_pbe(self):
        report = RuleValidator().validate(make_ctx())
        assert findings_of(report, "R6") == []

    def test_pbe_plus_u(self):
        report = RuleValidator().validate(
            make_ctx(
                family=FunctionalFamily.PBE_U,
                functional="PBE+U",
                hubbard_u=[HubbardU(element="Co", orbital=Orbital.D, u_ev=3.5)],
            )
        )
        assert findings_of(report, "R6") == []


# ============ R7 ============


class TestR7:
    @pytest.mark.parametrize("encut", [350.0, 399.99])
    def test_encut_below_400(self, encut):
        report = RuleValidator().validate(make_ctx(encut=encut))
        hits = findings_of(report, "R7")
        assert len(hits) == 1
        f = hits[0]
        assert f.severity is Severity.WARNING
        assert f.field == "parameters.encut_ev"
        assert f.message == f"ENCUT = {encut} eV 低于 400 eV，可能欠收敛"
        assert f.suggestion == "核对 ENMAX；过渡金属氧化物常用 450–520 eV"

    @pytest.mark.parametrize("encut", [400.0, 520.0])
    def test_encut_at_least_400(self, encut):
        report = RuleValidator().validate(make_ctx(encut=encut))
        assert findings_of(report, "R7") == []

    def test_encut_none(self):
        report = RuleValidator().validate(make_ctx(encut=None))
        assert findings_of(report, "R7") == []


# ============ ValidationReport properties ============


class TestValidationReportProperties:
    def test_error_only_blocks(self):
        report = RuleValidator().validate(
            make_ctx(family=FunctionalFamily.PBE_U, functional="PBE+U", hubbard_u=[])
        )
        assert [f.rule_id for f in report.errors] == ["R1"]
        assert report.warnings == []
        assert report.infos == []
        assert report.missing_fields == []
        assert report.has_blocking_issues is True

    def test_warning_and_info_partition(self):
        report = RuleValidator().validate(
            make_ctx(
                family=FunctionalFamily.OTHER,
                functional="SCAN",
                hubbard_u=[HubbardU(element="Co", orbital=Orbital.D, u_ev=3.5)],
                encut=350.0,
            )
        )
        assert {f.rule_id for f in report.warnings} == {"R2", "R7"}
        assert [f.rule_id for f in report.infos] == ["R6"]
        assert report.errors == []
        assert report.has_blocking_issues is True

    def test_info_only_does_not_block(self):
        report = RuleValidator().validate(
            make_ctx(family=FunctionalFamily.OTHER, functional="HSE06", hubbard_u=[])
        )
        assert [f.rule_id for f in report.infos] == ["R6"]
        assert report.errors == []
        assert report.warnings == []
        assert report.missing_fields == []
        assert report.has_blocking_issues is False

    def test_missing_fields_only_blocks(self):
        report = RuleValidator().validate(make_ctx(encut=None))
        assert report.errors == []
        assert [f.rule_id for f in report.warnings] == ["R5"]
        assert report.missing_fields == ["parameters.encut_ev"]
        assert report.has_blocking_issues is True

    def test_clean_report(self):
        report = RuleValidator().validate(make_ctx())
        assert report.findings == []
        assert report.missing_fields == []
        assert report.rules_version == "1.0"
        assert report.has_blocking_issues is False


# ============ normalizer ============


class TestNormalizer:
    def test_formula_equivalents(self):
        assert normalize_formula("Co₃O₄") == ("Co3O4", [])
        assert normalize_formula("Co3 O4") == ("Co3O4", [])
        assert normalize_formula("Co3O4") == ("Co3O4", [])

    def test_surface_marker_stripped(self):
        assert normalize_formula("Co3O4(111)") == ("Co3O4", [])

    def test_hydrate_partial_normalization(self):
        assert normalize_formula("Co3O4·1.5H2O") == ("Co3O4·1.5H2O", [])
        assert normalize_formula("Na2SO4*10H2O") == ("Na2SO4·10H2O", [])

    def test_unparseable_fallback(self):
        normalized, warnings = normalize_formula("NotAFormula!")
        assert normalized == "NotAFormula!"
        assert len(warnings) == 1

    def test_doping_not_corrupted(self):
        # 非末尾括号（掺杂标记）不得被表面标记剥离破坏，应走回退保留原文
        assert normalize_formula("La(1-x)Sr(x)MnO3")[0] == "La(1-x)Sr(x)MnO3"

    @pytest.mark.parametrize(
        "text",
        ["Co₃O₄", "Co3 O4", "Co3O4", "Co3O4(111)", "Co3O4·1.5H2O", "Na2SO4*10H2O", "NotAFormula!"],
    )
    def test_formula_idempotent(self, text):
        assert normalize_formula(normalize_formula(text)[0]) == normalize_formula(text)


class TestFunctionalNormalizer:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("PBE", "PBE"),
            ("pbe", "PBE"),
            ("PBE + U", "PBE+U"),
            ("PBE+U (GGA)", "PBE+U"),
            ("GGA-PBE", "PBE"),
            ("HSE06", "HSE06"),
            ("PBE0", "PBE0"),
        ],
    )
    def test_normalize_functional(self, name, expected):
        assert normalize_functional(name) == expected

    def test_family_mapping(self):
        assert functional_family("PBE+U") is FunctionalFamily.PBE_U
        assert functional_family("PBE") is FunctionalFamily.PBE
        assert functional_family("HSE06") is FunctionalFamily.OTHER
        assert functional_family("SCAN") is FunctionalFamily.OTHER
        assert functional_family("PBE0") is FunctionalFamily.OTHER

    @pytest.mark.parametrize("name", ["PBE", "pbe", "PBE + U", "PBE+U (GGA)", "GGA-PBE", "hse06", "PBE0"])
    def test_functional_idempotent(self, name):
        assert normalize_functional(normalize_functional(name)) == normalize_functional(name)


# ============ Schema roundtrip ============


def _full_parameters() -> DFTParameters:
    return DFTParameters(
        calculation_type=CalculationType.RELAX,
        functional="PBE+U",
        functional_family=FunctionalFamily.PBE_U,
        hubbard_u=[
            HubbardU(element="Co", orbital=Orbital.D, u_ev=3.5),
            HubbardU(element="Co", orbital=Orbital.P, u_ev=2.0, j_ev=0.5),
        ],
        encut_ev=520.0,
        kpoints=KPoints(
            scheme=KPointsScheme.GAMMA,
            mesh=(3, 3, 1),
            gamma_centered=True,
            raw="3x3x1",
        ),
        ediff=1e-5,
        ediffg=-0.02,
        spin_polarized=True,
        vdw="D3(BJ)",
        pseudopotential="PAW",
        extra_incar={"LORBIT": "11"},
        notes="roundtrip example",
        evidence=[
            Evidence(
                source=SourceKind.TEXT,
                page=5,
                bbox=(10.0, 20.0, 30.0, 40.0),
                quote="ENCUT = 520 eV",
                field="encut_ev",
                confidence=0.9,
            )
        ],
    )


def _full_surface() -> SurfaceModel:
    return SurfaceModel(
        miller=(1, 1, 1),
        layers=6,
        vacuum_angstrom=15.0,
        termination="Co-terminated",
        fixed_layers=2,
        dipole_correction=True,
        supercell=(2, 2, 1),
        slab_thickness_angstrom=12.3,
        evidence=[Evidence(source=SourceKind.TABLE, page=4, quote="vacuum 15 A")],
    )


class TestSchemaRoundtrip:
    def test_dftparameters_roundtrip(self):
        model = _full_parameters()
        dumped = model.model_dump(mode="json")
        assert DFTParameters.model_validate(dumped).model_dump(mode="json") == dumped

    def test_surfacemodel_roundtrip(self):
        model = _full_surface()
        dumped = model.model_dump(mode="json")
        assert SurfaceModel.model_validate(dumped).model_dump(mode="json") == dumped

    def test_calculationcard_roundtrip(self):
        now = datetime.now(timezone.utc)
        card = CalculationCard(
            card_id=uuid4(),
            paper_id="9f2c",
            system_formula="Co3O4",
            system_normalized="Co3O4",
            parameters=_full_parameters(),
            surface=_full_surface(),
            missing_fields=[],
            validation=ValidationReport(findings=[], missing_fields=[], rules_version="1.0"),
            provenance=[
                Evidence(source=SourceKind.CAPTION, page=5, quote="Table 2. DFT parameters")
            ],
            review=ReviewRecord(
                reviewer="user",
                decision=ReviewDecision.EDIT,
                edited_fields=["parameters.encut_ev"],
                comment="按表 2 修正",
                reviewed_at=now,
                round=1,
            ),
            review_status="reviewed",
            created_at=now,
            updated_at=now,
        )
        dumped = card.model_dump(mode="json")
        assert CalculationCard.model_validate(dumped).model_dump(mode="json") == dumped


# ============ Schema validators（负向） ============


class TestSchemaValidators:
    def test_kpoints_gamma_requires_mesh(self):
        with pytest.raises(ValidationError):
            KPoints(scheme=KPointsScheme.GAMMA)

    def test_kpoints_mesh_components_positive(self):
        with pytest.raises(ValidationError):
            KPoints(scheme=KPointsScheme.MONKHORST_PACK, mesh=(0, 1, 1))

    def test_kpoints_explicit_requires_list(self):
        with pytest.raises(ValidationError):
            KPoints(scheme=KPointsScheme.EXPLICIT)
        KPoints(scheme=KPointsScheme.EXPLICIT, explicit=[(0.0, 0.0, 0.0)])  # 合法

    def test_hubbard_u_ge_j(self):
        with pytest.raises(ValidationError):
            HubbardU(element="Co", orbital=Orbital.D, u_ev=2.0, j_ev=3.0)

    def test_surface_miller_not_all_zero(self):
        with pytest.raises(ValidationError):
            SurfaceModel(miller=(0, 0, 0), layers=4, vacuum_angstrom=15.0)

    def test_surface_supercell_positive(self):
        with pytest.raises(ValidationError):
            SurfaceModel(
                miller=(1, 1, 1),
                layers=4,
                vacuum_angstrom=15.0,
                supercell=(0, 1, 1),
            )
