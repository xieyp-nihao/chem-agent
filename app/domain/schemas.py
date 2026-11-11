"""计算参数语义层：Pydantic v2 模型定义。

对应 docs/technical-design.md：
- §3.2 枚举与基础模型
- §3.3 证据与参数模型（3.3.1–3.3.9）
- §3.3.10 校验与序列化约定

约定：
- 除 §3.3.9 的 StructureRecord / SlabResult（arbitrary_types_allowed）外，
  所有模型统一 extra="forbid"。
- 字段缺失用 None 表示；单位写入字段名（*_ev、*_angstrom）。
- 本模块不依赖 pymatgen，仅依赖 pydantic 与标准库。
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ============ §3.2 枚举 ============


class CalculationType(str, Enum):
    RELAX = "relax"
    STATIC = "static"


class FunctionalFamily(str, Enum):
    PBE = "PBE"
    PBE_U = "PBE+U"
    OTHER = "other"


class KPointsScheme(str, Enum):
    GAMMA = "gamma"
    MONKHORST_PACK = "monkhorst-pack"
    EXPLICIT = "explicit"


class Orbital(str, Enum):
    S = "s"
    P = "p"
    D = "d"
    F = "f"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class SourceKind(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CAPTION = "caption"


class PaperStatus(str, Enum):
    QUEUED = "queued"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    NEEDS_REVIEW = "needs_review"
    SAVED = "saved"
    METHOD_NOT_FOUND = "method_not_found"
    FAILED = "failed"
    REJECTED = "rejected"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class ConfirmDecision(str, Enum):
    APPROVE = "approve"
    PATCH = "patch"
    ABORT = "abort"


# ============ §3.3.1 Evidence ============


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: SourceKind
    page: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    quote: str = Field(..., min_length=1, max_length=500)
    field: str | None = None          # 例如 "parameters.hubbard_u[0].u_ev"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


# ============ §3.3.2 HubbardU ============


class HubbardU(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element: str = Field(..., pattern=r"^[A-Z][a-z]?$")
    orbital: Orbital
    u_ev: float = Field(..., ge=0.0, le=20.0)   # 有效 U（U_eff）
    j_ev: float = Field(default=0.0, ge=0.0, le=10.0)
    evidence: Evidence | None = None

    @model_validator(mode="after")
    def _effective_u_ge_j(self):
        if self.u_ev < self.j_ev:
            raise ValueError("u_ev 必须大于等于 j_ev")
        return self


# ============ §3.3.3 KPoints ============


class KPoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheme: KPointsScheme
    mesh: tuple[int, int, int] | None = None
    spacing: float | None = Field(default=None, gt=0.0)   # 倒空间密度，Å^-1
    gamma_centered: bool | None = None
    explicit: list[tuple[float, float, float]] | None = None
    raw: str | None = None                                 # 论文原始写法

    @model_validator(mode="after")
    def _scheme_consistency(self):
        if self.scheme in (KPointsScheme.GAMMA, KPointsScheme.MONKHORST_PACK):
            if self.mesh is None:
                raise ValueError("gamma / monkhorst-pack 方案必须提供 mesh")
            if any(n < 1 for n in self.mesh):
                raise ValueError("mesh 的每个分量必须 >= 1")
        if self.scheme is KPointsScheme.EXPLICIT and not self.explicit:
            raise ValueError("explicit 方案必须提供 explicit 列表")
        return self


# ============ §3.3.4 DFTParameters ============


class DFTParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calculation_type: CalculationType
    functional: str = Field(..., min_length=1, max_length=80)   # 归一化展示名，如 "PBE"、"PBE+U"、"HSE06"
    functional_family: FunctionalFamily
    hubbard_u: list[HubbardU] = Field(default_factory=list)
    encut_ev: float | None = Field(default=None, ge=50.0, le=1000.0)
    kpoints: KPoints | None = None
    ediff: float | None = Field(default=None, gt=0.0, le=1e-2)
    ediffg: float | None = Field(default=None, le=0.0)   # 负值表示力收敛
    spin_polarized: bool | None = None
    vdw: str | None = Field(default=None, max_length=40)       # "D3(BJ)"、"D3(zero)"、"D2"
    pseudopotential: str | None = Field(default=None, max_length=80)
    extra_incar: dict[str, str] = Field(default_factory=dict)   # 论文报告但未建模的标签
    notes: str | None = Field(default=None, max_length=2000)
    evidence: list[Evidence] = Field(default_factory=list)


# ============ §3.3.5 SurfaceModel ============


class SurfaceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    miller: tuple[int, int, int]
    # 文献侧允许未知（论文未报告层数时置 null，进入 R5 missing_fields）；
    # 计算侧生成 slab 时必填（capability_check / SurfaceBuilder 校验）
    layers: int | None = Field(default=None, ge=1, le=30)  # 沿表面法向的目标 hkl 平面/层数
    # 文献侧允许未知（论文未报告真空层时置 null，进入 R5 missing_fields）；
    # 计算侧生成 slab 时必填（SurfaceBuilder 校验）
    vacuum_angstrom: float | None = Field(default=None, ge=0.0, le=100.0)
    termination: str | None = Field(default=None, max_length=120)
    fixed_layers: int | None = Field(default=None, ge=0, le=30)
    dipole_correction: bool = False
    supercell: tuple[int, int, int] = (1, 1, 1)
    slab_thickness_angstrom: float | None = Field(default=None, ge=0.0)
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("miller")
    @classmethod
    def _miller_not_zero(cls, v):
        if all(i == 0 for i in v):
            raise ValueError("Miller 指数不能全为 0")
        return v

    @field_validator("supercell")
    @classmethod
    def _supercell_positive(cls, v):
        if any(i < 1 for i in v):
            raise ValueError("supercell 的每个分量必须 >= 1")
        return v


# ============ §3.3.6 ValidationContext / RuleFinding / ValidationReport ============


class StructureInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formula: str
    elements: list[str]
    num_sites: int = Field(..., ge=1)
    is_slab: bool = False
    max_enmax_ev: float | None = Field(default=None, gt=0.0)  # 预留：由 POTCAR 推导


class ValidationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: DFTParameters
    surface: SurfaceModel | None = None
    structure: StructureInfo | None = None
    calculation_type: CalculationType


class RuleFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(..., pattern=r"^R[1-9][0-9]*$")
    severity: Severity
    field: str | None = None
    message: str
    suggestion: str | None = None
    context: dict[str, str] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[RuleFinding] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    rules_version: str = "1.0"

    @property
    def errors(self) -> list[RuleFinding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[RuleFinding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def infos(self) -> list[RuleFinding]:
        return [f for f in self.findings if f.severity is Severity.INFO]

    @property
    def has_blocking_issues(self) -> bool:
        """文献侧是否触发人工审核；info 永不触发。"""
        return bool(self.errors or self.warnings or self.missing_fields)


# ============ §3.3.7 ReviewRecord / CalculationCard ============


class ReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str | None = None
    decision: ReviewDecision
    edited_fields: list[str] = Field(default_factory=list)
    comment: str | None = Field(default=None, max_length=2000)
    reviewed_at: datetime
    round: int = Field(..., ge=1)


class CalculationCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_id: UUID
    paper_id: str
    system_formula: str
    system_normalized: str
    parameters: DFTParameters
    surface: SurfaceModel | None = None
    missing_fields: list[str] = Field(default_factory=list)
    validation: ValidationReport
    provenance: list[Evidence] = Field(default_factory=list)
    review: ReviewRecord | None = None
    review_status: Literal["auto_saved", "reviewed"] = "auto_saved"
    schema_version: str = "1.0"
    rules_version: str = "1.0"
    created_at: datetime
    updated_at: datetime


# ============ §3.3.8 抽取中间模型 ============


class ExtractedCalculation(BaseModel):
    """LLM 输出的一条计算；尚未分配 card_id。"""

    model_config = ConfigDict(extra="forbid")

    system_formula: str
    parameters: DFTParameters
    surface: SurfaceModel | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


# ============ §3.3.9 计算槽位与结构/草案模型 ============


class CalcSlots(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calculation_type: CalculationType | None = None
    functional: str | None = None
    functional_family: FunctionalFamily | None = None
    hubbard_u: list[HubbardU] = Field(default_factory=list)
    encut_ev: float | None = Field(default=None, ge=50.0, le=1000.0)
    kpoints: KPoints | None = None
    ediff: float | None = Field(default=None, gt=0.0, le=1e-2)
    ediffg: float | None = Field(default=None, le=0.0)   # 负值表示力收敛
    spin_polarized: bool | None = None
    vdw: str | None = None
    formula: str | None = None
    mp_id: str | None = None
    cif_path: str | None = None
    structure_source: Literal["mp_id", "formula", "cif_upload"] | None = None
    surface: SurfaceModel | None = None
    card_id: str | None = None
    notes: str | None = None
    missing_slots: list[str] = Field(default_factory=list)  # 派生，不来自用户


class StructureRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    structure: object            # pymatgen Structure；仅在 Service 内部传递
    source: Literal["mp_id", "formula", "cif_upload"]
    mp_id: str | None = None
    formula: str
    formula_normalized: str
    num_sites: int
    energy_above_hull: float | None = None
    warnings: list[str] = Field(default_factory=list)


class SlabResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    structure: object            # pymatgen Structure
    miller: tuple[int, int, int]
    layers: int
    vacuum_angstrom: float
    slab_thickness_angstrom: float
    termination: str | None = None
    available_terminations: list[str] = Field(default_factory=list)
    fixed_indices: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VASPDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incar: str
    poscar: str
    kpoints: str
    potcar_order: list[str]
    overrides_applied: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    generator_version: str = "1.0"


__all__ = [
    "CalculationType", "FunctionalFamily", "KPointsScheme", "Orbital", "Severity",
    "SourceKind", "PaperStatus", "ReviewDecision", "ConfirmDecision",
    "Evidence", "HubbardU", "KPoints", "DFTParameters", "SurfaceModel",
    "StructureInfo", "ValidationContext", "RuleFinding", "ValidationReport",
    "ReviewRecord", "CalculationCard", "ExtractedCalculation",
    "CalcSlots", "StructureRecord", "SlabResult", "VASPDraft",
]
