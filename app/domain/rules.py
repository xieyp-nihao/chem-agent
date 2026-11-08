"""领域规则引擎 R1–R7（纯规则，不调 LLM、不读数据库、不发网络请求）。

对应 docs/technical-design.md §3.5：
- 每条规则是纯函数：RuleFn = Callable[[ValidationContext], list[RuleFinding]]
- 输出顺序固定 R1→R7；同一规则内按字段名排序；同一 (rule_id, field, message) 去重
- R1–R7 的级别/触发条件/字段/消息/建议严格按 §3.5.2 表格
- missing_fields 由 R5 生成，与 R5 warning 一一对应
"""
from typing import Callable

from app.domain.schemas import (
    FunctionalFamily,
    Orbital,
    RuleFinding,
    Severity,
    ValidationContext,
    ValidationReport,
)

RULES_VERSION = "1.0"

RuleFn = Callable[[ValidationContext], list[RuleFinding]]


def r1(ctx: ValidationContext) -> list[RuleFinding]:
    """R1 error：PBE+U 未提供任何 U 值。"""
    p = ctx.parameters
    if p.functional_family is FunctionalFamily.PBE_U and not p.hubbard_u:
        return [
            RuleFinding(
                rule_id="R1",
                severity=Severity.ERROR,
                field="parameters.hubbard_u",
                message="泛函为 PBE+U，但未提供任何 U 值",
                suggestion="补充元素/轨道/U_eff，或改用 PBE",
                context={"functional": p.functional},
            )
        ]
    return []


def r2(ctx: ValidationContext) -> list[RuleFinding]:
    """R2 warning：已提供 U 值但泛函不是 PBE+U。"""
    p = ctx.parameters
    if p.hubbard_u and p.functional_family is not FunctionalFamily.PBE_U:
        return [
            RuleFinding(
                rule_id="R2",
                severity=Severity.WARNING,
                field="parameters.functional",
                message="已提供 U 值，但泛函不是 PBE+U",
                suggestion="将泛函改为 PBE+U，或确认生成时忽略 U 值",
                context={"functional": p.functional},
            )
        ]
    return []


def r3(ctx: ValidationContext) -> list[RuleFinding]:
    """R3 warning：表面真空层小于 15 Å。

    vacuum_angstrom 允许为 None（论文未报告，S4 起 Optional）——None 不参与比较，
    缺失由 R5 归入 missing_fields。
    """
    s = ctx.surface
    if s is not None and s.vacuum_angstrom is not None and s.vacuum_angstrom < 15.0:
        m = s.miller
        return [
            RuleFinding(
                rule_id="R3",
                severity=Severity.WARNING,
                field="surface.vacuum_angstrom",
                message=f"表面真空层 {s.vacuum_angstrom} Å 小于 15 Å",
                suggestion="增大真空层至 ≥15 Å，避免周期像相互作用",
                context={"miller": f"({m[0]}{m[1]}{m[2]})"},
            )
        ]
    return []


def r4(ctx: ValidationContext) -> list[RuleFinding]:
    """R4 warning：Co 3d 的 U_eff 超出常用区间 [2.0, 5.0] eV。"""
    findings: list[RuleFinding] = []
    for i, u in enumerate(ctx.parameters.hubbard_u):
        if u.element == "Co" and u.orbital is Orbital.D and not (2.0 <= u.u_ev <= 5.0):
            findings.append(
                RuleFinding(
                    rule_id="R4",
                    severity=Severity.WARNING,
                    field=f"parameters.hubbard_u[{i}].u_ev",
                    message=f"Co 3d 的 U_eff = {u.u_ev} eV 超出常用区间 [2.0, 5.0] eV",
                    suggestion="核对文献；常见 Co3O4 取 3.0–4.0 eV",
                )
            )
    return findings


def r5(ctx: ValidationContext) -> list[RuleFinding]:
    """R5 warning：必填字段缺失（集合见 §3.5.2）。

    注：calculation_type / functional 及 surface 的 miller/layers/vacuum_angstrom
    是 Schema 必填字段，模型构造时即被保证，对应检查不会触发；按文档表格字面保留
    检查，实际可触发的是 encut_ev / kpoints。hubbard_u 由 R1 以 error 处理，不重复。
    """
    p = ctx.parameters
    missing: list[str] = []

    # 始终必填（calculation_type/functional 为模型必填，检查永不触发，保留以与表对齐）
    if p.calculation_type is None:
        missing.append("parameters.calculation_type")
    if p.functional is None:
        missing.append("parameters.functional")
    if p.encut_ev is None:
        missing.append("parameters.encut_ev")
    if p.kpoints is None:
        missing.append("parameters.kpoints")

    # 含表面必填（miller/layers/vacuum_angstrom 为模型必填，检查永不触发）
    if ctx.surface is not None:
        if ctx.surface.miller is None:
            missing.append("surface.miller")
        if ctx.surface.layers is None:
            missing.append("surface.layers")
        if ctx.surface.vacuum_angstrom is None:
            missing.append("surface.vacuum_angstrom")

    return [
        RuleFinding(
            rule_id="R5",
            severity=Severity.WARNING,
            field=field,
            message=f"缺少必填参数：{field}",
            suggestion="补充参数，或在审核中标注“论文未报告”",
        )
        for field in missing
    ]


def r6(ctx: ValidationContext) -> list[RuleFinding]:
    """R6 info：泛函不在第一版 PBE/PBE+U 生成能力内。"""
    p = ctx.parameters
    if p.functional_family is FunctionalFamily.OTHER:
        return [
            RuleFinding(
                rule_id="R6",
                severity=Severity.INFO,
                field="parameters.functional",
                message=f"泛函 {p.functional} 不在第一版 PBE/PBE+U 生成能力内",
                suggestion="卡片可保存；计算生成请改用 PBE/PBE+U",
                context={"functional": p.functional},
            )
        ]
    return []


def r7(ctx: ValidationContext) -> list[RuleFinding]:
    """R7 warning：ENCUT 低于 400 eV。"""
    p = ctx.parameters
    if p.encut_ev is not None and p.encut_ev < 400.0:
        return [
            RuleFinding(
                rule_id="R7",
                severity=Severity.WARNING,
                field="parameters.encut_ev",
                message=f"ENCUT = {p.encut_ev} eV 低于 400 eV，可能欠收敛",
                suggestion="核对 ENMAX；过渡金属氧化物常用 450–520 eV",
            )
        ]
    return []


RULES: list[tuple[str, RuleFn]] = [
    ("R1", r1),
    ("R2", r2),
    ("R3", r3),
    ("R4", r4),
    ("R5", r5),
    ("R6", r6),
    ("R7", r7),
]


class RuleValidator:
    """汇总 R1–R7，输出 ValidationReport。"""

    def validate(self, ctx: ValidationContext) -> ValidationReport:
        findings: list[RuleFinding] = []
        for _rule_id, fn in RULES:
            findings.extend(fn(ctx))

        # 同一 (rule_id, field, message) 去重，保持首次出现顺序
        seen: set[tuple[str, str | None, str]] = set()
        deduped: list[RuleFinding] = []
        for f in findings:
            key = (f.rule_id, f.field, f.message)
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        # 规则顺序 R1→R7 保持，同一规则内按字段名排序
        rule_order = {rule_id: i for i, (rule_id, _fn) in enumerate(RULES)}
        deduped.sort(key=lambda f: (rule_order[f.rule_id], f.field or ""))

        # missing_fields 由 R5 派生，与 R5 warning 一一对应
        missing_fields = [f.field for f in deduped if f.rule_id == "R5" and f.field is not None]

        return ValidationReport(
            findings=deduped,
            missing_fields=missing_fields,
            rules_version=RULES_VERSION,
        )
