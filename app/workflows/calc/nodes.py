"""计算图节点（§5.3.3）。

- 节点失败写 status + error，条件边进入 notify_* 后 END，不抛异常
- human_confirm 单次执行一个 interrupt；多轮由条件边自环驱动（与文献图同模式）
- patch 白名单：encut_ev / kpoints / hubbard_u / ediff / ediffg；出现其他键 →
  4002 GRAPH_RESUME_FAILED，提示"结构级修改请重新提交 /calc/draft"
- patch 应用后重新生成 INCAR/KPOINTS（结构未变，POSCAR 输出逐字节一致）并重新校验
- output_files 目录 calc_{session_id}（session_id 先 sanitize，不含冒号）；
  清理策略不实现（TODO 留给运维脚本）
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pymatgen.core import Structure

from app.domain.normalizer import functional_family as family_of
from app.domain.normalizer import normalize_formula, normalize_functional
from app.domain.schemas import (
    CalcSlots,
    CalculationType,
    DFTParameters,
    FunctionalFamily,
    StructureRecord,
    SurfaceModel,
    ValidationReport,
    VASPDraft,
)
from app.storage.files import FileStore, ServiceError, sanitize_filename

logger = logging.getLogger(__name__)

MAX_CONFIRM_ROUNDS = 3
_PATCH_WHITELIST = ("encut_ev", "kpoints", "hubbard_u", "ediff", "ediffg")
_UNSUPPORTED_KEYWORDS = (
    "氧空位", "oxygen vacancy", "缺陷形成能", "formation energy", "界面", "interface",
    "异质结", "heterostructure", "过渡态", "transition state", "AIMD", "分子动力学",
    "声子", "phonon", "能带", "band structure", "DOS", "态密度", "光学性质", "GW",
    "HSE06", "SCAN", "r2SCAN",
)
_UNSUPPORTED_SUGGESTIONS = [
    "先对完整 slab 做 relax/static",
    "缺陷结构可通过 CIF 上传后计算",
    "界面/异质结可拆成两个单相 slab 分别计算，或上传已构建的超胞",
    "能带/DOS/声子/AIMD 请使用专门工作流，本系统只生成基态输入",
]


@dataclass
class CalcServices:
    """计算图依赖（注入，无模块级状态）。"""

    session_factory: Any
    file_store: FileStore
    query_cards: Any
    llm_extractor: Any            # 复用其 _model（同一 thinking-disabled 配置）
    mp_client: Any
    surface_builder: Any
    vasp_generator: Any
    rule_validator: Any


def _error(code: int, message: str, details: dict | None = None) -> dict:
    return {"code": code, "message": message, "details": details or {}}


def _to_dft_parameters(slots: CalcSlots) -> DFTParameters:
    """由合并槽位构造校验用 DFTParameters（functional 缺省按 family 补展示名）。"""
    display = slots.functional or (
        "PBE+U" if slots.functional_family is FunctionalFamily.PBE_U else "PBE"
    )
    return DFTParameters(
        calculation_type=slots.calculation_type or CalculationType.RELAX,
        functional=display,
        functional_family=slots.functional_family or FunctionalFamily.PBE,
        hubbard_u=list(slots.hubbard_u),
        encut_ev=slots.encut_ev,
        kpoints=slots.kpoints,
        ediff=slots.ediff,
        ediffg=slots.ediffg,
        spin_polarized=slots.spin_polarized,
        vdw=slots.vdw,
    )


def _slots_from_card(card) -> CalcSlots:
    p = card.parameters
    return CalcSlots(
        calculation_type=p.calculation_type,
        functional=p.functional,
        functional_family=p.functional_family,
        hubbard_u=list(p.hubbard_u),
        encut_ev=p.encut_ev,
        kpoints=p.kpoints,
        ediff=p.ediff,
        ediffg=p.ediffg,
        spin_polarized=p.spin_polarized,
        vdw=p.vdw,
        surface=card.surface,
    )


def _merge_slots(base: CalcSlots, overlay: CalcSlots) -> CalcSlots:
    """overlay 覆盖 base（同名字段非 None/非空才覆盖）。"""
    base_data = base.model_dump()
    for key, value in overlay.model_dump().items():
        if value is not None and value != []:
            base_data[key] = value
    return CalcSlots.model_validate(base_data)


# ---- LLM 槽位解析 ----


_SLOT_PROMPT = """你是 VASP 计算参数解析器。从用户的自然语言请求中提取计算槽位，只输出 JSON 对象（不要其他文本）。

JSON 结构（没提到的字段用 null / []）：
{
  "calculation_type": "relax" 或 "static"（弛豫/优化→relax；单点/静态→static；没提到→"relax" 的默认留给系统，输出 null 亦可）,
  "functional": "PBE" 或 "PBE+U"（含 DFT+U、Hubbard U 修正→PBE+U）,
  "formula": "化学式，如 Co3O4",
  "mp_id": "如 mp-18748",
  "encut_ev": 数值（eV）或 null,
  "kpoints": {"scheme": "gamma" 或 "monkhorst-pack", "mesh": [n1, n2, n3]} 或 null,
  "hubbard_u": [{"element": "Co", "orbital": "d", "u_ev": 3.5, "j_ev": 0.0}] 或 [],
  "surface": {"miller": [h, k, l], "layers": 整数, "vacuum_angstrom": 数值} 或 null,
  "notes": "原请求中的补充说明"
}

规则：
1. 只提取用户明确提到的内容；没提到 ENCUT / k 点 → null（系统会用预设补齐，不要编造）。
2. U 值：提到具体数值才填 hubbard_u；只说"DFT+U/PBE+U"没给数值 → hubbard_u 空数组 []。
3. 表面：(111) 这类是 Miller 指数 [1,1,1]；"6 层"是 layers；"真空 15 Å"是 vacuum_angstrom。
4. 单位：能量 eV、长度 Å。"""


async def _llm_parse_slots(services: CalcServices, raw_message: str) -> CalcSlots:
    model = services.llm_extractor._model  # 复用同一客户端（thinking 已关闭）
    messages = [
        SystemMessage(content=_SLOT_PROMPT),
        HumanMessage(content=raw_message),
    ]
    try:
        response = await model.ainvoke(messages)
    except Exception as exc:
        raise ServiceError(5001, "LLM_UNAVAILABLE", {"reason": str(exc)[:200]}) from exc
    content = response.content
    if isinstance(content, list):
        content = json.dumps(content, ensure_ascii=False)
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
        return CalcSlots.model_validate(data)
    except Exception as exc:
        raise ServiceError(1000, "INVALID_REQUEST", {"reason": f"槽位解析失败: {exc}"}) from exc


# ---- 节点 ----


async def collect_slots(state: dict, services: CalcServices) -> dict:
    slots = CalcSlots()
    card_snapshot = None
    if state.get("card_id"):
        card = await services.query_cards.get_by_id(state["card_id"])
        if card is not None:
            card_snapshot = card.model_dump(mode="json")
            slots = _slots_from_card(card)
    if state.get("user_slots"):
        slots = _merge_slots(slots, CalcSlots.model_validate(state["user_slots"]))
    # 自然语言槽位（LLM 解析结果同样算"当前请求"，user_slots 之后覆盖）
    if state.get("raw_message"):
        try:
            llm_slots = await _llm_parse_slots(services, state["raw_message"])
        except ServiceError as exc:
            return {
                "status": "failed",
                "error": _error(exc.code, exc.message, exc.details),
            }
        slots = _merge_slots(slots, llm_slots)
        if state.get("user_slots"):
            slots = _merge_slots(slots, CalcSlots.model_validate(state["user_slots"]))
    # 归一化
    if slots.functional:
        name = normalize_functional(slots.functional)
        slots = slots.model_copy(update={
            "functional": name,
            "functional_family": slots.functional_family or family_of(name),
        })
    if slots.formula:
        normalized, _w = normalize_formula(slots.formula)
        slots = slots.model_copy(update={"formula": normalized})
    return {
        "card_snapshot": card_snapshot,
        "slots": slots.model_dump(mode="json"),
    }


async def capability_check(state: dict, services: CalcServices) -> dict:
    slots = CalcSlots.model_validate(state["slots"])
    text = (state.get("raw_message") or "") + " " + (slots.notes or "")
    lowered = text.lower()

    # 能力边界：命中关键词即拒（第一版确实不支持）
    for kw in _UNSUPPORTED_KEYWORDS:
        if kw.lower() in lowered:
            return {
                "status": "unsupported",
                "capability": {"supported": False, "reason": f"第一版不支持：{kw}",
                               "suggestions": _UNSUPPORTED_SUGGESTIONS},
                "error": _error(3002, f"第一版不支持：{kw}",
                                {"suggestions": _UNSUPPORTED_SUGGESTIONS}),
            }
    if slots.calculation_type is not None and slots.calculation_type not in (
        CalculationType.RELAX, CalculationType.STATIC
    ):
        reason = "calculation_type 必须是 relax/static"
        return {"status": "unsupported", "capability": {"supported": False, "reason": reason,
                "suggestions": _UNSUPPORTED_SUGGESTIONS},
                "error": _error(3002, reason, {"suggestions": _UNSUPPORTED_SUGGESTIONS})}
    # family 为 None 属于"缺槽位"（3001）；给出非支持值才是"不支持"（3002）
    if slots.functional_family is not None and slots.functional_family not in (
        FunctionalFamily.PBE, FunctionalFamily.PBE_U
    ):
        reason = "functional_family 必须是 PBE/PBE+U（第一版生成能力）"
        return {"status": "unsupported", "capability": {"supported": False, "reason": reason,
                "suggestions": ["改用 PBE 或 PBE+U 重新提交"]},
                "error": _error(3002, reason, {"suggestions": ["改用 PBE 或 PBE+U 重新提交"]})}

    # 必填槽位
    missing: list[str] = []
    if slots.calculation_type is None:
        missing.append("calculation_type")
    if slots.functional_family is None:
        missing.append("functional")
    if not (slots.mp_id or slots.formula or slots.cif_path):
        missing.append("structure_source")
    if slots.surface is not None:
        if not slots.surface.miller:
            missing.append("surface.miller")
        if not slots.surface.layers:
            missing.append("surface.layers")
        if slots.surface.vacuum_angstrom is None:
            missing.append("surface.vacuum_angstrom")
    if slots.functional_family is FunctionalFamily.PBE_U and not slots.hubbard_u:
        missing.append("hubbard_u")
    if missing:
        return {
            "status": "missing_slots",
            "capability": {"supported": False, "reason": "缺少必填槽位",
                           "suggestions": missing},
            "error": _error(3001, "缺少必填槽位", {"missing_slots": missing}),
        }
    # ENCUT/kpoints 不要求显式给出（§5.3.4）：缺省用预设
    return {"status": "ok", "capability": {"supported": True, "reason": "", "suggestions": []}}


async def fetch_structure(state: dict, services: CalcServices) -> dict:
    slots = CalcSlots.model_validate(state["slots"])
    try:
        if slots.cif_path:
            data = services.file_store.read("structures", slots.cif_path)
            # pymatgen 2025.4.16 的 from_str 需要显式 fmt
            structure = Structure.from_str(data.decode("utf-8", errors="replace"), fmt="cif")
            record = StructureRecord(
                structure=structure, source="cif_upload", mp_id=None,
                formula=structure.composition.formula,
                formula_normalized=normalize_formula(structure.composition.formula)[0],
                num_sites=len(structure),
            )
        else:
            record = await services.mp_client.get_bulk(
                mp_id=slots.mp_id, formula=slots.formula
            )
        structure = record.structure
        slab_meta = None
        slab_dict = None
        if slots.surface is not None:
            slab = services.surface_builder.build(structure, slots.surface)
            structure = slab.structure
            slab_meta = {
                "miller": list(slab.miller),
                "layers": slab.layers,
                "vacuum_angstrom": slab.vacuum_angstrom,
                "slab_thickness_angstrom": slab.slab_thickness_angstrom,
                "termination": slab.termination,
                "available_terminations": slab.available_terminations,
                "fixed_indices": slab.fixed_indices,
                "warnings": slab.warnings,
            }
            slab_dict = structure.as_dict()
        return {
            "structure_dict": structure.as_dict(),
            "structure_meta": {
                "source": record.source, "mp_id": record.mp_id,
                "formula": record.formula, "formula_normalized": record.formula_normalized,
                "num_sites": record.num_sites,
                "energy_above_hull": record.energy_above_hull,
                "warnings": record.warnings,
            },
            "slab_dict": slab_dict,
            "slab_meta": slab_meta,
        }
    except ServiceError as exc:
        return {"status": "failed", "error": _error(exc.code, exc.message, exc.details)}
    except Exception as exc:  # CIF 解析等
        return {"status": "failed", "error": _error(1002, "INVALID_FILE", {"reason": str(exc)[:200]})}


async def generate_draft(state: dict, services: CalcServices) -> dict:
    slots = CalcSlots.model_validate(state["slots"])
    structure = Structure.from_dict(state["structure_dict"])
    surface = slots.surface
    try:
        draft = services.vasp_generator.generate(structure, slots, surface)
    except ServiceError as exc:
        return {"status": "failed", "error": _error(exc.code, exc.message, exc.details)}
    return {"draft": draft.model_dump(mode="json")}


async def validate_draft(state: dict, services: CalcServices) -> dict:
    slots = CalcSlots.model_validate(state["slots"])
    draft = VASPDraft.model_validate(state["draft"])
    params = _to_dft_parameters(slots)
    report = services.rule_validator.validate(
        parameters=params, surface=slots.surface,
        calculation_type=params.calculation_type,
    )
    findings = list(report.findings)
    # 草案一致性检查
    poscar_lines = draft.poscar.strip().splitlines()
    poscar_els = [line.split() for line in poscar_lines[5:6]][0] if len(poscar_lines) > 5 else []
    potcar_els = [s.split("_")[0] for s in draft.potcar_order]
    if potcar_els and poscar_els and potcar_els != poscar_els:
        from app.domain.schemas import RuleFinding, Severity
        findings.append(RuleFinding(
            rule_id="R99", severity=Severity.ERROR, field="potcar_order",
            message=f"POTCAR 顺序 {potcar_els} 与 POSCAR 元素 {poscar_els} 不一致",
        ))
    if not draft.poscar.strip() or not draft.kpoints.strip():
        return {"status": "failed", "error": _error(3006, "DRAFT_VALIDATION_FAILED",
                                                    {"reason": "POSCAR/KPOINTS 为空"})}
    report = report.model_copy(update={"findings": findings})
    return {"validation": report.model_dump(mode="json")}


# ---- human_confirm ----


def _build_confirm_request(state: dict, round_num: int) -> dict:
    draft = VASPDraft.model_validate(state["draft"])
    validation = ValidationReport.model_validate(state["validation"])
    return {
        "type": "confirm_request",
        "session_id": state["session_id"],
        "thread_id": state.get("thread_id") or f"calc:{state['session_id']}",
        "round": round_num,
        "draft": draft.model_dump(mode="json"),
        "validation": validation.model_dump(mode="json"),
        "applied_overrides": draft.overrides_applied,
    }


def _check_patch_whitelist(patch: dict) -> list[str]:
    forbidden = [k for k in patch if k not in _PATCH_WHITELIST]
    return forbidden


async def human_confirm(state: dict, services: CalcServices) -> dict:
    round_num = int(state.get("confirm_round", 0)) + 1
    decision: dict = interrupt(_build_confirm_request(state, round_num))
    messages: list[str] = list(state.get("messages", []))
    patch_history: list[dict] = list(state.get("patch_history", []))
    slots = CalcSlots.model_validate(state["slots"])
    draft = VASPDraft.model_validate(state["draft"])
    validation = ValidationReport.model_validate(state["validation"])

    choice = decision.get("decision") if isinstance(decision, dict) else None

    if choice == "abort":
        return {"status": "aborted", "confirm_round": round_num,
                "confirm_payload": decision, "messages": messages}

    if choice == "approve":
        if validation.errors:
            if round_num >= MAX_CONFIRM_ROUNDS:
                return {"status": "failed", "confirm_round": round_num,
                        "confirm_payload": decision,
                        "error": _error(3007, "CONFIRM_REQUIRED",
                                        {"reason": "存在 error，approve 无效且已达确认轮数上限"})}
            return {"status": "needs_confirm", "confirm_round": round_num,
                    "confirm_payload": decision,
                    "messages": messages + ["存在 error，approve 无效，需 patch 修正"]}
        return {"status": "approved", "confirm_round": round_num,
                "confirm_payload": decision, "messages": messages}

    if choice == "patch":
        patch = decision.get("patch") if isinstance(decision.get("patch"), dict) else {}
        forbidden = _check_patch_whitelist(patch)
        if forbidden:
            # 结构级修改拒绝：不应用，直接失败（4002）
            return {
                "status": "failed", "confirm_round": round_num,
                "confirm_payload": decision,
                "error": _error(4002, "GRAPH_RESUME_FAILED",
                                {"reason": f"禁止 patch 的字段：{forbidden}；"
                                           "结构级修改请重新提交 /calc/draft"}),
            }
        # 应用 patch（model_dump 合并 + model_validate 重建，保证嵌套模型强转）
        if patch:
            merged = slots.model_dump()
            merged.update(patch)
            slots = CalcSlots.model_validate(merged)
            patch_history.append({"round": round_num, "patch": patch})
            try:
                structure = Structure.from_dict(state["structure_dict"])
                draft = services.vasp_generator.generate(structure, slots, slots.surface)
            except ServiceError as exc:
                return {"status": "failed", "confirm_round": round_num,
                        "confirm_payload": decision,
                        "error": _error(exc.code, exc.message, exc.details)}
        # 重新校验
        params = _to_dft_parameters(slots)
        validation = services.rule_validator.validate(
            parameters=params, surface=slots.surface,
            calculation_type=params.calculation_type,
        )
        if validation.errors and round_num >= MAX_CONFIRM_ROUNDS:
            return {"status": "failed", "confirm_round": round_num,
                    "confirm_payload": decision,
                    "error": _error(3007, "CONFIRM_REQUIRED",
                                    {"reason": "patch 后仍有 error 且已达确认轮数上限"})}
        return {
            "slots": slots.model_dump(mode="json"),
            "draft": draft.model_dump(mode="json"),
            "validation": validation.model_dump(mode="json"),
            "confirm_round": round_num,
            "confirm_payload": decision,
            "patch_history": patch_history,
            "status": "needs_confirm" if validation.errors else "approved",
            "messages": messages + (["patch 后仍有 error，需继续修正"] if validation.errors else []),
        }

    return {"status": "failed", "confirm_round": round_num, "confirm_payload": decision,
            "error": _error(1000, "INVALID_REQUEST", {"reason": f"未知 decision: {choice}"})}


# ---- output_files ----


async def output_files(state: dict, services: CalcServices) -> dict:
    session_id = sanitize_filename(state["session_id"])
    draft = VASPDraft.model_validate(state["draft"])
    validation = ValidationReport.model_validate(state["validation"])
    slots = CalcSlots.model_validate(state["slots"])

    # TODO: 清理策略留给运维脚本（manifest 已记 created_at/regenerated_at）
    root = services.file_store.root / "outputs"
    target_dir = (root / f"calc_{session_id}").resolve()
    if not target_dir.is_relative_to(root.resolve()):
        return {"status": "failed",
                "error": _error(6002, "PATH_FORBIDDEN", {"reason": "输出目录越界"})}
    target_dir.mkdir(parents=True, exist_ok=True)
    regenerated = (target_dir / "INCAR").exists()

    now = datetime.now(timezone.utc)
    (target_dir / "INCAR").write_text(draft.incar, encoding="utf-8")
    (target_dir / "POSCAR").write_text(draft.poscar, encoding="utf-8")
    (target_dir / "KPOINTS").write_text(draft.kpoints, encoding="utf-8")
    manifest = {
        "session_id": state["session_id"],
        "thread_id": state.get("thread_id") or f"calc:{state['session_id']}",
        "system_formula": slots.formula,
        "parameters": _to_dft_parameters(slots).model_dump(mode="json"),
        "surface": slots.surface.model_dump(mode="json") if slots.surface else None,
        "validation": validation.model_dump(mode="json"),
        "overrides_applied": draft.overrides_applied,
        "generator_version": draft.generator_version,
        "created_at": now.isoformat(),
    }
    if regenerated:
        manifest["regenerated_at"] = now.isoformat()
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    files = []
    for name in ("INCAR", "POSCAR", "KPOINTS", "manifest.json"):
        p = target_dir / name
        files.append({
            "name": name,
            "path": str(p),
            "url": f"/files/calc_{session_id}/{name}",
            "size": p.stat().st_size,
        })
    return {"output_files": files}


# ---- notify 节点 ----


async def notify_unsupported(state: dict, services: CalcServices) -> dict:
    return {}


async def notify_missing_slots(state: dict, services: CalcServices) -> dict:
    return {}


async def notify_structure_failure(state: dict, services: CalcServices) -> dict:
    return {}


async def notify_aborted(state: dict, services: CalcServices) -> dict:
    return {}
