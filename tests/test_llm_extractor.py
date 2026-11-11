"""S4 LLM 抽取测试：真实 DeepSeek 评测 + mock 修复重试路径。

真实 LLM 用例（不打 mock）：
- 5 个评测集 case 逐篇抽取，与 expected 对比，输出字段级 precision/recall
- 合成样本：PBE+U 未给 U 值（R1 触发前置）、非 PBE/PBE+U 泛函（family=other）
mock 用例：非法 JSON 修复重试、重试耗尽 2004、API 异常 5001、空 method_text 契约。
"""
from __future__ import annotations

import asyncio
import json
import pathlib

import fitz
import pytest
from langchain_core.messages import AIMessage

from app.domain.schemas import FunctionalFamily
from app.services.llm_extractor import LLMExtractor
from app.services.pdf_parser import PDFParser
from app.storage.files import ServiceError

TESTS_DIR = pathlib.Path(__file__).parent
EVAL_SET = json.loads((TESTS_DIR / "eval_set.json").read_text(encoding="utf-8"))
FIXTURES = TESTS_DIR / "fixtures"

# ============ 评测集与解析 ============


def test_eval_set_loaded():
    cases = EVAL_SET.get("cases") or []
    assert len(cases) >= 5, "请先填 tests/eval_set.json（cases 至少 5 条）"
    total = sum(len(c.get("expected_calculations") or []) for c in cases)
    assert total >= 1, "tests/eval_set.json 的 expected_calculations 为空"


@pytest.fixture(scope="module")
def parsed_eval_papers():
    """评测集涉及的论文只解析一次。"""
    names = {c["paper_filename"] for c in EVAL_SET["cases"]}
    missing = [n for n in names if not (FIXTURES / "papers" / n).exists()]
    if missing:
        pytest.fail(f"fixtures 缺失：{missing}")
    parser = PDFParser()
    return {n: parser.parse(FIXTURES / "papers" / n) for n in names}


def _run_extract(parsed):
    return asyncio.run(
        LLMExtractor().extract(
            paper_id=parsed.paper_id,
            method_text=parsed.method_text,
            tables=parsed.tables,
            captions=parsed.captions,
        )
    )


# ============ 字段级对比 ============


def _match_calcs(expected_calcs, extracted_calcs):
    """贪婪唯一匹配：family(+2) + type(+1) + miller(+3) + U 元素集合(+1)，得分 ≥2 才配对。"""
    pairs = []
    used = set()
    for e in expected_calcs:
        best_i, best_score = None, 0
        for i, x in enumerate(extracted_calcs):
            if i in used:
                continue
            score = 0
            if x.parameters.functional_family.value == e["functional_family"]:
                score += 2
            if x.parameters.calculation_type.value == e["calculation_type"]:
                score += 1
            e_surf, x_surf = e.get("surface"), x.surface
            if e_surf and x_surf and tuple(x_surf.miller) == tuple(e_surf["miller"]):
                score += 3
            e_els = {u["element"] for u in e["hubbard_u"]}
            x_els = {u.element for u in x.parameters.hubbard_u}
            if e_els and e_els == x_els:
                score += 1
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score >= 2:
            pairs.append((e, extracted_calcs[best_i]))
            used.add(best_i)
        else:
            pairs.append((e, None))
    leftovers = [x for i, x in enumerate(extracted_calcs) if i not in used]
    return pairs, leftovers


def _u_multiset_match(expected_us, extracted_us, tol=0.5):
    """U 条目 multiset 匹配：(element, orbital) 相同且 |Δu|≤tol。"""
    tp = 0
    remaining = list(extracted_us)
    for e in expected_us:
        for i, x in enumerate(remaining):
            if (
                x.element == e["element"]
                and x.orbital.value == e["orbital"]
                and abs(x.u_ev - e["u_ev"]) <= tol
                and abs(x.j_ev - e["j_ev"]) <= 1e-6
            ):
                tp += 1
                remaining.pop(i)
                break
    return tp, len(expected_us) - tp, len(remaining)


def _compare_pair(e, x, metrics):
    """按对比容差累计 TP/FP/FN（x 为 None 表示该 expected 未匹配）。"""
    p = x.parameters if x else None
    x_surf = x.surface if x else None
    e_surf = e.get("surface")

    # calculation_type / functional_family
    for key, e_val, x_val in [
        ("calculation_type", e["calculation_type"], p.calculation_type.value if p else None),
        ("functional_family", e["functional_family"], p.functional_family.value if p else None),
    ]:
        if x_val is None:
            metrics[key]["fn"] += 1
        elif x_val == e_val:
            metrics[key]["tp"] += 1
        else:
            metrics[key]["fp"] += 1
            metrics[key]["fn"] += 1

    # encut_ev
    e_encut, x_encut = e["encut_ev"], p.encut_ev if p else None
    if e_encut is None:
        if x_encut is not None:
            metrics["encut_ev"]["fp"] += 1
    elif x_encut is None:
        metrics["encut_ev"]["fn"] += 1
    elif abs(x_encut - e_encut) <= 10.0:
        metrics["encut_ev"]["tp"] += 1
    else:
        metrics["encut_ev"]["fp"] += 1
        metrics["encut_ev"]["fn"] += 1

    # kpoints_scheme / kpoints_mesh
    x_kp = p.kpoints if p else None
    x_scheme = x_kp.scheme.value if x_kp else None
    x_mesh = tuple(x_kp.mesh) if x_kp and x_kp.mesh else None
    for key, e_val, x_val in [
        ("kpoints_scheme", e["kpoints_scheme"], x_scheme),
        ("kpoints_mesh", tuple(e["kpoints_mesh"]) if e["kpoints_mesh"] else None, x_mesh),
    ]:
        if e_val is None:
            if x_val is not None:
                metrics[key]["fp"] += 1
        elif x_val is None:
            metrics[key]["fn"] += 1
        elif x_val == e_val:
            metrics[key]["tp"] += 1
        else:
            metrics[key]["fp"] += 1
            metrics[key]["fn"] += 1

    # hubbard_u（条目级 multiset）
    tp, fn, fp = _u_multiset_match(
        e["hubbard_u"], list(p.hubbard_u) if p else []
    )
    metrics["hubbard_u"]["tp"] += tp
    metrics["hubbard_u"]["fn"] += fn
    metrics["hubbard_u"]["fp"] += fp

    # surface 字段
    for key, getter, tol in [
        ("vacuum_angstrom", (lambda s: s.vacuum_angstrom if s else None), 2.0),
        ("miller", (lambda s: tuple(s.miller) if s else None), None),
        ("layers", (lambda s: s.layers if s else None), None),
    ]:
        e_val = e_surf.get(key) if e_surf else None
        if key == "miller" and e_val is not None:
            e_val = tuple(e_val)  # JSON 数组 [h,k,l] → 元组，与抽取结果同型
        x_val = getter(x_surf)
        if e_val is None:
            if x_val is not None:
                metrics[key]["fp"] += 1
        elif x_val is None:
            metrics[key]["fn"] += 1
        elif tol is None:
            if x_val == e_val:
                metrics[key]["tp"] += 1
            else:
                metrics[key]["fp"] += 1
                metrics[key]["fn"] += 1
        elif abs(x_val - e_val) <= tol:
            metrics[key]["tp"] += 1
        else:
            metrics[key]["fp"] += 1
            metrics[key]["fn"] += 1

    # missing_fields_expected 一致性
    for field in e.get("missing_fields_expected") or []:
        metrics["missing"]["total"] += 1
        if field == "surface.vacuum_angstrom":
            ok = x_surf is not None and x_surf.vacuum_angstrom is None
        elif field == "kpoints_mesh":
            ok = x_mesh is None
        elif field == "hubbard_u":
            ok = p is not None and p.hubbard_u == []
        else:
            ok = False
        if ok:
            metrics["missing"]["hit"] += 1

    # evidence 覆盖率
    if x is not None:
        all_ev = list(p.evidence) + list(x.evidence) + (
            list(x_surf.evidence) if x_surf else []
        )
        ev_fields = {ev.field for ev in all_ev}
        for field in e.get("evidence_expected") or []:
            metrics["evidence"]["total"] += 1
            if any(f and f.startswith(field) for f in ev_fields):
                metrics["evidence"]["hit"] += 1


_TRACKED = [
    "calculation_type", "functional_family", "encut_ev", "kpoints_scheme",
    "kpoints_mesh", "hubbard_u", "vacuum_angstrom", "miller", "layers",
]


def _render_metrics(metrics) -> str:
    lines = ["字段级对比（TP / FP / FN / precision / recall）："]
    for key in _TRACKED:
        m = metrics[key]
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        total = tp + fp + fn
        if total == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        lines.append(f"  {key:20s} TP={tp:3d} FP={fp:3d} FN={fn:3d} P={prec:.2f} R={rec:.2f}")
    miss = metrics["missing"]
    ev = metrics["evidence"]
    lines.append(
        f"  {'missing 一致率':18s} {miss['hit']}/{miss['total']}"
        + (f" = {miss['hit']/miss['total']:.2f}" if miss["total"] else "（无标注）")
    )
    lines.append(
        f"  {'evidence 覆盖率':18s} {ev['hit']}/{ev['total']}"
        + (f" = {ev['hit']/ev['total']:.2f}" if ev["total"] else "（无标注）")
    )
    return "\n".join(lines)


# ============ 真实 DeepSeek 评测 ============


@pytest.mark.llm  # 真实 DeepSeek ×5（评测集逐篇抽取）
def test_eval_extraction_all_cases(parsed_eval_papers):
    metrics = {k: {"tp": 0, "fp": 0, "fn": 0} for k in _TRACKED}
    metrics["missing"] = {"hit": 0, "total": 0}
    metrics["evidence"] = {"hit": 0, "total": 0}
    all_leftovers = 0

    for case in EVAL_SET["cases"]:
        name = case["paper_filename"]
        parsed = parsed_eval_papers[name]
        extracted = _run_extract(parsed)
        assert len(extracted) >= 1, f"{name}：未抽出任何计算"

        pairs, leftovers = _match_calcs(case["expected_calculations"], extracted)
        all_leftovers += len(leftovers)
        for e, x in pairs:
            _compare_pair(e, x, metrics)

    table = _render_metrics(metrics)
    print("\n" + table + "\n")
    print(f"未匹配的抽取余量：{all_leftovers} 条")

    # 下限断言（首轮可行水平）
    def r(key):
        m = metrics[key]
        tp, fn = m["tp"], m["fn"]
        return tp / (tp + fn) if tp + fn else 1.0

    assert r("functional_family") >= 0.8, table
    assert r("encut_ev") >= 0.6, table
    assert r("kpoints_mesh") >= 0.6, table
    assert r("hubbard_u") >= 0.5, table
    assert r("vacuum_angstrom") >= 0.5, table
    assert r("miller") >= 0.6, table
    assert r("layers") >= 0.5, table
    if metrics["missing"]["total"]:
        assert (
            metrics["missing"]["hit"] / metrics["missing"]["total"] >= 0.6
        ), table


# ============ 合成样本（真实 LLM） ============


_SYNTH_PBE_U_NO_U = (
    "Computational details. We performed DFT+U (PBE+U) calculations with VASP using the "
    "PBE functional with an on-site Hubbard U correction applied to the Co 3d states. "
    "The values of U_eff used in this work are listed in Table S1 of the Supporting "
    "Information. The kinetic energy cutoff was 500 eV and the Brillouin zone was "
    "sampled with a 3x3x1 Gamma-centered k-point mesh. The Co3O4(111) slab model "
    "contains 8 layers and a vacuum region of 15 A."
)


@pytest.mark.llm  # 真实 DeepSeek（合成样本）
def test_pbe_plus_u_without_u_precondition():
    calcs = asyncio.run(LLMExtractor().extract("synth-r1", _SYNTH_PBE_U_NO_U, [], []))
    assert len(calcs) >= 1
    pbeu = [c for c in calcs if c.parameters.functional_family is FunctionalFamily.PBE_U]
    assert pbeu, "未抽出 PBE+U 计算"
    assert all(c.parameters.hubbard_u == [] for c in pbeu), (
        "PBE+U 但原文未给 U 值时，hubbard_u 应为空（R1 触发前置）"
    )


_SYNTH_HSE06 = (
    "Computational details. All DFT calculations were performed with the hybrid "
    "HSE06 exchange-correlation functional as implemented in VASP. A plane-wave "
    "cutoff of 520 eV and a 4x4x4 Monkhorst-Pack k-point grid were used for the "
    "bulk structure optimization."
)


@pytest.mark.llm  # 真实 DeepSeek（合成样本）
def test_other_functional_family():
    calcs = asyncio.run(LLMExtractor().extract("synth-other", _SYNTH_HSE06, [], []))
    assert len(calcs) >= 1
    assert any(
        c.parameters.functional_family is FunctionalFamily.OTHER
        and c.parameters.functional == "HSE06"
        for c in calcs
    )


# ============ mock：修复重试 / 错误路径 ============


_VALID_CALC_JSON = json.dumps({
    "calculations": [
        {
            "system_formula": "Co3O4",
            "parameters": {
                "calculation_type": "relax",
                "functional": "PBE",
                "functional_family": "PBE",
                "hubbard_u": [],
                "encut_ev": 400.0,
                "kpoints": {"scheme": "gamma", "mesh": [1, 1, 1], "gamma_centered": True},
                "evidence": [],
            },
            "surface": None,
            "evidence": [],
            "confidence": 0.9,
        }
    ]
})


class FakeModel:
    """鸭类型 ainvoke：按序列返回 AIMessage 或抛异常。"""

    def __init__(self, responses):
        self._responses = list(responses)

    async def ainvoke(self, messages):
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return AIMessage(content=resp)


def test_retry_invalid_json_then_success():
    model = FakeModel(["this is not json at all", _VALID_CALC_JSON])
    calcs = asyncio.run(
        LLMExtractor(model=model).extract("fake", "method text with VASP and PBE.", [], [])
    )
    assert len(calcs) == 1
    assert calcs[0].parameters.encut_ev == 400.0


def test_retry_exhausted_raises():
    model = FakeModel(["not json", "still not json"])
    with pytest.raises(ServiceError) as exc:
        asyncio.run(LLMExtractor(model=model).extract("fake", "method text.", [], []))
    assert exc.value.code == 2004


def test_llm_unavailable_raises():
    model = FakeModel([RuntimeError("api timeout")])
    with pytest.raises(ServiceError) as exc:
        asyncio.run(LLMExtractor(model=model).extract("fake", "method text.", [], []))
    assert exc.value.code == 5001


# ============ 分数层数 / 单条失败隔离（B 修复） ============


def _calc_item() -> dict:
    return {
        "system_formula": "Fe3O4",
        "parameters": {
            "calculation_type": "relax",
            "functional": "PBE",
            "functional_family": "PBE",
            "hubbard_u": [],
            "encut_ev": 520.0,
            "kpoints": {"scheme": "gamma", "mesh": [3, 3, 1], "gamma_centered": True},
            "evidence": [],
        },
        "surface": None,
        "evidence": [],
        "confidence": 0.9,
    }


def test_fractional_layers_rounded_with_note():
    item = _calc_item()
    item["surface"] = {"miller": [0, 0, 1], "layers": 7.5, "vacuum_angstrom": 10.6}
    model = FakeModel([json.dumps({"calculations": [item]})])
    calcs = asyncio.run(
        LLMExtractor(model=model).extract("fake", "method text with layers.", [], [])
    )
    assert len(calcs) == 1
    assert calcs[0].surface is not None
    assert calcs[0].surface.layers == 8  # 7.5 → 8（四舍五入）
    notes = calcs[0].parameters.notes or ""
    assert "7.5" in notes and "8" in notes


def test_missing_layers_accepted_as_none():
    """论文未报告层数（如 La-Doped 论文）→ layers=None 应通过抽取，交给 R5 提示。"""
    item = _calc_item()
    item["surface"] = {"miller": [3, 1, 1], "layers": None, "vacuum_angstrom": 16.0}
    model = FakeModel([json.dumps({"calculations": [item]})])
    calcs = asyncio.run(
        LLMExtractor(model=model).extract("fake", "method text with (311) slab.", [], [])
    )
    assert len(calcs) == 1
    assert calcs[0].surface is not None
    assert calcs[0].surface.layers is None


def test_invalid_calculation_skipped_keeps_valid():
    good = _calc_item()
    bad = _calc_item()
    bad["parameters"]["encut_ev"] = -5  # ge=50 失败
    fractional = _calc_item()
    fractional["surface"] = {"miller": [0, 0, 1], "layers": 7.5, "vacuum_angstrom": 10.6}
    model = FakeModel([json.dumps({"calculations": [good, bad, fractional]})])
    calcs = asyncio.run(
        LLMExtractor(model=model).extract("fake", "method text.", [], [])
    )
    assert len(calcs) == 2  # 无效的第 2 条被跳过，其余保留
    assert [c.surface.layers for c in calcs if c.surface] == [8]


def test_all_invalid_raises_2004_with_detail():
    bad = _calc_item()
    bad["parameters"]["encut_ev"] = -5
    payload = json.dumps({"calculations": [bad]})
    model = FakeModel([payload, payload])  # 修复重试仍失败
    with pytest.raises(ServiceError) as exc:
        asyncio.run(LLMExtractor(model=model).extract("fake", "method text.", [], []))
    assert exc.value.code == 2004
    assert "全部 calculations 校验失败" in str(exc.value.details.get("reason"))


def test_empty_method_text_rejected(tmp_path):
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    out = tmp_path / "scanned.pdf"
    doc.save(out)
    doc.close()
    parsed = PDFParser().parse(out)
    assert parsed.is_scanned is True
    with pytest.raises(ServiceError) as exc:
        asyncio.run(
            LLMExtractor(model=FakeModel([_VALID_CALC_JSON])).extract(
                "scanned", parsed.method_text, [], []
            )
        )
    assert exc.value.code == 2003
