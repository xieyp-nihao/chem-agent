"""化学式 / 泛函归一化（确定性，不调 LLM）。

对应 docs/technical-design.md §3.4：
- Unicode 下标转数字 → 剥表面标记 → 去空白 → Composition.reduced_formula
- 水合物（·、•、*、⋅ 分隔）：pymatgen 2025.4.16 无法解析任何水合物形式（实测），
  采用「拆分 + 部分归一化」：主式走 Composition，水合物部分文本归一，
  用规范分隔符 · 重接。已知限制：括号写法 "Co3O4(H2O)1.5" 不被识别为水合物，
  会被 Composition 合并进主式。
- 解析失败：保留去空格原文 + warning，不抛异常。
- 归一化必须幂等：normalize_formula(normalize_formula(x)[0]) == normalize_formula(x)。
"""
from __future__ import annotations

import re

from pymatgen.core import Composition

from app.domain.schemas import FunctionalFamily

# 下标 ₀-₉ → 0-9（必须先于 Composition，pymatgen 不识别 Unicode 下标）
_SUBSCRIPT_TRANS = str.maketrans({c: str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")})

# 水合物分隔符（统一输出为 ·）
_HYDRATE_SEPARATORS = "·•*⋅"

# 末尾表面标记，如 (111)、(001)、(1-10)；锚定行尾，避免误剥掺杂式中的 (1-x)
_SURFACE_MARKER = re.compile(r"\(\d[0-9\-]*\)$")

# 末尾括号限定词，如 (GGA)、(PAW)；锚定行尾
_PAREN_QUALIFIER = re.compile(r"\([^()]*\)$")


def normalize_formula(text: str) -> tuple[str, list[str]]:
    """返回 (system_normalized, warnings)。

    幂等：normalize_formula(normalize_formula(x)[0]) == normalize_formula(x)。
    """
    cleaned = text.translate(_SUBSCRIPT_TRANS)
    # 剥末尾表面标记（可能多段，如 (111)(2x2)，循环剥保证幂等）
    prev = None
    while cleaned != prev:
        prev = cleaned
        cleaned = _SURFACE_MARKER.sub("", cleaned)
    cleaned = "".join(cleaned.split())  # 去所有空白
    if not cleaned:
        return cleaned, [f"无法解析化学式：{text}，已保留原文"]

    # 水合物拆分：取最后一个分隔符，两侧非空才认定
    positions = [i for i, ch in enumerate(cleaned) if ch in _HYDRATE_SEPARATORS]
    if positions:
        pos = positions[-1]
        main, hydrate = cleaned[:pos], cleaned[pos + 1:]
        if main and hydrate:
            main_normalized, main_warnings = _via_composition(main)
            if main_warnings:
                # 主式解析失败 → 整体回退
                return cleaned, [f"无法解析化学式：{text}，已保留原文"]
            return f"{main_normalized}·{hydrate}", []

    normalized, warnings = _via_composition(cleaned)
    return normalized, warnings


def _via_composition(formula: str) -> tuple[str, list[str]]:
    try:
        return Composition(formula).reduced_formula, []
    except Exception:
        return formula, [f"无法解析化学式：{formula}，已保留原文"]


def normalize_functional(name: str) -> str:
    """泛函展示名归一化：只做明确映射，其余保留大写化原样。幂等。"""
    cleaned = name.upper()
    cleaned = "".join(cleaned.split())
    # 剥末尾括号限定词（可能多段，循环剥保证幂等）
    prev = None
    while cleaned != prev:
        prev = cleaned
        cleaned = _PAREN_QUALIFIER.sub("", cleaned)
    if cleaned in ("GGA-PBE", "GGAPBE"):
        return "PBE"
    if cleaned == "PBE+U":
        return "PBE+U"
    if cleaned == "PBE":
        return "PBE"
    # PBE0 / PBESOL / HSE06 / SCAN 等不误归一为 PBE，保留展示名
    return cleaned


def functional_family(name: str) -> FunctionalFamily:
    """归一化后映射到泛函族：PBE / PBE+U / other。"""
    normalized = normalize_functional(name)
    if normalized == "PBE":
        return FunctionalFamily.PBE
    if normalized == "PBE+U":
        return FunctionalFamily.PBE_U
    return FunctionalFamily.OTHER
