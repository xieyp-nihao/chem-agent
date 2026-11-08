"""LLM 参数抽取：方法段 + 表格/表注 → list[ExtractedCalculation]。

对应 docs/technical-design.md §4.3：
- DeepSeek（langchain-openai ChatOpenAI），JSON mode + Pydantic 校验，不依赖 tool calling
- 只输出 JSON、不得编造、缺失用 null、逐字段 evidence、一次计算一个对象
- 输入预算约 24000 字符（可配置）：方法段优先，表格/表注按 DFT 关键词相关度
  降序，超限先丢相关度最低的表格、再截方法段尾部
- Pydantic 校验失败 → 错误摘要回传模型，最多修复重试 1 次；仍失败抛 2004
- API 异常（超时/鉴权/限流）→ 5001 LLM_UNAVAILABLE
- functional / functional_family 由 normalizer 归一化后写入（§3.4）
- 无状态：截断等 warning 走 logging；不写数据库、不建卡片、不调 RuleValidator
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app.config import settings
from app.domain.normalizer import functional_family as family_of
from app.domain.normalizer import normalize_functional
from app.domain.schemas import (
    DFTParameters,
    Evidence,
    ExtractedCalculation,
    FunctionalFamily,
    HubbardU,
    KPoints,
    SurfaceModel,
)
from app.services.pdf_parser import TextBlock
from app.storage.files import ServiceError

logger = logging.getLogger(__name__)

DEFAULT_MAX_INPUT_CHARS = 24000
_DFT_KEYWORDS = re.compile(
    r"VASP|DFT|PBE|cutoff|k-point|k point|Hubbard|ENCUT|U_eff|Ueff|\beV\b|\bRy\b|Å|Angstrom",
    re.IGNORECASE,
)
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _clean_model_dict(obj: Any, cls: type) -> dict:
    """宽容解析（§3.3.10）：丢弃未知键；非 Optional 且有默认值的字段，null 回退默认。

    LLM 常对非 Optional 字段输出 null（如 surface.dipole_correction、hubbard_u[].j_ev），
    严格校验会误伤；先清理再 model_validate。必填且无默认的字段保持 null → 校验失败
    → 触发修复重试（语义正确：模型必须补上）。
    """
    if not isinstance(obj, dict):
        return obj
    fields = cls.model_fields
    out: dict = {}
    for k, v in obj.items():
        if k not in fields:
            continue  # 丢弃未知键
        f = fields[k]
        if v is None and not f.is_required() and f.default is not None:
            out[k] = f.default_factory() if f.default_factory is not None else f.default
        else:
            out[k] = v
    return out


def _clean_item(item: Any) -> dict:
    """对 LLM 输出的 calculation 元素做嵌套宽容清理。"""
    if not isinstance(item, dict):
        return item
    params = item.get("parameters")
    if isinstance(params, dict):
        params = _clean_model_dict(params, DFTParameters)
        kp = params.get("kpoints")
        if isinstance(kp, dict):
            params["kpoints"] = _clean_model_dict(kp, KPoints)
        hu = params.get("hubbard_u")
        if isinstance(hu, list):
            cleaned_u = []
            for u in hu:
                if isinstance(u, dict):
                    u = _clean_model_dict(u, HubbardU)
                    if isinstance(u.get("evidence"), dict):
                        u["evidence"] = _clean_model_dict(u["evidence"], Evidence)
                cleaned_u.append(u)
            params["hubbard_u"] = cleaned_u
        ev = params.get("evidence")
        if isinstance(ev, list):
            params["evidence"] = [
                _clean_model_dict(e, Evidence) if isinstance(e, dict) else e for e in ev
            ]
        item["parameters"] = params
    surf = item.get("surface")
    if isinstance(surf, dict):
        surf = _clean_model_dict(surf, SurfaceModel)
        ev = surf.get("evidence")
        if isinstance(ev, list):
            surf["evidence"] = [
                _clean_model_dict(e, Evidence) if isinstance(e, dict) else e for e in ev
            ]
        item["surface"] = surf
    ev = item.get("evidence")
    if isinstance(ev, list):
        item["evidence"] = [
            _clean_model_dict(e, Evidence) if isinstance(e, dict) else e for e in ev
        ]
    return item

_SYSTEM_PROMPT = """你是计算化学文献参数抽取器。任务：从论文的"方法段 + 相关表格/表注"中抽取 DFT 计算参数，输出严格的 JSON。

必须遵守：
1. 只输出一个 JSON 对象：{"calculations": [...]}，不要输出任何解释或其他文本。
2. 按"一次计算一个对象"拆分：bulk 与 slab、不同表面、不同泛函、不同体系各是一条 calculation；一篇论文可以有 1 到多条。
3. 只使用输入文本中明确出现的参数；没有出现的字段一律用 null；禁止编造数值。
4. 单位：能量统一为 eV（原文是 Ry 时按 1 Ry = 13.6057 eV 换算）；长度统一为 Å。
5. k 点：原文给出显式网格（如 3×3×1、Monkhorst-Pack 5×5×5、Gamma 1×1×1）→ kpoints.mesh 填三个整数；原文只给密度（如 0.03×2π Å⁻¹）→ kpoints 用 null。
6. hubbard_u：泛函为 PBE+U/DFT+U 且原文给出 U 数值 → 数组（元素/轨道/u_ev）；纯 PBE 无 U → 空数组 []；+U 但原文未给数值 → 空数组 []（不要编造常见值）。同一元素可有多条（不同格位），站点信息写进该条 evidence.quote。
7. functional_family：PBE → "PBE"；PBE+U（含 DFT+U）→ "PBE+U"；其他（HSE06、SCAN、B3LYP 等）→ "other"（functional 保留归一化展示名，照常抽取，不要跳过）。
8. evidence：尽量给字段级证据；quote 必须逐字复制自输入文本；field 用点路径（如 "parameters.encut_ev"）；source 用 "text"（方法段）、"table"（表格）、"caption"（表注）；page 用输入中的页码标记（没有则 null）。
9. surface：论文描述了表面模型才给；miller 是 Miller 指数三元组 [h,k,l]；layers 是论文表述的层数；vacuum_angstrom 是真空层厚度（Å）。
10. calculation_type：结构优化/弛豫 → "relax"；单点/电子结构计算 → "static"。

每个 calculation 元素的 JSON Schema（可省字段用默认值）：
{
  "system_formula": "字符串，如 Co3O4",
  "parameters": {
    "calculation_type": "relax" 或 "static",
    "functional": "字符串",
    "functional_family": "PBE" 或 "PBE+U" 或 "other",
    "hubbard_u": [{"element": "Co", "orbital": "d", "u_ev": 3.0, "j_ev": 0.0, "evidence": null}],
    "encut_ev": 数值 或 null,
    "kpoints": null 或 {"scheme": "gamma" 或 "monkhorst-pack" 或 "explicit",
                        "mesh": [n1, n2, n3],
                        "gamma_centered": true 或 false 或 null,
                        "raw": "原文写法"},
    "ediff": null, "ediffg": null, "spin_polarized": null, "vdw": null,
    "pseudopotential": null, "extra_incar": {}, "notes": null,
    "evidence": [{"source": "text", "page": 4, "quote": "逐字原文",
                  "field": "parameters.encut_ev", "confidence": null}]
  },
  "surface": null 或 {"miller": [h, k, l], "layers": 整数,
                      "vacuum_angstrom": 数值 或 null, "termination": null,
                      "fixed_layers": 整数 或 null, "dipole_correction": null,
                      "evidence": []},
  "evidence": [],
  "confidence": null
}"""


class LLMExtractor:
    """方法段/表格 → 结构化参数（构造同步、抽取异步、无状态）。"""

    def __init__(self, model: Any | None = None, *, max_input_chars: int = DEFAULT_MAX_INPUT_CHARS):
        if model is None:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model_name=settings.DEEPSEEK_MODEL,
                openai_api_key=settings.DEEPSEEK_API_KEY,
                openai_api_base=settings.DEEPSEEK_BASE_URL,
                temperature=0,
                max_tokens=8000,
                request_timeout=settings.LLM_TIMEOUT_SECONDS,
                model_kwargs={"response_format": {"type": "json_object"}},
                # 关闭推理模式：deepseek-v4-pro 默认开启 thinking，会吃满 max_tokens
                # 导致 finish_reason=length（实测 reasoning_tokens=8000）。
                # 必须走 extra_body（openai SDK 的 parse() 只认 extra_body 合并进请求体）
                extra_body={"thinking": {"type": "disabled"}},
            )
        self._model = model
        self._max_input_chars = max_input_chars

    async def extract(
        self,
        paper_id: str,
        method_text: str,
        tables: list[TextBlock],
        captions: list[TextBlock],
    ) -> list[ExtractedCalculation]:
        if not method_text.strip():
            raise ServiceError(
                2003, "METHOD_SECTION_NOT_FOUND",
                {"reason": "method_text 为空（扫描版/无方法段应在 PDFParser 层拦截）"},
            )
        input_text = self._build_input(method_text, tables, captions)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"论文 ID：{paper_id}\n\n{input_text}"),
        ]
        return await self._invoke_and_parse(messages)

    # ---- 输入拼装 ----

    def _build_input(
        self, method_text: str, tables: list[TextBlock], captions: list[TextBlock]
    ) -> str:
        scored = sorted(
            ((len(_DFT_KEYWORDS.findall(t.text)), t) for t in tables),
            key=lambda x: -x[0],
        )
        extras: list[tuple[str, TextBlock]] = (
            [("table", t) for _s, t in scored] + [("caption", c) for c in captions]
        )
        budget = self._max_input_chars
        parts: list[str] = []
        used = 0
        for kind, block in extras:
            chunk = f"[{kind.upper()} (page {block.page})]\n{block.text}\n"
            if used + len(chunk) > budget:
                logger.warning("input 超出预算，丢弃剩余表格/表注（%s 个）", len(extras) - len(parts))
                break
            parts.append(chunk)
            used += len(chunk)
        method_budget = budget - used
        if len(method_text) > method_budget:
            logger.warning("方法段 %d 字符超过剩余预算 %d，截断尾部", len(method_text), method_budget)
            method_text = method_text[:method_budget] + "\n[TRUNCATED]"
        return "[METHOD SECTION]\n" + method_text + "\n\n" + "".join(parts)

    # ---- 调用与校验 ----

    async def _invoke_and_parse(self, messages: list) -> list[ExtractedCalculation]:
        try:
            response = await self._model.ainvoke(messages)
        except Exception as exc:  # 超时/鉴权/限流等 API 异常
            raise ServiceError(5001, "LLM_UNAVAILABLE", {"reason": str(exc)[:200]}) from exc
        try:
            return self._parse_and_validate(response.content)
        except ServiceError as first_error:
            # 校验失败：错误摘要回传，最多修复重试 1 次
            fix_prompt = HumanMessage(
                content=(
                    "你的上一次输出未通过校验，错误如下：\n"
                    f"{first_error.details.get('reason', first_error.message)}\n"
                    "请修正错误，重新只输出 JSON 对象 {\"calculations\": [...]}。"
                )
            )
            try:
                response2 = await self._model.ainvoke(
                    messages + [AIMessage(content=response.content), fix_prompt]
                )
            except Exception as exc:
                raise ServiceError(5001, "LLM_UNAVAILABLE", {"reason": str(exc)[:200]}) from exc
            return self._parse_and_validate(response2.content)

    def _parse_and_validate(self, content: Any) -> list[ExtractedCalculation]:
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        text = _JSON_FENCE.sub("", content.strip()).strip()
        try:
            data = json.loads(text)
        except Exception as exc:
            raise ServiceError(2004, "EXTRACTION_FAILED", {"reason": f"输出不是合法 JSON: {exc}"}) from exc
        items = data.get("calculations") if isinstance(data, dict) else None
        if not isinstance(items, list) or not items:
            raise ServiceError(2004, "EXTRACTION_FAILED", {"reason": "calculations 缺失或为空"})
        calcs: list[ExtractedCalculation] = []
        for i, item in enumerate(items):
            try:
                calc = ExtractedCalculation.model_validate(_clean_item(item))
            except ValidationError as exc:
                summary = "; ".join(
                    f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()[:10]
                )
                raise ServiceError(
                    2004, "EXTRACTION_FAILED", {"reason": f"calculations[{i}] 校验失败: {summary}"}
                ) from exc
            # §3.4：functional 归一化 + family 由 normalizer 计算并写入。
            # 保留 LLM 的 +U 意图：LLM 可能写 functional="PBE" + family="PBE+U"
            # （或带 hubbard_u），归一化时不能把 +U 抹掉。
            raw_params = calc.parameters
            raw_family = raw_params.functional_family
            raw_name = normalize_functional(raw_params.functional)
            if raw_family is FunctionalFamily.PBE_U or raw_params.hubbard_u:
                name = "PBE+U" if raw_name in ("PBE", "PBE+U") else raw_name
                fam = FunctionalFamily.PBE_U
            else:
                name = raw_name
                fam = family_of(name)
            params = raw_params.model_copy(update={"functional": name, "functional_family": fam})
            calcs.append(calc.model_copy(update={"parameters": params}))
        logger.info("extract ok: calculations=%d", len(calcs))
        return calcs
