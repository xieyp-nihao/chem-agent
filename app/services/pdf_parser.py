"""PDF 解析：定位方法段、提取相关表格/表注、判断扫描版/加密。

对应 docs/technical-design.md §4.2：
- 方法段识别正则（大小写不敏感）；命中后取到下一个同级标题之前；跨页段落合并
- 无明确标题时回退到前 60% 页面含 DFT 关键词的连续段落 + warning
- 表格用 page.find_tables()；表注取表格附近文本块；纳入条件 = 表格在方法段
  页范围或表注含 DFT 关键词
- 扫描版判断：>80% 页面可提取字符 < 50 → is_scanned=True（工作流层才映射
  SCANNED_PDF_UNSUPPORTED，本层只标记）
- 加密 PDF（needs_pass）→ ServiceError(1002, "INVALID_FILE")
- bbox 用 PyMuPDF point 坐标（左上角原点），格式 (x0, y0, x1, y1)

同步、无状态；不调 LLM、不写数据库。
同级标题检测说明：实测样例论文的标题与正文同字号，不能用字号判断；采用
行形启发式——短行（≤80 字符）+ 编号树（编号段在方法段编号子树内继续、同级
且编号更大的标题结束）+ 下一节关键词（results/conclusion/...）。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

import fitz
from pydantic import BaseModel, ConfigDict

from app.storage.files import ServiceError

_METHOD_HEADING = re.compile(
    r"\b(?:computational\s+(?:details|methods)|methods?|calculation\s+details|"
    r"dft\s+calculations|computational\s+setup|计算方法|计算细节|计算设置)\b",
    re.IGNORECASE,
)
_NUM_PREFIX = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(.*)$")
_NEXT_SECTION = re.compile(
    r"^(?:\s*\d+(?:\.\d+)*\.?\s*)?(?:results?|conclusion|discussion|acknowledg|"
    r"references?|summary|appendix|supplementary|实验结果|结果与讨论|结论|致谢|参考文献)",
    re.IGNORECASE,
)
_DFT_KEYWORDS = re.compile(r"VASP|DFT|PBE|cutoff|k-point|Hubbard", re.IGNORECASE)
_HEADING_MAX_LEN = 80
_HEADING_MAX_WORDS = 10
_METHOD_MAX_CHARS = 6000
_SCANNED_CHAR_THRESHOLD = 50
_SCANNED_PAGE_RATIO = 0.8
_CAPTION_GAP_PT = 60.0
_SENTENCE_END = ".!?;:"


class TextBlock(BaseModel):
    """文本块（§4.2）。"""

    model_config = ConfigDict(extra="forbid")

    text: str
    page: int          # 1-based
    bbox: tuple[float, float, float, float]
    kind: Literal["method", "table", "caption", "body"]
    table_index: int | None = None


class ParsedPaper(BaseModel):
    """PDFParser 输出（§4.2）。"""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    page_count: int
    is_scanned: bool
    method_found: bool
    method_text: str
    blocks: list[TextBlock]
    tables: list[TextBlock]
    captions: list[TextBlock]
    warnings: list[str]


def _number_of(line: str) -> str | None:
    """行首编号，如 '2.1.' → '2.1'；无编号返回 None。"""
    m = _NUM_PREFIX.match(line)
    return m.group(1).rstrip(".") if m else None


def _in_tree(number: str, prefix: str) -> bool:
    """编号是否在方法段标题的编号子树内（'2.1' 在 '2' 树内，'2.2' 不在）。"""
    return number == prefix or number.startswith(prefix + ".")


def _looks_like_heading(text: str, page_no: int) -> bool:
    """标题行形启发式：短行、词数少、不以句尾标点结尾。"""
    stripped = text.strip()
    if not stripped or len(stripped) > _HEADING_MAX_LEN:
        return False
    if len(stripped.split()) > _HEADING_MAX_WORDS:
        return False
    if stripped[-1] in _SENTENCE_END:
        return False
    return True


class PDFParser:
    """方法段 + 表格/表注解析（同步、无状态）。"""

    def parse(self, pdf_path: Path | str) -> ParsedPaper:
        path = Path(pdf_path)
        paper_id = hashlib.sha256(path.read_bytes()).hexdigest()

        doc = fitz.open(path)
        try:
            if doc.needs_pass:
                raise ServiceError(1002, "INVALID_FILE", {"reason": "encrypted PDF"})
            pages: list[list[dict]] = []   # 每页: [{"text", "bbox"}, ...]
            page_blocks: list[list[dict]] = []  # 每页: [{"text", "bbox"}, ...]（块级）
            low_char_pages = 0
            for page in doc:
                d = page.get_text("dict")
                lines: list[dict] = []
                blocks: list[dict] = []
                for b in d["blocks"]:
                    if b.get("type") != 0:
                        continue
                    block_texts: list[str] = []
                    for line in b["lines"]:
                        text = "".join(sp["text"] for sp in line["spans"]).strip()
                        if not text:
                            continue
                        lines.append({"text": text, "bbox": tuple(line["bbox"])})
                        block_texts.append(text)
                    if block_texts:
                        blocks.append({"text": " ".join(block_texts), "bbox": tuple(b["bbox"])})
                chars = sum(len(l["text"]) for l in lines)
                if chars < _SCANNED_CHAR_THRESHOLD:
                    low_char_pages += 1
                pages.append(lines)
                page_blocks.append(blocks)
            page_count = len(doc)

            is_scanned = page_count > 0 and (low_char_pages / page_count) > _SCANNED_PAGE_RATIO
            if is_scanned:
                return ParsedPaper(
                    paper_id=paper_id,
                    page_count=page_count,
                    is_scanned=True,
                    method_found=False,
                    method_text="",
                    blocks=[],
                    tables=[],
                    captions=[],
                    warnings=[f"扫描版 PDF：{low_char_pages}/{page_count} 页可提取字符 < {_SCANNED_CHAR_THRESHOLD}"],
                )

            heading_pos = self._find_heading(pages)
            warnings: list[str] = []
            if heading_pos is None:
                method_lines, first_page, last_page = self._fallback_keyword_run(pages)
                if not method_lines:
                    return ParsedPaper(
                        paper_id=paper_id, page_count=page_count, is_scanned=False,
                        method_found=False, method_text="", blocks=[], tables=[],
                        captions=[], warnings=["未定位到方法段，且前 60% 页面无 DFT 关键词段落"],
                    )
                warnings.append("未找到明确的方法段标题，回退为关键词连续段落")
            else:
                hp, hi = heading_pos
                method_lines, first_page, last_page = self._collect_section(
                    pages, hp, hi, warnings
                )

            method_text = "\n".join(l["text"] for l in method_lines)
            blocks: list[TextBlock] = []
            if method_text.strip():
                # 每页的方法段行合并为一个 method 块
                by_page: dict[int, list[dict]] = {}
                for l in method_lines:
                    by_page.setdefault(l["page"], []).append(l)
                for pno in sorted(by_page):
                    ls = by_page[pno]
                    xs = [v for l in ls for v in (l["bbox"][0], l["bbox"][2])]
                    ys = [v for l in ls for v in (l["bbox"][1], l["bbox"][3])]
                    blocks.append(TextBlock(
                        text="\n".join(l["text"] for l in ls),
                        page=pno,
                        bbox=(min(xs), min(ys), max(xs), max(ys)),
                        kind="method",
                    ))

            method_page_range = (first_page, last_page)
            tables, captions = self._extract_tables(doc, pages, page_blocks, method_page_range)
            blocks.extend(tables)
            blocks.extend(captions)

            return ParsedPaper(
                paper_id=paper_id,
                page_count=page_count,
                is_scanned=False,
                method_found=bool(method_text.strip()),
                method_text=method_text,
                blocks=blocks,
                tables=tables,
                captions=captions,
                warnings=warnings,
            )
        finally:
            doc.close()

    # ---- 标题定位 ----

    def _find_heading(self, pages: list[list[dict]]) -> tuple[int, int] | None:
        """返回 (页序号 0-based, 行序号)；无命中返回 None。

        候选行：正则命中 + 行形启发式 + （有编号 / 全大写 / 不在第 1 页）
        ——第 1 页摘要里的方法段关键词提及是最常见的误报源。
        """
        for pno, lines in enumerate(pages):
            for idx, line in enumerate(lines):
                text = line["text"]
                if not _looks_like_heading(text, pno + 1):
                    continue
                if not _METHOD_HEADING.search(text):
                    continue
                if _number_of(text) or text.isupper() or pno >= 1:
                    return pno, idx
        return None

    # ---- 方法段收集（同级标题边界） ----

    def _collect_section(
        self, pages: list[list[dict]], hp: int, hi: int, warnings: list[str]
    ) -> tuple[list[dict], int, int]:
        heading_text = pages[hp][hi]["text"]
        prefix = _number_of(heading_text)
        collected: list[dict] = []
        total_chars = 0
        last_page = hp
        for pno in range(hp, len(pages)):
            lines = pages[pno]
            start = hi if pno == hp else 0
            for idx in range(start, len(lines)):
                text = lines[idx]["text"]
                if self._is_boundary(text, prefix):
                    return collected, hp + 1, last_page + 1
                total_chars += len(text)
                if total_chars > _METHOD_MAX_CHARS:
                    warnings.append(f"方法段超过 {_METHOD_MAX_CHARS} 字符，已截断")
                    return collected, hp + 1, last_page + 1
                collected.append({"text": text, "bbox": lines[idx]["bbox"], "page": pno + 1})
            last_page = pno
        return collected, hp + 1, last_page + 1

    def _is_boundary(self, text: str, prefix: str | None) -> bool:
        """是否到达方法段边界（下一个同级标题 / 下一节关键词标题）。"""
        if not _looks_like_heading(text, 1):
            return False
        stripped = text.strip()
        # ① 下一节关键词行（编号在子树内则视为方法段内的子标题，继续）
        if _NEXT_SECTION.match(stripped):
            number = _number_of(stripped)
            return not (prefix is not None and number is not None and _in_tree(number, prefix))
        # ② 编号标题：不在方法段子树内且编号不早于方法段 → 边界
        #    （覆盖同级 "2.2"、上级 "3."、以及更深层级 "3.1.2" —— 都属于后续章节；
        #     编号回退的列表行如 "1." 继续收集）
        number = _number_of(stripped)
        if number is None or prefix is None:
            return False
        first_candidate, first_prefix = int(number.split(".")[0]), int(prefix.split(".")[0])
        if first_prefix > first_candidate:
            return False  # 编号早于方法段（如列表 "1."）→ 继续
        if _in_tree(number, prefix):
            return False  # 方法段子树内的子标题 → 继续
        return True

    # ---- 回退：关键词连续段落 ----

    def _fallback_keyword_run(self, pages: list[list[dict]]) -> tuple[list[dict], int, int]:
        limit = max(1, int(len(pages) * 0.6))
        best: list[dict] = []
        current: list[dict] = []
        for pno in range(limit):
            for line in pages[pno]:
                if _DFT_KEYWORDS.search(line["text"]):
                    current.append({"text": line["text"], "bbox": line["bbox"], "page": pno + 1})
                else:
                    if len(current) > len(best):
                        best = current
                    current = []
            if len(current) > len(best):
                best = current
        if not best:
            return [], 0, 0
        return best, best[0]["page"], best[-1]["page"]

    # ---- 表格/表注 ----

    def _extract_tables(
        self,
        doc,
        pages: list[list[dict]],
        page_blocks: list[list[dict]],
        method_page_range: tuple[int, int],
    ) -> tuple[list[TextBlock], list[TextBlock]]:
        tables: list[TextBlock] = []
        captions: list[TextBlock] = []
        first, last = method_page_range
        for pno in range(len(doc)):
            page = doc[pno]
            found = page.find_tables()
            for ti, table in enumerate(found.tables):
                raw_rows = table.extract() or []
                # 空单元格在 PyMuPDF 提取结果中是 None，统一转空字符串
                rows = [[cell or "" for cell in (cells or [])] for cells in raw_rows]
                text = "\n".join(" | ".join(cells) for cells in rows)
                bbox = tuple(table.bbox)
                # 表注：表格下方 60pt 内、x 方向重叠的文本块
                caption_text = ""
                caption_bbox = bbox
                for blk in page_blocks[pno]:
                    x0, y0, x1, y1 = blk["bbox"]
                    overlap = min(x1, bbox[2]) - max(x0, bbox[0])
                    if (y0 >= bbox[3] and y0 - bbox[3] <= _CAPTION_GAP_PT
                            and overlap > 0.3 * (x1 - x0)):
                        caption_text = blk["text"]
                        caption_bbox = blk["bbox"]
                        break
                in_method_pages = first <= pno + 1 <= last
                caption_has_kw = bool(_DFT_KEYWORDS.search(caption_text))
                if in_method_pages or caption_has_kw:
                    tables.append(TextBlock(text=text, page=pno + 1, bbox=bbox,
                                            kind="table", table_index=ti))
                    if caption_text:
                        captions.append(TextBlock(text=caption_text, page=pno + 1,
                                                  bbox=caption_bbox, kind="caption"))
        return tables, captions
