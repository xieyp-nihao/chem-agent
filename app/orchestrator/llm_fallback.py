"""LLM 兜底分类（§2.4 编排层约定）。

- 只调一次 DeepSeek，max_tokens 约 64（model.bind per-call 覆盖），超时 10s
- 复用 LLMExtractor()._model（同一 thinking-disabled 客户端，不新建）
- 输出 {"intent": "literature|calc|clarify", "reason": "..."}
- 失败 / 超时 / 非法 JSON / intent 不在枚举 → clarify；不缓存结果
"""
from __future__ import annotations

import asyncio
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

_VALID_INTENTS = {"literature", "calc", "clarify"}
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_SYSTEM_PROMPT = """你是意图分类器。把用户的请求分为三类，只输出一个 JSON 对象（不要输出任何其他文本）：
{"intent": "literature|calc|clarify", "reason": "一句话说明"}
- literature：用户要读取/提取/查询论文文献中的 DFT 参数
- calc：用户要生成/创建 VASP 输入文件（INCAR/POSCAR/KPOINTS）或描述计算任务
- clarify：无法判断或两者都涉及"""


async def llm_classify(model, message: str) -> str:
    bound = model.bind(max_tokens=64)  # per-call 覆盖共享客户端的 8000
    try:
        response = await asyncio.wait_for(
            bound.ainvoke([SystemMessage(content=_SYSTEM_PROMPT),
                           HumanMessage(content=message)]),
            timeout=10.0,
        )
    except Exception:
        return "clarify"
    content = response.content
    if isinstance(content, list):
        content = json.dumps(content, ensure_ascii=False)
    text = _JSON_FENCE.sub("", content.strip()).strip()
    try:
        data = json.loads(text)
        intent = data.get("intent")
    except Exception:
        return "clarify"
    return intent if intent in _VALID_INTENTS else "clarify"
