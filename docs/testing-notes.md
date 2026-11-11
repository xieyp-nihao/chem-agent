# 测试笔记

> 记录回归命令、耗时、标记口径与已知偶发问题。数据基于 2026-11-11 实测（211 用例基线：197 非 LLM + 14 LLM）。

## 回归命令

| 场景 | 命令 | 耗时 | 结果 |
|---|---|---|---|
| 日常回归（推荐） | `HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python -m pytest tests/ -m "not llm" -q` | 约 4 分 18 秒 | 197 passed |
| 全量回归（发版前） | `HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 python -m pytest tests/ -q` | 约 28 分钟 | 211 用例（197 非 LLM + 14 LLM；本次未跑全量） |
| 仅真实调用用例 | `HF_HUB_OFFLINE=1 python -m pytest tests/ -m "llm" -q -rs` | 约 7 分钟 | 14 用例（见下） |
| 前端构建 | `cd frontend && npm run build` | 约 6–7 秒 | 0 TS 错误，1481 modules |

全量回归慢的原因是 14 个真实网络/模型调用用例（MaaS 端点响应慢 + MP 网络往返 + 嵌入模型加载）。

## @pytest.mark.llm 标记（共 14 个，pytest.ini 注册）

| 文件 | 数量 | 用例 |
|---|---|---|
| tests/test_llm_extractor.py | 3 | test_eval_extraction_all_cases、test_pbe_plus_u_without_u_precondition、test_other_functional_family（真实 DeepSeek） |
| tests/test_services_online.py | 6 | MP×4（mp_id/formula/cache_hit/invalid_key）+ Chroma×2（upsert_and_search_real、querycards_chroma_integration——加载本地嵌入模型，慢，按慢用例标记） |
| tests/test_literature_workflow.py | 2 | test_review_approve_flow（真实 Chroma/模型）、test_full_chain_with_real_llm（真实 DeepSeek） |
| tests/test_calc_workflow.py | 2 | test_json_draft_with_mp_approve_writes_files（真实 MP）、test_natural_language_slots_with_real_llm（真实 DeepSeek + MP） |
| tests/test_orchestrator.py | 1 | test_real_deepseek（真实 llm_fallback） |

文献套件为 **2 个**（`test_review_approve_flow`、`test_full_chain_with_real_llm`），不是 1 个。

Chroma 相关标记共 3 个（services_online 2 + literature 1）：模型来自本地缓存，但加载重、且 huggingface_hub 可能发 HEAD 探测，按"慢用例跳过日常回归"口径标记。

## 已知偶发问题

- **test_json_draft_with_mp_approve_writes_files 偶发失败**：`-m "llm"` 全量跑时曾失败一次（MP 端点并发限流），单独重跑约 15 秒通过。**不是代码回归，是环境问题**——该用例历次单跑全绿。
- 处理建议：发版前先跑 `-m "not llm"`，再单独跑 `-m "llm"`；`-m "llm"` 有失败时先单独重跑失败用例，确认是环境波动还是真回归，再决定是否排查代码。

## 环境注意事项

- 运行含真实调用的用例需要 `.env` 的 DEEPSEEK_API_KEY / MP_API_KEY。
- 嵌入模型已缓存在本地；huggingface.co 不可达时，**启动服务和跑 pytest 都必须加 `HF_HUB_OFFLINE=1`**；否则 sentence-transformers 的 HEAD 探测会卡 60+ 分钟（本次第一次非 LLM 回归即卡住，加变量后 4m18 完成）。
- Windows 控制台输出中文建议 `PYTHONIOENCODING=utf-8`。
