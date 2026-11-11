# 种子论文抽取结果报告（9 篇）

> 数据来源：P0 阶段对 `tests/fixtures/papers/` 下 9 篇论文的真实 DeepSeek 抽取扫描（只读、不落库）。
> 扫描时间点的代码版本：R3 真空层 None 守卫**修复前**、纯 PBE 剥除 +U **修复后**。
> 标注"未记录"的条目不臆测，需要时重跑扫描确认。
>
> ⚠️ 本文是 P0 阶段（F2 之前）的扫描快照，非当前库状态。当前库状态见文末「当前库状态」一节。

## 结果总表

| # | 论文（文件名缩写） | DFT 软件 | 抽取结果 | 审核触发 | 失败/异常说明 |
|---|---|---|---|---|---|
| 1 | HCHO catalytic oxidation over Pd-Co3O4 | VASP | 成功，3 calcs | 无 | — |
| 2 | Bulk and surface properties of spinel Co3O4 | **DMol3** | EXTRACTION_FAILED | — | LLM 输出校验失败（2004，修复重试后仍失败）；具体校验错误未记录，需重跑确认 |
| 3 | Facet-Dependent ORR on Co3O4 | **DMol3** | EXTRACTION_FAILED | — | 同上（LLM 输出校验失败，未记录细节） |
| 4 | OER on Co3O4 (Ir doping) | VASP | 异常（TypeError） | — | 命中 R3 对 None 真空层的比较崩溃——**当时为 app bug，已修复**；修复后的抽取成败未重扫，需重跑确认 |
| 5 | Ni Substituted Spinel (DFT+U) | Quantum ESPRESSO | 成功，2 calcs | **R4 warning ×2**（U_LR 6.7 / 6.6 超出 [2.0, 5.0] eV） | 论文报告的 U 值本身超区间，规则按预期告警 |
| 6 | Highly Water-Resistant La-Doped Co3O4 | 未确认 | EXTRACTION_FAILED | — | LLM 输出校验失败，具体错误未记录；方法段仅 1839 字符，可能过简，需重跑确认 |
| 7 | CO oxidation on (001)/(110) surfaces | Quantum ESPRESSO | 成功，4 calcs | 无 | 多计算拆分正确 |
| 8 | Pressure-Induced Phase Transitions | VASP | 成功，2 calcs | **R1 error**（calc1：PBE+U 静态计算未抽到 U 值） | 论文文本中 U=3.0 eV 未被 LLM 命中 |
| 9 | Synergistic adsorption on Co3O4(110) | 未确认 | EXTRACTION_FAILED | — | LLM 输出校验失败，具体错误未记录；方法段前段（6000 字符截断内）未见参数，需重跑确认 |

**统计**：成功 4 篇（其中 2 篇触发审核）、失败 4 篇、异常 1 篇（已随 R3 修复消除，待重扫）。

## 失败原因分析

按成因分两类：

1. **非 VASP 软件（2 篇）**：Bulk and surface、Facet-Dependent 均为 DMol3 计算（数值轨道基组、轨道截断半径等参数体系与 VASP 不同）。抽取 prompt 面向 VASP 参数体系优化，LLM 输出无法通过 Pydantic 校验。
2. **方法段过简/参数缺失（2 篇，未完全确认）**：La-Doped 方法段短（1839 字符）；Synergistic 的方法段前段（解析器 6000 字符截断内）未见参数描述。两篇的具体校验错误未记录——需重跑扫描确认是"方法段信息不足"还是"LLM 输出格式问题"。

**改进方向**：
- 增加 DMol3/QE 参数体系映射（轨道截断 → 对应字段，或明确标记"非 VASP，仅记录不生成"）；
- 放宽 LLM 输出的宽容度（已有清理层，可针对失败样本的具体校验错误迭代）；
- 记录每次 EXTRACTION_FAILED 的校验错误摘要（当前日志只记 2004 码），便于归因。

## 演示素材指引

| 场景 | 论文 | 触发 | 演示效果 |
|---|---|---|---|
| 审核 happy-path | Ni Substituted Spinel | R4 warning ×2 | upload → needs_review → approve → 2 张卡 saved |
| 审核 error-path | Pressure-Induced | R1 error | approve 被拒 → 必须 reject/edit 才能收尾（P1-6 可选演示） |
| 直通 saved | HCHO Pd-Co3O4 | 无 | 上传 → 抽取干净 → 直接保存（无审核） |

## 当前库状态（2026-11-11）

> 数据来源：只读查询 `data/chem-agent.db` 的 `papers` / `cards` 表。上面的 P0 结果表保持历史快照，状态以本表为准。

| paper_id（前 12 位） | 文件名 | 状态 | error_code | 说明 |
|---|---|---|---|---|
| `c6d1cca2164f` | A DFT and microkinetic study of HCHO catalytic oxidation mechanism over Pd-Co3O4 catalysts... | `saved` | — | 3 张卡 |
| `bbd83de32d98` | liu2020（001）.pdf | `saved` | — | **B 修复后已 saved**；4 张卡；不在原 9 篇 fixture 中 |
| `4a27a0dec3a9` | Formation, Electronic Structure, and Defects of Ni Substituted Spinel Cobalt Oxide-a DFT+U Study.pdf | `saved` | — | 2 张卡（reviewed） |
| `036d8e221080` | Highly Water-Resistant La-Doped Co3O4 Catalyst for CO Oxidation.pdf | `needs_review` | — | P0 表为 EXTRACTION_FAILED，当前已进入人工审核 |
| `410502f8a497` | Synergistic adsorption effect on Co3O4(110) surface to promote the ethanol.pdf | `failed` | 2004 | `EXTRACTION_FAILED: calculations 缺失或为空` |
| `bc1b9acdf844` | d5cp05056a.pdf | `failed` | 2005 | 第 3 轮审核后仍存在未解决的 error 候选（上限 3 轮） |
| `cc0558c53ddd` | Water interaction with perfect and fluorine-doped Co3O4 (100) surface.pdf | `saved` | — | 1 张卡 |
