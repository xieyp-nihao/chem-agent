
# 简历项目描述与面试问题预演

> 所有内容均对应代码/文档中的真实实现，无编造；文中给出关键文件与章节便于现场指认。
> 数据口径：211 用例基线（197 非 LLM + 14 LLM，2026-11-11 实测）；评测集为 5 篇手工标注论文、10 条期望计算。

## 第一部分：简历项目描述（三个版本）

### 版本 1：一句话（约 30 字，简历项目列表）

面向计算化学的领域智能体：读论文抽参数、按参数生成 VASP 输入草案。

### 版本 2：一段话（约 100 字，简历项目详情）

面向计算化学的领域智能体（Python 3.11 / FastAPI / LangGraph 1.0.9 / pymatgen / ChromaDB / DeepSeek；前端 React 18 / Vite 5 / TypeScript 5 / Ant Design 5 / React Router 6）。文献侧从论文 PDF 抽取 DFT 参数，经 R1–R7 规则校验与人工审核生成参数卡片；计算侧据此生成可审阅的 VASP 输入草案（确认/patch 后落盘）。两条工作流共享同一套 Schema 与规则库。211 用例基线（197 非 LLM + 14 LLM），评测集核心字段 precision/recall 1.00。

### 版本 3：详细版（约 300 字，面试自我介绍展开）

项目是一个面向计算化学的领域智能体，后端六层架构（接入/编排/工作流/能力/领域/存储），依赖方向严格自上而下；前端是独立 Vite SPA（React 18 + TypeScript + Ant Design），四页覆盖健康、文献、计算、助手。

三个亮点：
1. **领域语义层**：Pydantic Schema、R1–R7 纯规则引擎、化学式归一化作为两条工作流共享资产——文献侧抽出的参数卡片可直接作为计算侧的 card_id 输入，形成"读论文→用参数→生成输入"闭环；规则与 LLM 分工明确（LLM 只做非结构化抽取，校验/生成覆盖走确定性规则）。
2. **两条 LangGraph 工作流 + Human-in-the-Loop**：文献审核与草案确认都是 interrupt 暂停 + Command(resume) 恢复，checkpoint 存 SQLite，进程重启后可恢复；3 轮上限、patch 白名单（仅 INCAR 级字段，结构级修改被拒绝并提示重新提交）。
3. **降级矩阵**：LLM/MP/Chroma/checkpointer 各自故障有明确路径——规则路由不依赖 LLM、CIF 上传兜底 MP、SQLite 卡片保存不被 Chroma 失败阻断。

量化结果：211 用例基线（197 非 LLM + 14 个真实 LLM/MP 调用）；评测集 10 条期望计算上 encut/k 点/真空层/Miller/层数等核心字段 precision/recall 全 1.00；日常非 LLM 回归 197 用例 4m18；前端 4 页，`npm run build` 0 TS 错误。

## 第二部分：面试问题预演（14 问，每题"要点 + 关键位置 + 可能追问"）

### 架构类

**1. 为什么用 LangGraph 而不是 LangChain 的 Chain？**
- 要点：流程是显式状态机（上传→抽取→校验→审核→保存；槽位→能力检查→取结构→草案→确认→落盘），有条件边和"暂停等人"的 HitL；Chain 是线性执行模型，做不到任意点暂停 + 持久化恢复。实测 LangGraph 1.0.9 的三个关键机制：ainvoke 遇 interrupt 不抛异常而是返回 `__interrupt__` 状态、`Command(resume=payload)` 恢复且 `interrupt()` 返回 resume 值、pre-interrupt 的状态更新会持久化到下一次 interrupt 的 checkpoint。
- 关键位置：`app/workflows/literature/graph.py`、`app/workflows/calc/graph.py`。
- 追问：为什么不用自主 Agent 循环？→ 成本可控（LLM 只出现在抽取/兜底/槽位解析三处）、状态可审计（每步状态落 papers 表）、规则先行。

**2. 六层架构的分层依据是什么？**
- 要点：依赖单向自上而下（§2.2 依赖表）；领域层零外部依赖，是"共享语义层"的根基；能力层无状态、构造注入、可独立测试；工作流层只做图编排；接入层只做 HTTP 契约与统一 envelope。分层让每一层可以独立测试与替换（如 MPClient 换成 mp-api 不影响上层）。
- 关键位置：`docs/technical-design.md` §2.1/§2.2。
- 追问：为什么编排层和工作流层分开？→ 编排层处理"意图"（拆分/路由/兜底/缓存/槽位），工作流层处理"确定性执行"；意图逻辑可以独立演进而不动图。

**3. 两条工作流为什么共享同一套 Schema 和规则库？**
- 要点：文献抽出的参数卡片直接作为计算侧 card_id 输入（闭环）；抽取侧和生成侧用同一份校验规则，口径一致。反面案例真实发生过：SurfaceModel.vacuum 改 Optional 后 R3 直接比较 None 崩溃——正因为规则共享，bug 在一次扫描中暴露、一次修复即两侧生效。
- 关键位置：`app/domain/schemas.py`、`app/domain/rules.py`、`docs/technical-design.md` §3.5。
- 追问：Schema 演进怎么办？→ 卡片带 schema_version/rules_version，后续可迁移重校验。

### 实现类

**4. HitL 的 interrupt/resume 怎么实现？**
- 要点：节点内 `interrupt(review_request)`；服务端用 AsyncSqliteSaver 持久化；恢复走 `/chat` 的 action=review/confirm → `graph.ainvoke(Command(resume=payload), thread_id)`。踩过的坑：空 `resume={}` 会被当作无恢复值原样重抛；同一 session 已有暂停图用 `aget_state` 的 tasks 判 4003。
- 关键位置：`app/workflows/literature/nodes.py` 的 human_review、`app/api/chat.py`。
- 追问：为什么最多 3 轮？→ error 未清零时 approve 无效、只能 edit/reject，3 轮上限防止无限自环，超限置 failed（2005）。

**5. 进程重启后状态怎么恢复？**
- 要点：AsyncSqliteSaver 与业务表共用同一 SQLite 文件（§6.2.4），checkpoint 落库；重启恢复用同一文件新建 saver + 同一 thread_id 继续 resume。有专门测试：第一次 invoke 到 interrupt，关 saver 重开再 resume 成功。
- 关键位置：`tests/test_literature_workflow.py::TestCheckpointerRestart`。
- 追问：retry 有什么坑？→ 见第 10 问。

**6. R1–R7 规则引擎为什么不用 LLM？**
- 要点：规则是确定性的（可测试、零成本、可解释，输出 rule_id/字段路径/建议）；LLM 只做非结构化文本→结构化参数这一步。分工：抽取靠 LLM，校验与生成覆盖靠规则。真实效果：9 篇论文扫描中 R1（PBE+U 无 U 值）和 R4（Co U 超区间）各触发一次，均正确拦截。
- 关键位置：`app/domain/rules.py`、`docs/technical-design.md` §3.5.2。
- 追问：规则会不会漏判？→ R5 把必填缺失归入 missing_fields 兜底，再叠加人工审核。

**7. patch 白名单为什么只允许 INCAR 级字段？**
- 要点：INCAR 是标量参数，patch 后重新生成文本即可；miller/layers/vacuum 等结构级修改必须重新切 slab（POSCAR 会变），服务端直接 4002 拒绝并提示重新提交 /calc/draft，避免"改了结构但 POSCAR 没变"的不一致。patch 应用后重新跑生成器与校验（POSCAR 逐字节不变，因为结构未变）。
- 关键位置：`app/workflows/calc/nodes.py` 的 `_PATCH_WHITELIST`、human_confirm。

### 踩坑类

**8. 纯 PBE 请求带 +U 的 bug 怎么发现和修的？**
- 要点：写端到端 demo 真实跑通后读 INCAR 输出，发现纯 PBE 请求里出现 LDAU=3.32——根因是 pymatgen MPRelaxSet 对 Co3O4 自带 Materials Project 兼容 +U 参数，被生成逻辑原样继承。修复：family≠PBE+U 时从预设剥除七个标签（LDAU/LDAUTYPE/LDAUL/LDAUU/LDAUJ/LMAXMIX/LDAUPRINT）并加 warning；剥除而非设 False，避免残留空标签。带回归测试断言标签不存在。
- 关键位置：`app/services/vasp_generator.py`、`tests/test_services_offline.py::test_pbe_strips_preset_ldau`。
- 追问：为什么不是设 LDAU=False？→ 残留空标签会让审阅者困惑，剥除更干净。

**9. 审核重复卡片的 bug 怎么发现和修的？**
- 要点：demo 重跑时发现 2 条抽取产出 3 张卡、其中 2 张逐字段重复。根因：跨轮已决索引未持久化，客户端按 payload 全量提交 decisions 时已批准候选被二次批准。修复（方案 A）：approved/rejected 条目加 index、`prior_decided` 去重跳过重复决策、pending 排除跨轮已决（连带修了"跳过的候选永久 pending 导致自环"）。回归：1 条基础 + 3 条边界（全量含已决+reject、同轮重复 index、跨 3 轮含 error 候选）。
- 关键位置：`app/workflows/literature/nodes.py` human_review、`tests/test_literature_workflow.py::TestReviewDedup`。
- 追问：为什么不让客户端自己记住已决状态？→ payload 未标记已决，客户端无从知道；服务端幂等去重才是正解。

**10. retry 的 checkpoint 清理为什么用 adelete_thread 而不是版本化 thread_id？**
- 要点：papers=failed 后 retry，同一 thread_id 重新 invoke 会从旧 checkpoint 恢复（review_round=3 的失败现场）→ retry 永远失败。先探测 langgraph-checkpoint-sqlite 3.1.3 确认有 `adelete_thread`（方案 A），retry 前删除线程即可从零开始；方案 B（版本化 thread_id）需要新字段和迁移，不必要。安全性：retry 有 status 白名单（仅 failed/method_not_found），删除时无活跃任务，影响面仅目标线程。
- 关键位置：`app/api/documents.py` retry、`tests/test_literature_workflow.py::TestRetryResetsCheckpoint`。

### 边界类

**11. 9 篇种子论文 4 篇抽取失败，怎么解释？**
- 要点：诚实口径——5 篇成功（其中 2 篇触发审核、1 篇直通 saved）、4 篇失败。归因两类：DMol3 两篇是参数体系不匹配（抽取 prompt 面向 VASP）；La-Doped 与 Synergistic 是方法段过简（具体校验错误未落日志，标注"需重扫确认"而非猜测）。这是"输入不匹配"而非"抽取能力差"。已记录的改进方向：DMol3/QE 参数体系映射、EXTRACTION_FAILED 落错误摘要。
- 关键位置：`docs/paper-extraction-report.md`。
- 追问：评测集为什么只有 5 篇？→ 手工标注 ground truth 成本高，选择覆盖 bulk/slab/PBE/PBE+U/密度 k 点/多计算拆分的代表性论文。

**12. 第一版不支持什么，为什么？**
- 要点：能力边界前置到 `capability_check`：缺陷形成能/界面/能带/DOS/声子/AIMD/HSE06/SCAN 等关键词命中即拒，拒绝响应带替代建议。原则是"明确拒绝好过给出错结果"。非 VASP 论文可抽取但不承诺质量（已在 README 与抽取报告注明）。
- 关键位置：`app/workflows/calc/nodes.py` capability_check、`docs/technical-design.md` §1.3。

**13. 如果让你重构，会改哪里？**
- 要点（全部来自真实踩坑，不是泛泛而谈）：① retry/重传的 checkpoint 生命周期（已修，adelete_thread）；② EXTRACTION_FAILED 时把 Pydantic 校验错误摘要落日志，便于抽取失败归因（当前只记错误码）；③ Chroma 与 SQLite 的一致性补偿（SQLite 是事实源 + chroma_indexed 标记已设计，但 reindex 补偿脚本未实现）；④ 测试分层已做（llm 标记），后续可加并发场景。
- 追问：架构上最大的妥协是什么？→ 单进程假设（SQLite + BackgroundTasks + Chroma 持久客户端），多 worker 需要外部队列，第一版明确不做。

### 前端类

**14. Vite proxy / 统一 envelope / ApiError 如何串起四个页面？**
- 要点：`frontend/vite.config.ts` 把 `/api` 代理到 `http://127.0.0.1:8000`，并用 `rewrite` 去掉 `/api` 前缀（后端没有 `/api`）。`frontend/src/api/client.ts` 的 `request<T>()` 统一解析 `{code, message, data, request_id}`：`code=0` 返回 `data`，`code≠0` 抛 `ApiError(code, message, data, requestId)`。四个页面通过 `health.ts` / `documents.ts` / `calc.ts` / `chat.ts` 的类型化函数调用，最终都走 `request<T>()`；页面捕获 `ApiError`，用 `Alert` 展示 `code` 和 `message`，并用 `request_id` 排查。
- 关键位置：`frontend/vite.config.ts`、`frontend/src/api/client.ts`、`frontend/src/api/{health,documents,calc,chat}.ts`、`frontend/src/pages/`。
- 追问：`/health` 为什么特殊？→ 后端 down 时返回 HTTP 503 + envelope `code=0`，`getHealth()` 用 `allowNonOkEnvelope` 放行，以便页面展示红色 down。
