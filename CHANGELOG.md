# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的结构，版本号按语义化版本（SemVer）管理。

## [Unreleased]

P1 收尾清单（状态跟踪）：

### Fixed
- retry 复用同一 `thread_id` 的 checkpoint 隐患：retry 前调用 `adelete_thread`，从零重跑（已修，详见 0.2.0）。
- 纯 PBE 生成时 pymatgen 预设残留 `LDAU` / `LDAUPRINT` 等七个标签：已剥除并加 warning（已修，详见 0.2.0）。

### Added
- `@pytest.mark.llm` 测试分层：14 个真实 LLM/MP/嵌入模型用例默认从日常回归跳过（已完成）。

### Known Issues
- 非 VASP（DMol3 / Quantum ESPRESSO）论文抽取失败率仍高，待参数体系映射。
- 压力路径演示（并发请求、多 worker、长任务）尚未实现。

## [0.3.0] - 2026-11-11

### Added
- F1 前端脚手架：Vite + React + TypeScript + Ant Design，统一 API 客户端与 `ApiError`。
- F1 健康检查页：展示 `/health` 的六个组件状态。
- F2 文献页：PDF 上传、状态轮询、卡片列表、审核 Modal。
- F2 后端扩展：`GET /documents/{paper_id}` 返回 `cards` 与 `review_request`（兼容保留 `card_ids`）。
- F3 计算页：参数表单、草案预览、patch、确认、输出文件下载。
- F4 助手页：统一对话入口，按意图展示路由结果并跳转文献/计算页。
- `frontend/package-lock.json` 锁定前端依赖版本。

### Changed
- README 补前端技术栈、启动命令、四页面、目录结构与测试基线。
- 技术方案、抽取报告、测试笔记、简历材料与录屏指南同步当前实现与当前库状态。

## [0.2.0] - 2026-11-10

### Fixed
- 抽取单条失败隔离：逐条校验，单条失败跳过并 warning，全部失败才抛 `EXTRACTION_FAILED`（2004）。
- 分数层数归一化：`surface.layers=7.5` 四舍五入为整数，并在 `parameters.notes` 记录原始值。
- R3 真空层 `None` 守卫：论文未报告真空层时不再比较 `None`。
- 纯 PBE 预设 +U 标签剥除：`LDAU` / `LDAUTYPE` / `LDAUL` / `LDAUU` / `LDAUJ` / `LMAXMIX` / `LDAUPRINT` 七个标签全部剥除。
- retry checkpoint 清理：retry 前删除旧线程，避免从 3 轮失败现场恢复。
- 审核重复卡片去重：跨轮已决索引持久化，重复 `approve` 被服务端跳过。

### Changed
- 抽取宽容清理：丢弃未知键；非必填且有默认值的字段 `null` 回退默认值。
- 抽取失败日志保留 Pydantic 校验摘要，便于归因。
- 更新种子论文抽取报告与测试笔记。

## [0.1.0] - 2026-11-08

### Added
- 六层架构：接入层、编排层、工作流层、能力层、领域层、存储层。
- 两条 LangGraph 工作流：文献抽取/审核/保存；计算槽位/能力检查/草案/确认/落盘。
- Human-in-the-Loop：文献审核与草案确认均通过 `interrupt` 暂停、`Command(resume)` 恢复。
- `AsyncSqliteSaver` 持久化 checkpoint，进程重启后可恢复。
- 文献侧：PDFParser、LLMExtractor、R1–R7 RuleValidator、QueryCards、Chroma 向量索引。
- 计算侧：MPClient、SurfaceBuilder、VASPGenerator（INCAR / POSCAR / KPOINTS）。
- FastAPI 路由：`/chat`、`/documents/upload`、`/documents/{paper_id}`、`/documents/{paper_id}/retry`、`/calc/draft`、`/files/{path}`、`/health`。
- SQLite 业务表 + Chroma + FileStore 路径安全防线。
- 测试体系：规则/存储/服务/编排/工作流测试、9 篇论文 fixtures、`eval_set.json`。
