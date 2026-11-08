请先不要写任何代码。你的任务是深入理解我下面描述的项目，并产出一份详细的《技术方案设计文档》，保存在项目根目录的 `docs/technical-design.md` 中。

## 1. 项目定位
这是一个面向计算化学领域的智能体（Agent）系统，同时服务两类用户：
- 文献读者：从PDF论文中结构化提取DFT计算参数（泛函、U值、ENCUT、k点、表面模型等）。
- 计算设置者：用自然语言描述计算任务，生成可审阅的VASP输入文件草案（INCAR/POSCAR/KPOINTS）。

核心差异化在于“计算参数语义层”：一套Pydantic Schema、一套领域规则库、一个参数卡片库，被两条工作流共享。

## 2. 技术栈（已在 requirements.txt 中锁定）
- Agent编排：LangGraph (1.0.9) + LangChain (1.2.10)，通过 langchain-openai 调 DeepSeek API。
- PDF解析：PyMuPDF。
- 向量检索：ChromaDB + sentence-transformers (BAAI/bge-small-zh-v1.5)。
- 数据库：SQLite + aiosqlite + SQLAlchemy[asyncio]。
- 材料科学：pymatgen (2025.4.16)，用 MPRelaxSet/MPStaticSet 生成VASP输入，用 MPRester 查Materials Project结构。
- Web框架：FastAPI + Uvicorn。

## 3. 架构分层（共6层，自上而下）
**接入层**：FastAPI路由。提供 `/chat`（自然语言入口）、`/documents/upload`（PDF上传，202+后台任务）、`/calc/draft`（支持JSON和multipart两种Content-Type）。统一响应格式（Pydantic ResponseModel），统一错误处理。

**编排层**：
- rule_router：基于关键词（如“论文/文献” vs “生成/INCAR”）进行快速路由。
- intent_split：轻量版，仅识别“另外/还有/同时/顺便”四个连接词，最多拆两个子请求串行处理。
- cache_check：文献侧命中后，先调 query_cards 查是否已有卡片，命中则直接返回，未命中才跑图。
- llm_fallback：规则未命中时，调一次轻量 LLM 分类（literature / calc / clarify）。
- slot_manager：读写 SQLite sessions 表的槽位状态。

**工作流层**：两张独立的LangGraph图。
- 文献图：START → parse_pdf（定位方法段，输出method_found? false → notify_missing → END）→ extract_params（LLM+Pydantic结构化抽取）→ validate_params（纯规则校验，输出errors/warnings/missing_fields三级）→(有 errors/warnings → human_review) → save_card → END。
- 计算图：START → collect_slots（支持user输入/card_id/session三种来源）→ capability_check（第一版仅支持relax/static × PBE/PBE+U，不支持则明确告知）→ fetch_structure（MPRester或用户上传CIF）→ generate_draft（pymatgen+领域覆盖）→ validate_draft（三级输出）→ human_confirm（修改走patch，不重新生成）→ output_files → END。

**能力层**（Tools / Services，纯函数，可独立测试）：
- PDFParser：输入 PDF 路径，输出方法段落文本、页码坐标、method_found 布尔值。
- LLMExtractor：输入方法段落文本，输出 DFTParameters 对象。
- RuleValidator：输入参数字典，输出 {errors, warnings, info, missing_fields}。
- VASPGenerator：输入结构信息和槽位参数，输出 INCAR / POSCAR / KPOINTS 文件内容。
- MPClient：封装 pymatgen.ext.matproj.MPRester，按化学式或 ID 获取结构，获取表面 slab。
- VectorSearcher：封装 ChromaDB 查询。
- query_cards（独立 Service）：支持按system/functional/paper_id/card_id过滤和聚合）。

**领域层（核心资产,纯 Python 代码）**：
- schemas.py：定义 DFTParameters（泛函、U值、ENCUT、k点等）、SurfaceModel（Miller 指数、层数、真空层等）、CalculationCard（参数卡片）等 Pydantic 模型。
- rules.py：领域规则引擎（纯规则，不调 LLM）。每条规则有唯一 ID（R1, R2...）和级别（error / warning / info）。核心规则包括：+U 必须配 U值（error）、表面计算真空层 ≥ 15 Å（warning）、Co 3d 的 U 值在 [2.0, 5.0] eV 区间（warning）等。
- normalizer.py：化学式归一化（如将 Co₃O₄ 统一为 Co3O4），供聚合查询使用。

**存储层**：SQLite三张表[papers、cards（必须包含 system_normalized 和 missing_fields 字段）、sessions]，ChromaDB collection（名为param_cards），本地文件系统：./data/papers/（PDF）、./data/structures/（缓存的 CIF）、./data/outputs/（生成的 VASP 文件）。

## 4. 关键设计约束（必须在文档中体现）
- 成本控制：parse_pdf只把“方法段落”送LLM，不送整篇PDF；路由优先走规则，LLM兜底是最后手段；嵌入模型本地跑。
- HitL克制：文献侧只在errors/warnings时触发人工审核；计算侧只在输出文件前确认。不搞“到处interrupt”。
- 能力边界清晰：第一版不支持多结构任务（如氧空位形成能），capability_check直接拒绝并建议替代方案。
- 共享语义层：文献侧和计算侧共用同一套 Pydantic Schema 和规则库，这是“非通用 Agent”的根基。
- 可降级：LLM挂了，路由规则仍能处理大部分请求；MP挂了，用户上传CIF仍能走。

## 5. 你需要在文档中产出的内容
请按以下结构撰写 `docs/technical-design.md`：
1. 项目概述与核心价值
2. 架构总览（含一张Mermaid分层架构图，覆盖上述6层及模块）
3. 领域层详细设计：Pydantic Schema完整定义、R1-R7规则的具体逻辑与输出格式
4. 能力层Service接口定义：每个Service的输入/输出类型和职责边界
5. 两张LangGraph图的State定义与节点流转（含条件边和Human-in-the-Loop暂停点）
6. 存储层表结构与ChromaDB collection设计
7. 接入层API契约（请求/响应示例，覆盖 `/chat`、`/documents/upload`、`/calc/draft`）
8. 开发顺序建议与模块依赖关系，列出每个阶段的可验证产出
9. 风险与降级方案

请先阅读项目根目录下的 `requirements.txt`，确认技术栈。如果你认为上述描述中有任何不一致或缺失，先在对话中指出，不要直接开始写文档。