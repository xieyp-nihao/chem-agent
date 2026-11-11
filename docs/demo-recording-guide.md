# 端到端录屏操作指南（手动 Swagger 版）

> 目标：录一段 3–5 分钟的视频，展示"上传论文 → 审核 → 生成 VASP 草案 → 确认 → 下载 INCAR"完整链路。
> 本指南只操作运行中的服务与 Swagger UI，不改任何代码文件。
>
> 启动后端前先设置 `HF_HUB_OFFLINE=1`，否则 sentence-transformers 的 HEAD 探测会让启动卡数分钟。

## 0. 录屏前准备（必读）

### 0.1 一次性清理主库中的 Ni 论文记录（必做，否则录不到审核环节）

2026-11-11 核对：主库 `data/chem-agent.db` 里 Ni 尖晶石论文（paper_id `4a27a0de…`）当前是 `saved`，已有 2 张 `reviewed` 卡片。**重复上传同一 PDF 会命中"已存在"，直接返回 200 + saved，不会走到 needs_review**。录屏前需要清掉它（只是删数据，不碰代码）：

```bash
conda activate chem-agent
cd D:\App\biancheng\python\chem-agent
HF_HUB_OFFLINE=1 python - <<'EOF'
import sqlite3, asyncio
from app.storage.chroma import ChromaStore
from app.services.vector_searcher import VectorSearcher
from app.config import settings

TID = "lit:4a27a0dec3a925659cf5f7cdd98384ebc370c231291260159d8cf5dd735fd7a5"
con = sqlite3.connect("data/chem-agent.db")
card_ids = [r[0] for r in con.execute("SELECT card_id FROM cards WHERE paper_id=?", (TID,)).fetchall()]
con.execute("DELETE FROM cards WHERE paper_id=?", (TID,))
con.execute("DELETE FROM papers WHERE paper_id=?", (TID,))
con.execute("DELETE FROM checkpoints WHERE thread_id=?", (TID,))
con.execute("DELETE FROM writes WHERE thread_id=?", (TID,))
con.commit(); con.close()

async def clean():
    store = ChromaStore(settings.CHROMA_DIR, settings.EMBEDDING_MODEL)
    s = VectorSearcher(store)
    if s.is_available():
        for cid in card_ids:
            try:
                await s.delete(cid)
            except Exception:
                pass
asyncio.run(clean())
print("cleanup done, cards removed:", len(card_ids))
EOF
```

Ni 的 paper_id 已经是当前主库 ID（`4a27a0de…`），无需替换；保留 Ni 是因为它是唯一已证实触发 R4 warning×2、能稳定录到审核环节的候选。La-Doped 当前 `needs_review` 但没有卡片，审核候选/规则未验证，不作为默认剧本。

**备选方案（不想清理时）**：改用 `Pressure-Induced Phase Transitions in Co3O4 a First- principles Study.pdf`——它触发 R1 error（PBE+U 未抽到 U 值），审核环节 approve 会被拒，需要改为 reject 才能收尾。演示更有戏剧性，但比 Ni 多一步；且与第 5 步"approve 后看到 cards"的剧本不同。二选一即可，本指南默认用 Ni。

### 0.2 内存准备

后台服务此前两次因系统内存不足被回收。录屏前建议：
- 关掉 Chrome 多余标签页、其他大内存程序；
- 只保留：终端（跑服务）×1、终端（可选）×1、浏览器（Swagger）×1、录屏软件。

### 0.3 录屏软件

- **Win+G**（Xbox Game Bar）：按 Win+G → 捕获 → 开始录制；录完按 Win+G → 停止。最简单。
- **OBS**：新建"显示器采集"或"窗口采集"，开始/停止录制。
- 建议先试录 10 秒，确认画面和声音正常。

## 1. 启动服务（终端 1）

```bash
conda activate chem-agent
cd D:\App\biancheng\python\chem-agent
HF_HUB_OFFLINE=1 uvicorn app.main:app --reload --port 8000
```

**等待约 30 秒**，看到 `Application startup complete.` 和 `Uvicorn running on http://127.0.0.1:8000` 即就绪。
（`HF_HUB_OFFLINE=1` 必须有：huggingface 当前不可达时，不加它启动会卡几分钟。）

## 2. 预检：GET /health（浏览器）

打开浏览器 → 地址栏输入 `http://127.0.0.1:8000/docs` → 进入 Swagger UI。
找到绿色的 `GET /health` → 点开 → 点 **Try it out** → 点 **Execute**。

**确认看到**（Response body 里）：

```json
{ "data": { "type": "health", "status": "ok", "components": { "sqlite": "ok", "checkpointer": "ok", "llm": "ok", "mp": "ok", "chroma": "ok", "embedding_model": "ok" } } }
```

> 录屏提示：这一步可以口播"先看健康检查，六个组件全部 ok"。

## 3. 上传论文：POST /documents/upload

在 Swagger 找到 `POST /documents/upload` → **Try it out** → 点文件选择框，选：

```
D:\App\biancheng\python\chem-agent\tests\fixtures\papers\Formation, Electronic Structure, and Defects of Ni Substituted Spinel Cobalt Oxide-a DFT+U Study.pdf
```

→ 点 **Execute**。

**确认看到**：HTTP 202，Response body 里 `"status": "queued"` 和 `paper_id`（形如 `4a27a0de…`）。

**马上复制 paper_id**（Ctrl+C），下一步要用。

## 4. 轮询状态：GET /documents/{paper_id}

点开 `GET /documents/{paper_id}` → **Try it out** → 把 paper_id 粘贴进路径参数框 → **Execute**。

每隔 5–10 秒再点一次 Execute（Swagger 没有自动轮询）。**状态序列应为**：

```
queued → parsing → extracting → validating → needs_review
```

（真实 LLM 抽取约 20–40 秒；中间状态可能一闪而过，看到 needs_review 即可。）

**确认看到**：`"status": "needs_review"`、`"pending_action": "review"`；F2 后响应同时含 `review_request`（candidates）。若状态为 `saved`，则响应含 `cards`（卡片 brief 数组，`card_ids` 仍保留）。

## 5. 审核：POST /chat（action=review）

点开 `POST /chat` → **Try it out** → Request body 粘贴（把 paper_id 换成你的）：

```json
{
  "paper_id": "4a27a0de……（换成第 3 步拿到的）",
  "action": "review",
  "payload": { "decisions": [ { "index": 0, "decision": "approve" } ] }
}
```

→ **Execute**。

**预期**：Ni 论文有 2 个候选、都带 R4 warning（Co 的 U 值超出 [2.0, 5.0]），第 1 次 approve 后还有一个候选待决 → 返回 `{"type": "review_request", "round": 2, "candidates": [……]}`。**这一帧就是录屏重点**：口播"两个候选都触发了 R4 警告，进入人工审核"。

**再发一次**，把两个候选都 approve（index 0 已决会被服务端自动跳过）：

```json
{
  "paper_id": "4a27a0de……",
  "action": "review",
  "payload": { "decisions": [
    { "index": 0, "decision": "approve" },
    { "index": 1, "decision": "approve" }
  ] }
}
```

→ **Execute**。

**确认看到**：`{"type": "cards", "source": "extracted", "cards": [……]}`，2 张卡片。

## 6. 生成计算草案：POST /calc/draft

点开 `POST /calc/draft` → **Try it out** → Request body 粘贴：

```json
{
  "slots": {
    "calculation_type": "relax",
    "functional": "PBE",
    "functional_family": "PBE",
    "formula": "Co3O4",
    "mp_id": "mp-18748",
    "encut_ev": 520.0,
    "kpoints": { "scheme": "gamma", "mesh": [3, 3, 1], "gamma_centered": true },
    "surface": { "miller": [1, 1, 1], "layers": 6, "vacuum_angstrom": 15.0 }
  }
}
```

→ **Execute**。

**确认看到**：`{"type": "confirm_request", "session_id": "……", "round": 1, "draft": {"incar": "…", "poscar": "…", "kpoints": "…"}}`。
**复制 session_id**。

> 口播提示："草案已经生成，注意这是纯 PBE 请求，等下载后我们会看到 INCAR 里没有 LDAU。"

## 7. 确认草案：POST /chat（action=confirm）

回到 `POST /chat` → Request body 粘贴（换成你的 session_id）：

```json
{
  "session_id": "9e80a6f1-……（换成第 6 步拿到的）",
  "action": "confirm",
  "payload": { "decision": "approve" }
}
```

→ **Execute**。

**确认看到**：`{"type": "output_files", "files": [INCAR, POSCAR, KPOINTS, manifest.json]}`，每个带 `url` 和 `size`。

## 8. 下载 INCAR：GET /files/{path}

点开 `GET /files/{path}` → **Try it out** → path 框粘贴（换成你的 session_id）：

```
calc_9e80a6f1-……/INCAR
```

→ **Execute**。

**确认看到**：HTTP 200，Response body 是 INCAR 文本；**检查没有 LDAU/LDAUL/LDAUU/LDAUTYPE/LMAXMIX/LDAUPRINT**（纯 PBE 的预期），有 `ENCUT = 520.0`、`ISMEAR = 0`、`ISIF = 2`、`NSW = 100`。

> 这一帧是收尾重点：口播"纯 PBE 请求的 INCAR 里没有任何 +U 标签——这是 P0 修掉的预设残留问题"。

## 9. 收尾与停止录制

- Win+G 停止录制（或 OBS 停止）；
- 视频默认保存在 `此电脑 → 视频 → 捕获`（Xbox Game Bar）或 OBS 输出目录；
- 终端 1 按 Ctrl+C 停服务（可选，录屏可不包含）。

## 附：可选加分镜头

1. 在 Swagger 再点一次 `POST /documents/upload` 传同一 PDF → 返回 200 + `"status": "saved"`（演示重复上传去重）。
2. `GET /health` 再执行一次收尾，呼应开头。

## 验证清单（录屏前自查）

- [ ] GET /health 返回 status=ok（六组件 ok）
- [ ] 上传后轮询看到 queued → parsing → extracting → validating → needs_review
- [ ] needs_review 时 POST /chat action=review 返回 review_request（含 candidates）
- [ ] 审核完成后返回 cards（2 张）
- [ ] POST /calc/draft 返回 confirm_request（含 draft）
- [ ] action=confirm 返回 output_files（4 个文件）
- [ ] GET /files/calc_xxx/INCAR 返回 200，内容无 LDAU 标签
