"""chem-agent 端到端手动演示脚本（真实调用，不 mock）。

前置条件（脚本不自启服务）：
  1. 已配置 .env（DEEPSEEK_API_KEY / MP_API_KEY）
  2. 已初始化数据库：python scripts/init_db.py
  3. 已启动服务：uvicorn app.main:app --reload --workers 1

用法：
  python scripts/demo.py --base-url http://127.0.0.1:8000 [--paper <pdf 路径>]

流程：
  上传论文 PDF → 轮询状态 → （needs_review 时）审核 → 生成计算草案 → 确认 → 下载 INCAR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

POLL_INTERVAL_SECONDS = 5.0
POLL_TIMEOUT_SECONDS = 900.0   # 真实 LLM 抽取在慢端点下可能耗时数分钟

# /calc/draft 的完整槽位（与 §7.6 示例一致）
FULL_SLOTS = {
    "calculation_type": "relax",
    "functional": "PBE",
    "functional_family": "PBE",
    "formula": "Co3O4",
    "mp_id": "mp-18748",
    "encut_ev": 520.0,
    "kpoints": {"scheme": "gamma", "mesh": [3, 3, 1], "gamma_centered": True},
    "surface": {"miller": [1, 1, 1], "layers": 6, "vacuum_angstrom": 15.0},
}


def log(step: str, detail: str = "") -> None:
    print(f"[{step}] {detail}")


def _check(resp: httpx.Response, step: str) -> dict:
    """统一响应检查：HTTP 非 2xx 或 code != 0 即报错退出（不改服务端代码）。"""
    if resp.status_code >= 400:
        print(f"[{step}] HTTP {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)
    body = resp.json()
    if body.get("code") != 0:
        print(f"[{step}] code={body['code']} message={body['message']}: {json.dumps(body['data'], ensure_ascii=False)[:500]}")
        sys.exit(1)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="chem-agent 端到端演示")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--paper",
        default=None,
        help="论文 PDF 路径；默认 tests/fixtures/papers/ 下第一篇",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("chem-agent 端到端演示")
    print("前置：.env 已配置、python scripts/init_db.py 已执行、")
    print("      uvicorn app.main:app --reload --workers 1 已启动")
    print("=" * 64)

    t_start = time.perf_counter()
    client = httpx.Client(
        base_url=args.base_url,
        timeout=httpx.Timeout(300.0, connect=10.0),
    )

    # ---- 0. 健康预检 ----
    t0 = time.perf_counter()
    resp = client.get("/health")
    _check(resp, "health")
    health = resp.json()["data"]
    print(f"[health] status={health['status']} 耗时 {time.perf_counter()-t0:.1f}s")
    if health["status"] == "down":
        print("[health] 服务不可用，退出")
        sys.exit(1)

    # ---- 1. 上传论文 ----
    paper_path = Path(args.paper) if args.paper else sorted(
        (Path(__file__).parent.parent / "tests" / "fixtures" / "papers").glob("*.pdf")
    )[0]
    print(f"[upload] 论文：{paper_path.name}")
    t1 = time.perf_counter()
    with open(paper_path, "rb") as f:
        resp = client.post(
            "/documents/upload",
            files={"file": (paper_path.name, f.read(), "application/pdf")},
        )
    body = _check(resp, "upload")
    paper_id = body["data"]["paper_id"]
    print(f"[upload] paper_id={paper_id} status={body['data']['status']} "
          f"HTTP={resp.status_code} 耗时 {time.perf_counter()-t1:.1f}s")

    # ---- 2. 轮询文献状态 ----
    t2 = time.perf_counter()
    last_status = None
    while True:
        resp = client.get(f"/documents/{paper_id}")
        body = _check(resp, "status")
        data = body["data"]
        if data["status"] != last_status:
            print(f"[status] {data['status']} "
                  f"(pending_action={data.get('pending_action')})")
            last_status = data["status"]
        if data["status"] in ("saved", "needs_review"):
            break
        if data["status"] in ("failed", "method_not_found", "rejected"):
            print(f"[status] 终止状态 {data['status']} error_code={data.get('error_code')} "
                  f"error_msg={data.get('error_msg')}")
            sys.exit(1)
        if time.perf_counter() - t2 > POLL_TIMEOUT_SECONDS:
            print("[status] 轮询超时，退出")
            sys.exit(1)
        time.sleep(POLL_INTERVAL_SECONDS)
    print(f"[status] 轮询结束，耗时 {time.perf_counter()-t2:.1f}s")

    # ---- 3. 审核（needs_review 时） ----
    card_ids: list[str] = []
    if data["status"] == "needs_review":
        print("[review] 进入人工审核")
        decisions = [{"index": 0, "decision": "approve"}]  # 首轮盲审：candidates 从响应中获取
        for round_no in range(1, 4):
            t3 = time.perf_counter()
            resp = client.post("/chat", json={
                "paper_id": paper_id,
                "action": "review",
                "payload": {"decisions": decisions},
            })
            body = _check(resp, "review")
            review_data = body["data"]
            if review_data["type"] == "cards":
                card_ids = [c["card_id"] for c in review_data["cards"]]
                print(f"[review] 第 {round_no} 轮审核完成，卡片 {len(card_ids)} 张，"
                      f"耗时 {time.perf_counter()-t3:.1f}s")
                break
            if review_data["type"] == "review_request":
                candidates = review_data.get("candidates", [])
                print(f"[review] 第 {round_no} 轮返回 review_request（round="
                      f"{review_data.get('round')}），候选 {len(candidates)} 条，"
                      f"据此生成完整 decisions")
                decisions = [
                    {"index": c["index"],
                     "decision": (
                         "reject"
                         if any(f["severity"] == "error" for f in c["findings"])
                         else "approve"
                     )}
                    for c in candidates
                ]
                continue
            print(f"[review] 未知响应类型：{review_data['type']}")
            sys.exit(1)
        else:
            print("[review] 3 轮内未完成审核，退出")
            sys.exit(1)
    else:
        resp = client.get(f"/documents/{paper_id}")
        body = _check(resp, "status-final")
        card_ids = body["data"].get("card_ids", [])

    # ---- 4. 生成计算草案 ----
    t4 = time.perf_counter()
    resp = client.post("/calc/draft", json={"slots": FULL_SLOTS})
    body = _check(resp, "draft")
    draft_data = body["data"]
    session_id = draft_data["session_id"]
    print(f"[draft] type={draft_data['type']} session_id={session_id} "
          f"round={draft_data.get('round')} 耗时 {time.perf_counter()-t4:.1f}s")
    print(f"[draft] INCAR 片段（前 8 行）：")
    for line in draft_data["draft"]["incar"].splitlines()[:8]:
        print(f"        {line}")

    # ---- 5. 确认草案 ----
    t5 = time.perf_counter()
    resp = client.post("/chat", json={
        "session_id": session_id,
        "action": "confirm",
        "payload": {"decision": "approve"},
    })
    body = _check(resp, "confirm")
    confirm_data = body["data"]
    files = confirm_data.get("files", [])
    print(f"[confirm] type={confirm_data['type']} 文件 {len(files)} 个，"
          f"耗时 {time.perf_counter()-t5:.1f}s")

    # ---- 6. 下载 INCAR ----
    t6 = time.perf_counter()
    incar_url = f"/files/calc_{session_id}/INCAR"
    resp = client.get(incar_url)
    if resp.status_code != 200:
        print(f"[files] HTTP {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)
    print(f"[files] GET {incar_url} → {resp.status_code}，"
          f"耗时 {time.perf_counter()-t6:.1f}s")
    print("-" * 64)
    print("INCAR 内容（前 60 行）：")
    for line in resp.text.splitlines()[:60]:
        print(line)
    print("-" * 64)

    # ---- 7. 汇总 ----
    print("[summary]")
    print(f"  paper_id   : {paper_id}")
    print(f"  card_ids   : {card_ids}")
    print(f"  session_id : {session_id}")
    for f in files:
        print(f"  file       : {f['name']}  url={f['url']}  size={f['size']}B")
    print(f"  总耗时      : {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
