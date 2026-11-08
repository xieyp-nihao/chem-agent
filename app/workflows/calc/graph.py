"""计算图装配（§5.3.2 / §5.3.4）。

- thread_id = calc:{session_id}（只用于 checkpoint，不用作目录名）
- checkpointer 由调用方注入（lifespan 与文献图共用同一 saver）
- 条件边只读 state
"""
from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from app.workflows.calc import nodes
from app.workflows.calc.state import CalcState


def _after_collect(state: dict) -> str:
    if state.get("status") == "failed":
        return "notify_structure_failure"
    return "capability_check"


def _after_capability(state: dict) -> str:
    status = state.get("status")
    if status == "unsupported":
        return "notify_unsupported"
    if status == "missing_slots":
        return "notify_missing_slots"
    return "fetch_structure"


def _after_fetch(state: dict) -> str:
    if state.get("status") == "failed":
        return "notify_structure_failure"
    return "generate_draft"


def _after_generate(state: dict) -> str:
    if state.get("status") == "failed":
        return "notify_structure_failure"
    return "validate_draft"


def _after_validate(state: dict) -> str:
    if state.get("status") == "failed":
        return "notify_structure_failure"
    return "human_confirm"


def _after_confirm(state: dict) -> str:
    status = state.get("status")
    if status == "needs_confirm":
        return "human_confirm"  # 自环：仍有 error 且未满 3 轮
    if status == "approved":
        return "output_files"
    if status == "aborted":
        return "notify_aborted"
    return "notify_structure_failure"  # failed（含 patch 白名单违规/3 轮上限）


def build_calc_graph(checkpointer, services: nodes.CalcServices):
    g = StateGraph(CalcState)
    g.add_node("collect_slots", partial(nodes.collect_slots, services=services))
    g.add_node("capability_check", partial(nodes.capability_check, services=services))
    g.add_node("fetch_structure", partial(nodes.fetch_structure, services=services))
    g.add_node("generate_draft", partial(nodes.generate_draft, services=services))
    g.add_node("validate_draft", partial(nodes.validate_draft, services=services))
    g.add_node("human_confirm", partial(nodes.human_confirm, services=services))
    g.add_node("output_files", partial(nodes.output_files, services=services))
    g.add_node("notify_unsupported", partial(nodes.notify_unsupported, services=services))
    g.add_node("notify_missing_slots", partial(nodes.notify_missing_slots, services=services))
    g.add_node("notify_structure_failure", partial(nodes.notify_structure_failure, services=services))
    g.add_node("notify_aborted", partial(nodes.notify_aborted, services=services))

    g.add_edge(START, "collect_slots")
    g.add_conditional_edges(
        "collect_slots", _after_collect,
        {"capability_check": "capability_check",
         "notify_structure_failure": "notify_structure_failure"},
    )
    g.add_conditional_edges(
        "capability_check", _after_capability,
        {"notify_unsupported": "notify_unsupported",
         "notify_missing_slots": "notify_missing_slots",
         "fetch_structure": "fetch_structure"},
    )
    g.add_conditional_edges(
        "fetch_structure", _after_fetch,
        {"generate_draft": "generate_draft",
         "notify_structure_failure": "notify_structure_failure"},
    )
    g.add_conditional_edges(
        "generate_draft", _after_generate,
        {"validate_draft": "validate_draft",
         "notify_structure_failure": "notify_structure_failure"},
    )
    g.add_conditional_edges(
        "validate_draft", _after_validate,
        {"human_confirm": "human_confirm",
         "notify_structure_failure": "notify_structure_failure"},
    )
    g.add_conditional_edges(
        "human_confirm", _after_confirm,
        {"human_confirm": "human_confirm", "output_files": "output_files",
         "notify_aborted": "notify_aborted",
         "notify_structure_failure": "notify_structure_failure"},
    )
    for name in ("output_files", "notify_unsupported", "notify_missing_slots",
                 "notify_structure_failure", "notify_aborted"):
        g.add_edge(name, END)

    return g.compile(checkpointer=checkpointer)
