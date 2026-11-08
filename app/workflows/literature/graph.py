"""文献图装配（§5.2.2 / §5.2.4）。

- thread_id = lit:{paper_id}（只用于 checkpoint，不用作目录名）
- checkpointer 由调用方注入（lifespan 传 AsyncSqliteSaver，测试传 InMemorySaver）
- 条件边函数只读 state，不做 IO
"""
from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from app.workflows.literature import nodes
from app.workflows.literature.state import LiteratureState


def _after_parse(state: dict) -> str:
    if state.get("status") == "failed":
        return "notify_failure"
    if not state.get("method_found"):
        return "notify_missing"
    return "extract_params"


def _after_extract(state: dict) -> str:
    if state.get("status") == "failed":
        return "notify_failure"
    return "validate_params"


def _has_blocking(state: dict) -> str:
    from app.domain.schemas import ValidationReport

    for r in state.get("reports", []):
        report = ValidationReport.model_validate(r)
        if report.has_blocking_issues:
            return "human_review"
    return "save_cards"


def _after_review(state: dict) -> str:
    status = state.get("status")
    if status == "needs_review":
        return "human_review"  # 自环：仍有 error 且未满 3 轮
    if status == "approved":
        return "save_cards"
    if status == "rejected":
        return "notify_rejected"
    return "notify_failure"  # failed（含 3 轮上限）


def build_literature_graph(checkpointer, services: nodes.LiteratureServices):
    g = StateGraph(LiteratureState)
    g.add_node("parse_pdf", partial(nodes.parse_pdf, services=services))
    g.add_node("extract_params", partial(nodes.extract_params, services=services))
    g.add_node("validate_params", partial(nodes.validate_params, services=services))
    g.add_node("human_review", partial(nodes.human_review, services=services))
    g.add_node("save_cards", partial(nodes.save_cards, services=services))
    g.add_node("notify_missing", partial(nodes.notify_missing, services=services))
    g.add_node("notify_failure", partial(nodes.notify_failure, services=services))
    g.add_node("notify_rejected", partial(nodes.notify_rejected, services=services))

    g.add_edge(START, "parse_pdf")
    g.add_conditional_edges(
        "parse_pdf",
        _after_parse,
        {"notify_failure": "notify_failure", "notify_missing": "notify_missing",
         "extract_params": "extract_params"},
    )
    g.add_conditional_edges(
        "extract_params",
        _after_extract,
        {"notify_failure": "notify_failure", "validate_params": "validate_params"},
    )
    g.add_conditional_edges(
        "validate_params",
        _has_blocking,
        {"human_review": "human_review", "save_cards": "save_cards"},
    )
    g.add_conditional_edges(
        "human_review",
        _after_review,
        {"human_review": "human_review", "save_cards": "save_cards",
         "notify_rejected": "notify_rejected", "notify_failure": "notify_failure"},
    )
    for name in ("save_cards", "notify_missing", "notify_failure", "notify_rejected"):
        g.add_edge(name, END)

    return g.compile(checkpointer=checkpointer)
