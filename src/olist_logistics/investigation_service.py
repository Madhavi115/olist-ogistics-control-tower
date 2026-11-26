"""In-process LangGraph investigation service (no localhost HTTP recursion)."""

import json
from math import ceil
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import api
from .investigation_agent import generate_report, investigate_shipments
from .investigation_agent_llm import AIAnalysis, analyze_with_local_llm


class ServiceState(TypedDict, total=False):
    orders: list[dict]
    top_fraction: float
    ranking: dict
    findings: list[dict]
    ai_analysis: str
    verified_report: str
    structured_analysis: dict | None
    ai_status: str


def score_locally(state: ServiceState) -> dict:
    """Score in-process using the exact feature builder and model of the API."""
    orders = [api.OrderInput.model_validate(item) for item in state["orders"]]
    features = api.prepare_batch_features(orders)
    model = api.load_model()
    probabilities = model.predict_proba(features)[:, 1]

    indices = sorted(
        range(len(orders)),
        key=lambda index: (-float(probabilities[index]), index),
    )
    budget = ceil(len(orders) * state["top_fraction"])
    ranked = [
        {
            "rank": rank,
            "order_id": orders[index].order_id,
            "delay_risk_score": round(float(probabilities[index]), 4),
            "priority_investigation": rank <= budget,
        }
        for rank, index in enumerate(indices, start=1)
    ]
    return {
        "ranking": {
            "model": "random_forest_v3",
            "total_orders": len(orders),
            "priority_count": budget,
            "top_fraction": state["top_fraction"],
            "ranked_shipments": ranked,
        }
    }


def finish_report(state: ServiceState) -> dict:
    """Preserve verified facts and parse the locally generated JSON if valid."""
    verified = generate_report(state)["report"]
    raw_analysis = state.get("ai_analysis", "")
    try:
        analysis = AIAnalysis.model_validate(json.loads(raw_analysis))
        return {
            "verified_report": verified,
            "structured_analysis": analysis.model_dump(),
            "ai_status": "ok",
        }
    except (ValueError, TypeError):
        return {
            "verified_report": verified,
            "structured_analysis": None,
            "ai_status": "unavailable",
        }


def build_service_graph():
    """Use direct scoring, existing deterministic investigation, and local LLM."""
    graph = StateGraph(ServiceState)
    graph.add_node("score_locally", score_locally)
    graph.add_node("investigate", investigate_shipments)
    graph.add_node("local_llm", analyze_with_local_llm)
    graph.add_node("finish", finish_report)
    graph.add_edge(START, "score_locally")
    graph.add_edge("score_locally", "investigate")
    graph.add_edge("investigate", "local_llm")
    graph.add_edge("local_llm", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


WORKFLOW = build_service_graph()
