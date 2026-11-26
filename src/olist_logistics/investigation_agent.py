
"""LangGraph workflow for investigating shipment delivery risk."""

import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langgraph.graph import END, START, StateGraph

API_URL = "http://127.0.0.1:8000/predict/batch"


class InvestigationState(TypedDict, total=False):
    orders: list[dict]
    ranking: dict
    findings: list[dict]
    report: str


def score_shipments(state: InvestigationState) -> dict:
    """Call the existing batch prediction API."""

    payload = {
        "top_fraction": 0.10,
        "orders": state["orders"],
    }

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            ranking = json.load(response)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(
            "Cannot reach the prediction API. "
            "Check that Uvicorn is running."
        ) from exc

    return {"ranking": ranking}


def investigate_shipments(state: InvestigationState) -> dict:
    """Inspect priority shipments using observed order facts."""

    original_orders = {
        order["order_id"]: order
        for order in state["orders"]
    }

    findings = []

    for shipment in state["ranking"]["ranked_shipments"]:
        if not shipment["priority_investigation"]:
            continue

        order = original_orders[shipment["order_id"]]

        purchase = datetime.fromisoformat(
            order["purchase_timestamp"]
        )
        estimate = datetime.fromisoformat(
            order["estimated_delivery_date"]
        )

        promised_days = (
            estimate - purchase
        ).total_seconds() / 86400

        findings.append(
            {
                "order_id": order["order_id"],
                "rank": shipment["rank"],
                "risk_score": shipment["delay_risk_score"],
                "customer_state": order["customer_state"],
                "item_count": order["item_count"],
                "promised_days": round(promised_days, 1),
                "priority": "INVESTIGATE",
                "recommendations": [
                    "Check current carrier tracking.",
                    "Verify warehouse dispatch status.",
                    "Review the promised delivery date.",
                ],
                "explanation": (
                    "Selected because the ML model ranked "
                    "this order in the top 10% of the batch. "
                    "The available data does not establish "
                    "the cause of any potential delay."
                ),
            }
        )

    return {"findings": findings}


def generate_report(state: InvestigationState) -> dict:
    """Produce a grounded investigation summary."""

    ranking = state["ranking"]
    findings = state["findings"]

    lines = [
        "OLIST LOGISTICS AI CONTROL TOWER",
        "Shipment Risk Investigation Report",
        "=" * 45,
        f"Orders analyzed: {ranking['total_orders']}",
        f"Priority investigations: {ranking['priority_count']}",
        f"Model: {ranking['model']}",
        "",
    ]

    for finding in findings:
        lines.extend(
            [
                f"PRIORITY ORDER: {finding['order_id']}",
                f"Risk score: {finding['risk_score']:.4f}",
                f"Customer state: {finding['customer_state']}",
                f"Items: {finding['item_count']}",
                f"Promised days: {finding['promised_days']}",
                f"Assessment: {finding['explanation']}",
                "",
                "Recommended investigation steps:",
            ]
        )

        for recommendation in finding["recommendations"]:
            lines.append(f"- {recommendation}")

        lines.append("")

    lines.append(
        "Note: Scores are not calibrated probabilities. "
        "Recommendations are checks to perform, not "
        "confirmed explanations of shipment delay."
    )

    return {"report": "\n".join(lines)}


def build_investigation_graph():
    """Define the LangGraph investigation workflow."""

    graph = StateGraph(InvestigationState)

    graph.add_node("score_shipments", score_shipments)
    graph.add_node(
        "investigate_shipments", investigate_shipments
    )
    graph.add_node("generate_report", generate_report)

    graph.add_edge(START, "score_shipments")
    graph.add_edge(
        "score_shipments", "investigate_shipments"
    )
    graph.add_edge(
        "investigate_shipments", "generate_report"
    )
    graph.add_edge("generate_report", END)

    return graph.compile()


def main():
    """Run a demonstration using five sample orders."""

    sample_orders = [
        {
            "order_id": order_id,
            "purchase_timestamp": purchase,
            "estimated_delivery_date": estimate,
            "item_count": count,
            "total_price": price,
            "total_freight": freight,
            "customer_state": state,
        }
        for order_id, purchase, estimate, count, price, freight, state in [
            (
                "ORDER-001",
                "2018-03-01T10:00:00",
                "2018-03-20T00:00:00",
                2, 150, 20, "SP",
            ),
            (
                "ORDER-002",
                "2018-03-02T09:00:00",
                "2018-03-08T00:00:00",
                4, 450, 60, "AM",
            ),
            (
                "ORDER-003",
                "2018-03-03T15:00:00",
                "2018-04-01T00:00:00",
                1, 50, 10, "RJ",
            ),
            (
                "ORDER-004",
                "2018-03-04T11:00:00",
                "2018-03-12T00:00:00",
                3, 300, 45, "BA",
            ),
            (
                "ORDER-005",
                "2018-03-05T08:00:00",
                "2018-03-30T00:00:00",
                1, 90, 15, "MG",
            ),
        ]
    ]

    workflow = build_investigation_graph()

    result = workflow.invoke({"orders": sample_orders})

    print("\n" + result["report"])

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "shipment_investigation.txt"
    report_path.write_text(
        result["report"] + "\n",
        encoding="utf-8",
    )

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
