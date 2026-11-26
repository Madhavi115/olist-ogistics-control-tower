"""Tests that run without Streamlit, the ML artifact, FastAPI, or Ollama."""

import pytest
import requests

from dashboard.dashboard_core import (
    DashboardError,
    format_ai_analysis,
    orders_from_csv,
    request_investigation,
    validate_investigation,
)

CSV = """order_id,purchase_timestamp,estimated_delivery_date,item_count,total_price,total_freight,customer_state
A,2018-03-01T10:00:00,2018-03-20T00:00:00,2,150,20,SP
B,2018-03-02T09:00:00,2018-03-08T00:00:00,1,50,10,RJ
"""


def test_csv_loads_two_orders():
    result = orders_from_csv(CSV)
    assert len(result) == 2
    assert result[0]["order_id"] == "A"
    assert result[0]["item_count"] == 2


def test_duplicate_ids_rejected():
    duplicate = CSV.replace("B,2018", "A,2018")
    with pytest.raises(DashboardError, match="duplicate"):
        orders_from_csv(duplicate)


def test_missing_columns_rejected():
    with pytest.raises(DashboardError, match="Missing CSV columns"):
        orders_from_csv("order_id\nA\n")


def test_ranking_is_sorted_and_priority_count_matches():
    response = {
        "ranking": {
            "total_orders": 2,
            "priority_count": 1,
            "ranked_shipments": [
                {"rank": 1, "order_id": "B", "delay_risk_score": 0.6, "priority_investigation": True},
                {"rank": 2, "order_id": "A", "delay_risk_score": 0.2, "priority_investigation": False},
            ],
        }
    }
    frame = validate_investigation(response)
    assert frame.iloc[0]["order_id"] == "B"


def test_invalid_ranking_is_rejected():
    response = {
        "ranking": {
            "total_orders": 2,
            "priority_count": 1,
            "ranked_shipments": [
                {"rank": 1, "order_id": "A", "delay_risk_score": 0.2, "priority_investigation": True},
                {"rank": 2, "order_id": "B", "delay_risk_score": 0.6, "priority_investigation": False},
            ],
        }
    }
    with pytest.raises(DashboardError, match="not ordered"):
        validate_investigation(response)


def test_local_ai_fallback_remains_visible():
    output = format_ai_analysis({"ai_status": "unavailable", "ai_analysis": None})
    assert output["available"] is False
    assert "unavailable" in output["assessment"]


def test_api_client_sends_correct_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "ranking": {
                    "total_orders": 1,
                    "priority_count": 1,
                    "ranked_shipments": [
                        {"rank": 1, "order_id": "A", "delay_risk_score": 0.4, "priority_investigation": True}
                    ],
                },
                "ai_status": "unavailable",
                "ai_analysis": None,
            }

    def fake_post(url, json, timeout):
        captured.update({"url": url, "body": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    result = request_investigation("http://127.0.0.1:8000/", [{"order_id": "A"}], 0.1)
    assert result["ranking"]["total_orders"] == 1
    assert captured["url"] == "http://127.0.0.1:8000/investigate"
    assert captured["body"]["top_fraction"] == 0.1
