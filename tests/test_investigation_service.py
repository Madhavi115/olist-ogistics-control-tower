"""No Ollama server or trained model is needed for these service tests."""

import numpy as np
from fastapi.testclient import TestClient

from olist_logistics import api, investigation_service


class FakeModel:
    def predict_proba(self, features):
        scores = features["item_count"].map({1: 0.2, 2: 0.8}).to_numpy()
        return np.column_stack([1 - scores, scores])


def order(order_id, count):
    return {
        "order_id": order_id,
        "purchase_timestamp": "2018-03-01T10:00:00",
        "estimated_delivery_date": "2018-03-20T00:00:00",
        "item_count": count,
        "total_price": 100.0,
        "total_freight": 10.0,
        "customer_state": "SP",
    }


def test_investigation_success(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())
    monkeypatch.setattr(
        investigation_service,
        "analyze_with_local_llm",
        lambda state: {
            "ai_analysis": (
                '{"assessment":"Order B is ranked highest for review.",'
                '"unknown_information":["Carrier status"],'
                '"recommended_actions":["Check tracking"]}'
            )
        },
    )
    # WORKFLOW was built before monkeypatch. Build another from patched nodes.
    monkeypatch.setattr(
        investigation_service,
        "WORKFLOW",
        investigation_service.build_service_graph(),
    )
    response = TestClient(api.app).post(
        "/investigate",
        json={"orders": [order("A", 1), order("B", 2)]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ranking"]["priority_count"] == 1
    assert body["verified_findings"][0]["order_id"] == "B"
    assert body["ai_status"] == "ok"
    assert "Order B" in body["ai_analysis"]["assessment"]


def test_investigation_ollama_fallback(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())
    monkeypatch.setattr(
        investigation_service,
        "analyze_with_local_llm",
        lambda state: {"ai_analysis": "Ollama unavailable"},
    )
    monkeypatch.setattr(
        investigation_service,
        "WORKFLOW",
        investigation_service.build_service_graph(),
    )
    response = TestClient(api.app).post(
        "/investigate",
        json={"orders": [order("A", 1), order("B", 2)]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ai_status"] == "unavailable"
    assert body["ai_analysis"] is None
    assert "PRIORITY ORDER: B" in body["verified_report"]
