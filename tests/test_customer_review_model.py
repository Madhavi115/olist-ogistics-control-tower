"""Focused customer-review ML tests. All fixtures are synthetic."""
import numpy as np
import pandas as pd
import pytest

from olist_logistics.customer_review_model import (
    FEATURES, chronological_split, evaluate_ranked, make_review_cohort, train_and_evaluate,
)


def example_mart(n=100):
    dates = pd.date_range("2017-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "order_status": ["delivered"] * n,
        "order_purchase_timestamp": dates,
        "customer_state": ["SP", "RJ"] * (n // 2) + (["SP"] if n % 2 else []),
        "item_count": [1, 2] * (n // 2) + ([1] if n % 2 else []),
        "seller_count": [1] * n,
        "total_item_value": [float(10 + i % 20) for i in range(n)],
        "total_freight_value": [float(5 + i % 9) for i in range(n)],
        "late_delivery": [True] * n,
        "order_delivered_customer_date": dates + pd.Timedelta(days=8),
    })


def example_reviews(n=100):
    return pd.DataFrame({"order_id": [f"o{i}" for i in range(n)],
                         "review_score": [1 if i % 5 == 0 else 5 for i in range(n)]})


def test_no_outcome_leakage():
    data = make_review_cohort(example_mart(), example_reviews())
    assert set(FEATURES).issubset(data.columns)
    assert "late_delivery" not in data
    assert "order_delivered_customer_date" not in data
    assert "review_score" not in data
    assert data["low_review"].sum() == 20


def test_duplicate_review_policy():
    reviews = pd.concat([example_reviews(2),
                         pd.DataFrame([{"order_id": "o0", "review_score": 5}])], ignore_index=True)
    cohort = make_review_cohort(example_mart(2), reviews)
    assert cohort["order_id"].is_unique
    assert cohort.loc[cohort["order_id"] == "o0", "low_review"].iloc[0] == 0


def test_exclude_undelivered_orders():
    mart = example_mart(20)
    mart.loc[0, "order_status"] = "canceled"
    cohort = make_review_cohort(mart, example_reviews(20))
    assert len(cohort) == 19


def test_chronological_split():
    a, b, c = chronological_split(make_review_cohort(example_mart(), example_reviews()))
    assert (len(a), len(b), len(c)) == (60, 20, 20)
    assert a["order_purchase_timestamp"].max() < b["order_purchase_timestamp"].min()
    assert b["order_purchase_timestamp"].max() < c["order_purchase_timestamp"].min()


def test_ranked_metrics_known_answer():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, 0])
    score = evaluate_ranked(y, scores, .2)
    assert score["alert_count"] == 2
    assert score["detected_low_reviews"] == 1
    assert score["precision_at_10_pct"] == .5
    assert score["recall_at_10_pct"] == .5


def test_invalid_empty_reviews():
    with pytest.raises(ValueError):
        make_review_cohort(example_mart(4), pd.DataFrame(columns=["order_id", "review_score"]))


def test_pipeline_trains_and_reports():
    cohort = make_review_cohort(example_mart(100), example_reviews(100))
    model, metrics = train_and_evaluate(cohort)
    assert metrics["splits"] == {"train": 60, "validation": 20, "test": 20}
    assert metrics["selected_model"] in {"logistic_regression", "random_forest"}
    assert set(metrics["test"]) == {"dummy", "logistic_regression", "random_forest"}
    assert model.predict_proba(cohort[FEATURES].iloc[:3]).shape == (3, 2)
