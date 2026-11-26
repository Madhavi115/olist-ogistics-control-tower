"""Business analytics regression tests; no Streamlit, API, or Ollama needed."""

import pandas as pd
import pytest

from olist_logistics.business_intelligence import (
    benchmark_performance,
    business_findings,
    delivery_benchmark,
    review_severity,
    sales_month_coverage,
    seller_performance,
)


def order_mart():
    return pd.DataFrame({
        "order_id": ["a", "b", "c", "d"],
        "late_delivery": pd.Series([False, True, True, pd.NA], dtype="boolean"),
        "order_status": ["delivered", "delivered", "delivered", "canceled"],
        "total_item_value": [10.0, 20.0, 30.0, 40.0],
        "total_freight_value": [2.0, 3.0, 4.0, 5.0],
        "purchase_month": pd.Series(["2017-01", "2017-02", "2017-02", "2017-03"], dtype="string"),
        "order_purchase_timestamp": pd.to_datetime([
            "2017-01-29", "2017-02-02", "2017-02-18", "2017-03-01"
        ]),
        "order_delivered_customer_date": pd.to_datetime([
            "2017-02-01", "2017-02-07", "2017-02-18", None
        ]),
        "order_estimated_delivery_date": pd.to_datetime([
            "2017-02-01", "2017-02-05", "2017-02-10", None
        ]),
    })


def test_benchmark_excludes_missing_outcomes():
    assert delivery_benchmark(order_mart()) == pytest.approx(2 / 3)
    assert delivery_benchmark(order_mart().iloc[3:]) is None


def test_multiple_items_and_multi_seller_exposure_do_not_duplicate_seller_order():
    items = pd.DataFrame({
        "order_id": ["a", "a", "a", "b"],
        "seller_id": ["seller1", "seller1", "seller2", "seller1"],
        "price": [3., 7., 5., 20.],
        "freight_value": [1., 1., 1., 2.],
    })
    sellers = seller_performance(order_mart(), items).set_index("seller_id")
    assert sellers.loc["seller1", "orders"] == 2
    assert sellers.loc["seller1", "late_deliveries"] == 1
    assert sellers.loc["seller1", "item_value_brl"] == 30.
    assert sellers.loc["seller2", "orders"] == 1
    assert sellers.loc["seller2", "item_value_brl"] == 5.


def test_excess_calculation_preserves_negative_and_positive_exposure():
    groups = pd.DataFrame({
        "seller_id": ["x", "y"], "eligible_deliveries": [100, 50],
        "late_deliveries": [20, 2], "late_rate_pct": [20., 4.],
    })
    out = benchmark_performance(groups, 0.10)
    assert out.loc[0, "excess_late_deliveries"] == 10.0
    assert out.loc[1, "excess_late_deliveries"] == -3.0
    assert out.loc[0, "late_rate_gap_pp"] == 10.0


def test_month_flags_boundary_months_without_claiming_incomplete():
    months = sales_month_coverage(order_mart())
    assert months.set_index("purchase_month").loc["2017-01", "month_status"].startswith("boundary")
    assert months.set_index("purchase_month").loc["2017-02", "month_status"] == "interior month"
    assert months.set_index("purchase_month").loc["2017-03", "month_status"].startswith("boundary")


def test_review_severity_uses_calendar_days_and_ignores_missing_review():
    reviews = pd.DataFrame({
        "order_id": ["a", "b", "c", "c", "d"],
        "review_score": [5, 1, 1, 3, 2],
    })
    groups = review_severity(order_mart(), reviews).set_index("delay_group")
    assert groups.loc["On time / early", "low_review_rate_pct"] == 0
    assert groups.loc["1–3 days late", "low_review_rate_pct"] == 100
    assert groups.loc["8+ days late", "low_review_rate_pct"] == 100
    assert groups["reviewed_orders"].sum() == 3


def test_reviews_missing_is_safe():
    assert review_severity(order_mart(), None).empty
    with pytest.raises(ValueError, match="missing columns"):
        review_severity(order_mart(), pd.DataFrame({"order_id": ["a"]}))


def test_business_findings_does_not_claim_responsibility():
    mart = order_mart()
    seller_groups = pd.DataFrame({
        "seller_id": ["s"], "eligible_deliveries": [3],
        "late_deliveries": [3], "late_rate_pct": [100.0],
    })
    region_groups = pd.DataFrame({
        "customer_state": ["SP"], "eligible_deliveries": [3],
        "late_deliveries": [3], "late_rate_pct": [100.0],
    })
    result = business_findings(mart, seller_groups, region_groups, pd.DataFrame(),
                               min_seller_deliveries=5, min_region_deliveries=5)
    assert len(result) == 1
    assert "benchmark" in result[0]


def test_review_impact_counts_and_percentages_are_exact():
    reviews = pd.DataFrame({
        "order_id": ["a", "b", "c", "d"],
        "review_score": [1, 2, 5, 1],
    })
    out = review_severity(order_mart(), reviews).set_index("delay_group")
    # Order d is not an eligible delivered order and cannot be counted.
    assert out["reviewed_orders"].sum() == 3
    assert out["low_review_count"].sum() == 2
    assert out.loc["On time / early", "low_review_count"] == 1
    assert out.loc["1–3 days late", "low_review_count"] == 1
    assert out["share_of_low_reviews_pct"].sum() == pytest.approx(100)


def test_review_impact_zero_low_reviews_no_divide_by_zero():
    reviews = pd.DataFrame({"order_id": ["a", "b", "c"], "review_score": [5, 4, 3]})
    out = review_severity(order_mart(), reviews)
    assert out["low_review_count"].sum() == 0
    assert (out["share_of_low_reviews_pct"] == 0).all()


def test_duplicate_review_rows_use_existing_per_order_mean_convention():
    reviews = pd.DataFrame({"order_id": ["a", "a", "b"], "review_score": [1, 5, 2]})
    out = review_severity(order_mart(), reviews).set_index("delay_group")
    assert out["reviewed_orders"].sum() == 2
    # Mean score of order a is 3, not low; b score 2 is low.
    assert out.loc["On time / early", "low_review_count"] == 0
    assert out.loc["1–3 days late", "low_review_count"] == 1
