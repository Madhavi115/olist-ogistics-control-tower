"""Synthetic checks for experimental customer-review features."""
import numpy as np
import pandas as pd
import pytest

from olist_logistics.customer_review_improvements import (
    EXTRA_NUMERIC, EXTENDED_FEATURES, build_enriched_cohort,
    compare_models, diagnostic_groups, primary_category, seller_prior_counts,
)


def fixtures(n=100):
    stamps = pd.date_range("2017-01-01", periods=n, freq="D")
    mart = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "order_purchase_timestamp": stamps,
        "order_estimated_delivery_date": stamps + pd.Timedelta(days=10),
        "order_status": ["delivered"] * n,
        "customer_state": ["SP", "RJ"] * (n // 2),
        "item_count": [1] * n, "seller_count": [1] * n,
        "total_item_value": [40.0 + i % 20 for i in range(n)],
        "total_freight_value": [10.0] * n,
        "late_delivery": [True] * n,
        "order_delivered_customer_date": stamps + pd.Timedelta(days=12),
    })
    items = pd.DataFrame({"order_id": mart.order_id, "seller_id": ["s1"] * n,
                          "product_id": ["p1", "p2"] * (n // 2), "price": mart.total_item_value})
    reviews = pd.DataFrame({"order_id": mart.order_id,
                            "review_score": [1 if i % 5 == 0 else 5 for i in range(n)]})
    products = pd.DataFrame({"product_id": ["p1", "p2"],
                             "product_category_name": ["books", "toys"]})
    return mart, items, reviews, products


def test_seller_prior_count_uses_only_past_orders():
    mart, items, _, _ = fixtures(4)
    counts = seller_prior_counts(mart, items).set_index("order_id")
    assert counts.loc["o0", "seller_prior_orders_mean"] == 0
    assert counts.loc["o3", "seller_prior_orders_max"] == 3


def test_same_timestamp_does_not_count_peer_orders():
    mart, items, _, _ = fixtures(4)
    mart.loc[1, "order_purchase_timestamp"] = mart.loc[0, "order_purchase_timestamp"]
    counts = seller_prior_counts(mart, items).set_index("order_id")
    assert counts.loc["o0", "seller_prior_orders_mean"] == 0
    assert counts.loc["o1", "seller_prior_orders_mean"] == 0
    assert counts.loc["o2", "seller_prior_orders_mean"] == 2


def test_duplicate_items_same_order_not_double_counted():
    mart, items, _, _ = fixtures(4)
    links = pd.concat([items, items.iloc[[0]]], ignore_index=True)
    counts = seller_prior_counts(mart, links).set_index("order_id")
    assert counts.loc["o1", "seller_prior_orders_mean"] == 1


def test_primary_category_selected_by_item_value():
    _, items, _, products = fixtures(4)
    more = pd.DataFrame([{"order_id": "o0", "seller_id": "s1", "product_id": "p2", "price": 1000.0}])
    cat = primary_category(pd.concat([items, more]), products).set_index("order_id")
    assert cat.loc["o0", "primary_product_category"] == "toys"


def test_product_category_optional():
    _, items, _, _ = fixtures(4)
    assert primary_category(items, None).empty


def test_enriched_features_available_and_no_outcomes():
    mart, items, reviews, products = fixtures()
    cohort = build_enriched_cohort(mart, items, reviews, products)
    assert set(EXTENDED_FEATURES).issubset(cohort)
    assert not {"review_score", "late_delivery", "order_delivered_customer_date"}.intersection(cohort)
    assert cohort.loc[0, "days_until_promised_date"] == 10
    assert cohort.loc[0, "seller_prior_orders_mean"] == 0


def test_enriched_cohort_without_products():
    mart, items, reviews, _ = fixtures()
    cohort = build_enriched_cohort(mart, items, reviews)
    assert cohort["primary_product_category"].eq("unknown").all()


def test_diagnostics_do_not_identify_people():
    mart, items, reviews, products = fixtures()
    cohort = build_enriched_cohort(mart, items, reviews, products)
    groups = diagnostic_groups(cohort, np.linspace(0, 1, len(cohort)))
    assert isinstance(groups, list)
    assert all("order_id" not in g for g in groups)


def test_comparison_returns_same_test_rows_for_all_models():
    mart, items, reviews, products = fixtures()
    cohort = build_enriched_cohort(mart, items, reviews, products)
    result = compare_models(cohort)
    assert set(result["test_exploratory"]) == {"baseline_logistic", "baseline_forest", "enriched_logistic", "enriched_forest"}
    assert {m["rows"] for m in result["test_exploratory"].values()} == {20}
    assert result["selected_on_validation"] in result["validation"]


def test_invalid_product_schema_rejected():
    _, items, _, _ = fixtures(4)
    with pytest.raises(ValueError):
        primary_category(items, pd.DataFrame({"wrong": [1]}))
