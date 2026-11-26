import pandas as pd

from olist_logistics.business_intelligence import (
    business_summary, monthly_sales, regional_performance,
    satisfaction_by_delivery, seller_performance,
)


def sample_mart():
    return pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_state": ["SP", "SP", "RJ"],
        "total_item_value": [100.0, 50.0, 40.0],
        "total_freight_value": [10.0, 5.0, 4.0],
        "purchase_month": ["2018-01", "2018-01", "2018-02"],
        "late_delivery": pd.Series([True, False, pd.NA], dtype="boolean"),
        "order_status": ["delivered", "delivered", "canceled"],
    })


def test_summary_and_sales():
    m = sample_mart()
    assert business_summary(m)["item_value_brl"] == 190
    assert monthly_sales(m).iloc[0]["item_value_brl"] == 150


def test_regional_denominator_excludes_unknown():
    out = regional_performance(sample_mart()).set_index("customer_state")
    assert out.loc["SP", "late_rate_pct"] == 50
    assert pd.isna(out.loc["RJ", "late_rate_pct"])


def test_seller_counts_multiseller_orders_without_item_duplication():
    items = pd.DataFrame({"order_id": ["o1", "o1", "o2"],
                          "seller_id": ["s1", "s1", "s2"],
                          "price": [40.0, 60.0, 50.0],
                          "freight_value": [4.0, 6.0, 5.0]})
    out = seller_performance(sample_mart(), items).set_index("seller_id")
    assert out.loc["s1", "orders"] == 1
    assert out.loc["s1", "item_value_brl"] == 100
    assert out.loc["s1", "late_rate_pct"] == 100


def test_review_duplicate_averaged_per_order():
    reviews = pd.DataFrame({"order_id": ["o1", "o1", "o2", "o3"],
                            "review_score": [1, 1, 5, 1]})
    out = satisfaction_by_delivery(sample_mart(), reviews).set_index("delivery_group")
    assert out.loc["Late", "reviewed_orders"] == 1
    assert out.loc["Late", "low_review_rate_pct"] == 100
    assert out.loc["On time", "low_review_rate_pct"] == 0
