import pandas as pd
import pytest

from olist_logistics.pipeline import (
    build_data_quality_report,
    build_kpis,
    build_order_mart,
    find_missing_item_shipments,
)


def test_order_grain_and_item_aggregation(source_frames):
    mart = build_order_mart(*source_frames)
    assert len(mart) == 3
    assert mart["order_id"].is_unique
    first = mart.set_index("order_id").loc["o1"]
    assert first["item_count"] == 2
    assert first["seller_count"] == 2
    assert first["total_item_value"] == 30
    assert first["total_freight_value"] == 5


def test_calendar_day_late_label_and_eligible_denominator(source_frames):
    mart = build_order_mart(*source_frames)
    indexed = mart.set_index("order_id")
    assert bool(indexed.loc["o1", "late_delivery"])
    assert not bool(indexed.loc["o2", "late_delivery"])
    assert pd.isna(indexed.loc["o3", "late_delivery"])
    kpi = build_kpis(mart)
    assert kpi["total_orders"] == 3
    assert kpi["delivered_orders_with_valid_dates"] == 2
    assert kpi["late_orders"] == 1
    assert kpi["late_delivery_rate_pct"] == 50.0
    assert kpi["orders_without_item_rows"] == 1


def test_duplicate_order_ids_fail(source_frames):
    orders, items, customers = source_frames
    orders = pd.concat([orders, orders.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate primary key"):
        build_order_mart(orders, items, customers)


def test_missing_customer_fails(source_frames):
    orders, items, customers = source_frames
    with pytest.raises(ValueError, match="unmatched customer_id"):
        build_order_mart(orders, items, customers.iloc[:2])


def test_negative_freight_fails(source_frames):
    orders, items, customers = source_frames
    items.loc[0, "freight_value"] = -1
    with pytest.raises(ValueError, match="Negative"):
        build_order_mart(orders, items, customers)

def test_find_missing_item_shipments():
    orders = pd.DataFrame({
        "order_id": ["A101", "A102", "A103"],
        "order_delivered_carrier_date": [
            "2026-01-05",
            "2026-01-06",
            None
        ]
    })

    items = pd.DataFrame({
        "order_id": ["A101"]
    })

    result = find_missing_item_shipments(orders, items)

    assert len(result) == 1
    assert result.iloc[0]["order_id"] == "A102"


def test_build_data_quality_report():
    orders = pd.DataFrame({
        "order_id": ["A101", "A102", "A103", "A104"],
        "order_status": [
            "delivered",
            "shipped",
            "canceled",
            "delivered",
        ],
        "order_delivered_carrier_date": [
            "2026-01-05",
            "2026-01-06",
            None,
            "2026-01-08",
        ],
    })

    items = pd.DataFrame({
        "order_id": ["A101"],
    })

    result = build_data_quality_report(orders, items)

    assert result["total_orders"] == 4
    assert result["orders_without_items"] == 3
    assert result["missing_item_shipments"] == 2
    assert result["delivered_without_items"] == 1
