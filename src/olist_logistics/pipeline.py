"""One-row-per-order historical descriptive dataset, NOT an ML feature table."""

from pathlib import Path

import pandas as pd

from .schema import validate_sources

DATE_COLUMNS = [
    "order_purchase_timestamp", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
}


def load_sources(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [name for name in FILES.values() if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files in {raw_dir}: {', '.join(missing)}")
    orders = pd.read_csv(raw_dir / FILES["orders"], dtype={"order_id": "string", "customer_id": "string"})
    items = pd.read_csv(raw_dir / FILES["items"], dtype={"order_id": "string", "seller_id": "string"})
    customers = pd.read_csv(raw_dir / FILES["customers"], dtype={"customer_id": "string"})
    return orders, items, customers


def build_order_mart(orders: pd.DataFrame, items: pd.DataFrame,
                     customers: pd.DataFrame) -> pd.DataFrame:
    """Aggregate items BEFORE joining to avoid multiplying orders.

    Output contains delivery outcomes for historical reporting. It MUST NOT
    be passed directly as a model input: these outcomes are only known later.
    """
    validate_sources(orders, items, customers)
    orders = orders.copy()
    for col in DATE_COLUMNS:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    items = items.copy()
    for col in ("price", "freight_value"):
        items[col] = pd.to_numeric(items[col], errors="coerce")
    if items[["price", "freight_value"]].isna().any().any():
        raise ValueError("Non-numeric price or freight_value")
    if (items[["price", "freight_value"]] < 0).any().any():
        raise ValueError("Negative price or freight_value")

    item_metrics = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        seller_count=("seller_id", "nunique"),
        total_item_value=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
    )
    customer_attrs = customers[["customer_id", "customer_unique_id", "customer_state"]]
    mart = orders.merge(customer_attrs, on="customer_id", how="left", validate="many_to_one")
    mart = mart.merge(item_metrics, on="order_id", how="left", validate="one_to_one")
    if len(mart) != len(orders):
        raise AssertionError("Order grain changed during joins")

    mart["delivery_days"] = (
        mart["order_delivered_customer_date"] - mart["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    mart["dispatch_days"] = (
        mart["order_delivered_carrier_date"] - mart["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    # Olist promise is a calendar date (typically midnight); compare calendar dates.
    eligible = (
        mart["order_status"].eq("delivered")
        & mart["order_delivered_customer_date"].notna()
        & mart["order_estimated_delivery_date"].notna()
        & mart["order_purchase_timestamp"].notna()
        & mart["delivery_days"].ge(0)
    )
    mart["late_delivery"] = pd.Series(pd.NA, index=mart.index, dtype="boolean")
    mart.loc[eligible, "late_delivery"] = (
        mart.loc[eligible, "order_delivered_customer_date"].dt.normalize()
        > mart.loc[eligible, "order_estimated_delivery_date"].dt.normalize()
    ).astype(bool)
    mart["purchase_month"] = mart["order_purchase_timestamp"].dt.to_period("M").astype("string")
    return mart


def build_kpis(mart: pd.DataFrame) -> dict:
    """Use eligible deliveries as denominator, not every placed order."""
    eligible = mart[mart["late_delivery"].notna()]
    return {
        "total_orders": int(len(mart)),
        "delivered_orders_with_valid_dates": int(len(eligible)),
        "late_orders": int(eligible["late_delivery"].sum()),
        "late_delivery_rate_pct": (
            round(100 * float(eligible["late_delivery"].mean()), 2)
            if len(eligible) else None
        ),
        "average_delivery_days": (
            round(float(eligible["delivery_days"].mean()), 2)
            if len(eligible) else None
        ),
        "orders_without_item_rows": int(mart["item_count"].isna().sum()),
    }
def find_missing_item_shipments(orders, items):
    """
    Find orders with a carrier handover timestamp
    but no matching order-item records.
    """

    missing_orders = orders[
        ~orders["order_id"].isin(items["order_id"])
    ]

    flagged_orders = missing_orders[
        missing_orders["order_delivered_carrier_date"].notna()
    ]

    return flagged_orders.copy()

def build_data_quality_report(orders, items):
    """Calculate order-item data quality metrics."""

    missing_orders = orders[
        ~orders["order_id"].isin(items["order_id"])
    ]

    missing_item_shipments = find_missing_item_shipments(
        orders, items
    )

    delivered_without_items = missing_orders[
        missing_orders["order_status"] == "delivered"
    ]

    report = {
        "total_orders": int(len(orders)),
        "orders_without_items": int(len(missing_orders)),
        "missing_item_shipments": int(len(missing_item_shipments)),
        "delivered_without_items": int(len(delivered_without_items)),
    }

    return report
