"""Dataset schema checks: fail early instead of silently dropping data."""

import pandas as pd

REQUIRED = {
    "orders": {
        "order_id", "customer_id", "order_status", "order_purchase_timestamp",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    },
    "items": {"order_id", "order_item_id", "seller_id", "price", "freight_value"},
    "customers": {"customer_id", "customer_unique_id", "customer_state"},
}


def check_columns(frame: pd.DataFrame, table: str) -> None:
    missing = REQUIRED[table] - set(frame.columns)
    if missing:
        raise ValueError(f"{table}: missing required columns: {sorted(missing)}")


def check_unique(frame: pd.DataFrame, columns: list[str], table: str) -> None:
    if frame[columns].isna().any().any():
        raise ValueError(f"{table}: null primary key in {columns}")
    if frame.duplicated(subset=columns).any():
        raise ValueError(f"{table}: duplicate primary key in {columns}")


def validate_sources(orders: pd.DataFrame, items: pd.DataFrame,
                     customers: pd.DataFrame) -> None:
    for name, frame in (("orders", orders), ("items", items), ("customers", customers)):
        check_columns(frame, name)
    check_unique(orders, ["order_id"], "orders")
    check_unique(customers, ["customer_id"], "customers")
    check_unique(items, ["order_id", "order_item_id"], "items")
    if not orders["customer_id"].isin(customers["customer_id"]).all():
        raise ValueError("orders: unmatched customer_id")
    if not items["order_id"].isin(orders["order_id"]).all():
        raise ValueError("items: unmatched order_id")
