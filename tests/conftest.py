import pandas as pd
import pytest


@pytest.fixture
def source_frames():
    orders = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_id": ["c1", "c2", "c3"],
        "order_status": ["delivered", "delivered", "canceled"],
        "order_purchase_timestamp": ["2018-01-01 10:00", "2018-01-01 10:00", "2018-01-01 10:00"],
        "order_delivered_carrier_date": ["2018-01-02", "2018-01-02", None],
        "order_delivered_customer_date": ["2018-01-04 12:00", "2018-01-03 18:00", None],
        "order_estimated_delivery_date": ["2018-01-03", "2018-01-03", "2018-01-03"],
    })
    items = pd.DataFrame({
        "order_id": ["o1", "o1", "o2"],
        "order_item_id": [1, 2, 1],
        "seller_id": ["s1", "s2", "s1"],
        "price": [10.0, 20.0, 5.0],
        "freight_value": [2.0, 3.0, 1.0],
    })
    customers = pd.DataFrame({
        "customer_id": ["c1", "c2", "c3"],
        "customer_unique_id": ["u1", "u2", "u3"],
        "customer_state": ["SP", "RJ", "MG"],
    })
    return orders, items, customers
