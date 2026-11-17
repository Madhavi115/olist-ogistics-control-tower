# Phase 1 data contract

## Sources

- `orders`: `order_id` is unique; contains order status and lifecycle timestamps.
- `order_items`: unique (`order_id`, `order_item_id`); zero or more items per order.
- `customers`: unique `customer_id`; `customer_unique_id` identifies returning shoppers.

Reference: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## Output

`data/processed/order_mart.parquet` contains **one row per order** with customer state,
item counts, distinct seller count, order-item price and freight sums, and historical
fulfillment metrics. Orders with no item records remain in the output with missing
item aggregates. The system fails fast on invalid keys or bad numeric amounts.

`late_delivery` is nullable boolean: an order is eligible only if status is
`delivered`, purchase and delivery/estimate dates exist, and delivery is not before
purchase. Late means **actual delivery calendar date > estimated calendar date**.
An on-time delivery after midnight on the estimated date is not late.

## Limits and leakage

**This is a historical reporting mart, not the training feature set.** Actual
carrier handover, actual delivery, dispatch time, delivery days, late label, and
aggregates computed using the full dataset must NEVER be used as purchase-time
model features. Phase 2 builds time-aware features separately.

Freight is a recorded order-item charge, not an alternative carrier quote.
Dates are interpreted as Olist's provided naive timestamps; timezone is not
inferred. Metrics are historical, not live operational availability.
