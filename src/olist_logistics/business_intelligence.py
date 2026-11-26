"""Historical business intelligence; descriptive statistics, not causal estimates or ML features."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pipeline import build_order_mart, load_sources

REVIEWS_FILE = "olist_order_reviews_dataset.csv"


def read_business_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Read standard Olist CSVs; reviews are optional. No API/LLM dependency."""
    orders, items, customers = load_sources(raw_dir)
    mart = build_order_mart(orders, items, customers)
    reviews_path = raw_dir / REVIEWS_FILE
    reviews = pd.read_csv(reviews_path) if reviews_path.is_file() else None
    return mart, items, reviews


def monthly_sales(mart: pd.DataFrame) -> pd.DataFrame:
    """Item value, excluding freight; not net revenue or profit."""
    sales = mart.loc[mart["total_item_value"].notna() & mart["purchase_month"].notna()].copy()
    return (sales.groupby("purchase_month", as_index=False)
            .agg(orders=("order_id", "nunique"), item_value_brl=("total_item_value", "sum"),
                 freight_charged_brl=("total_freight_value", "sum"))
            .sort_values("purchase_month"))


def regional_performance(mart: pd.DataFrame) -> pd.DataFrame:
    """Late rate denominator is only delivered orders with valid dates."""
    base = mart.loc[mart["customer_state"].notna()].copy()
    base["eligible_delivery"] = base["late_delivery"].notna()
    base["late_order"] = base["late_delivery"].fillna(False).astype(int)
    out = (base.groupby("customer_state", as_index=False)
           .agg(orders=("order_id", "nunique"), eligible_deliveries=("eligible_delivery", "sum"),
                late_deliveries=("late_order", "sum"), item_value_brl=("total_item_value", "sum")))
    out["late_rate_pct"] = (100 * out["late_deliveries"] / out["eligible_deliveries"].replace(0, float("nan"))).round(2)
    return out.sort_values("orders", ascending=False).reset_index(drop=True)


def seller_performance(mart: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Seller exposure to order-level outcomes; multi-seller orders affect every participating seller."""
    links = items[["order_id", "seller_id", "price", "freight_value"]].copy()
    links["price"] = pd.to_numeric(links["price"], errors="raise")
    links["freight_value"] = pd.to_numeric(links["freight_value"], errors="raise")
    links = (links.groupby(["seller_id", "order_id"], as_index=False)
             .agg(item_value_brl=("price", "sum"), freight_charged_brl=("freight_value", "sum")))
    source = mart[["order_id", "late_delivery", "order_status"]]
    links = links.merge(source, on="order_id", how="left", validate="many_to_one")
    links["eligible_delivery"] = links["late_delivery"].notna()
    links["late_order"] = links["late_delivery"].fillna(False).astype(int)
    out = (links.groupby("seller_id", as_index=False)
           .agg(orders=("order_id", "nunique"), eligible_deliveries=("eligible_delivery", "sum"),
                late_deliveries=("late_order", "sum"), item_value_brl=("item_value_brl", "sum"),
                freight_charged_brl=("freight_charged_brl", "sum")))
    out["late_rate_pct"] = (100 * out["late_deliveries"] / out["eligible_deliveries"].replace(0, float("nan"))).round(2)
    return out.sort_values("orders", ascending=False).reset_index(drop=True)


def satisfaction_by_delivery(mart: pd.DataFrame, reviews: pd.DataFrame | None) -> pd.DataFrame:
    """Association, not causation. One mean score per reviewed order to avoid duplicate reviews."""
    columns = ["delivery_group", "reviewed_orders", "mean_review_score", "low_review_rate_pct"]
    if reviews is None:
        return pd.DataFrame(columns=columns)
    required = {"order_id", "review_score"}
    if not required.issubset(reviews.columns):
        raise ValueError(f"Review CSV missing columns: {sorted(required - set(reviews.columns))}")
    scores = reviews[["order_id", "review_score"]].copy()
    scores["review_score"] = pd.to_numeric(scores["review_score"], errors="coerce")
    scores = scores.loc[scores["review_score"].between(1, 5)]
    scores = scores.groupby("order_id", as_index=False)["review_score"].mean()
    joined = mart.loc[mart["late_delivery"].notna(), ["order_id", "late_delivery"]].merge(
        scores, on="order_id", how="inner", validate="one_to_one")
    if joined.empty:
        return pd.DataFrame(columns=columns)
    joined["delivery_group"] = joined["late_delivery"].map({True: "Late", False: "On time"})
    joined["low_review"] = joined["review_score"].le(2)
    out = joined.groupby("delivery_group", as_index=False).agg(
        reviewed_orders=("order_id", "nunique"),
        mean_review_score=("review_score", "mean"),
        low_review_rate_pct=("low_review", "mean"),
    )
    out["mean_review_score"] = out["mean_review_score"].round(2)
    out["low_review_rate_pct"] = (100 * out["low_review_rate_pct"]).round(2)
    return out[columns]


def business_summary(mart: pd.DataFrame) -> dict:
    valid_value = mart["total_item_value"].dropna()
    return {
        "orders": int(mart["order_id"].nunique()),
        "item_value_brl": float(valid_value.sum()),
        "average_order_item_value_brl": float(valid_value.mean()) if not valid_value.empty else None,
        "customer_states": int(mart["customer_state"].nunique()),
    }


def delivery_benchmark(mart: pd.DataFrame) -> float | None:
    """Observed late delivery rate, among orders with an eligible delivery outcome."""
    eligible = mart["late_delivery"].dropna()
    return float(eligible.astype(bool).mean()) if len(eligible) else None


def benchmark_performance(frame: pd.DataFrame, benchmark: float | None) -> pd.DataFrame:
    """Rank exposure groups by observed late deliveries in excess of a historical baseline.

    This is a prioritization heuristic, not a causal attribution or a statistical test.
    """
    out = frame.copy()
    if benchmark is None:
        out["expected_late_deliveries"] = float("nan")
        out["excess_late_deliveries"] = float("nan")
        out["late_rate_gap_pp"] = float("nan")
    else:
        out["expected_late_deliveries"] = (out["eligible_deliveries"] * benchmark).round(1)
        out["excess_late_deliveries"] = (
            out["late_deliveries"] - out["eligible_deliveries"] * benchmark
        ).round(1)
        out["late_rate_gap_pp"] = (
            out["late_rate_pct"] - 100 * benchmark
        ).round(2)
    return out


def sales_month_coverage(mart: pd.DataFrame) -> pd.DataFrame:
    """Flag first/last observed calendar months as boundary months.

    Boundary months may be incomplete; this check cannot establish completeness.
    """
    out = monthly_sales(mart)
    out["month_status"] = "interior month"
    if out.empty:
        return out
    purchase = pd.to_datetime(mart["order_purchase_timestamp"], errors="coerce").dropna()
    if purchase.empty:
        return out
    earliest, latest = purchase.min(), purchase.max()
    first, last = earliest.strftime("%Y-%m"), latest.strftime("%Y-%m")
    out.loc[out["purchase_month"].astype(str).isin([first, last]), "month_status"] = "boundary month — check coverage"
    out["first_observed_purchase"] = earliest.date().isoformat()
    out["last_observed_purchase"] = latest.date().isoformat()
    return out


def _reviewed_deliveries(mart: pd.DataFrame, reviews: pd.DataFrame | None) -> pd.DataFrame:
    if reviews is None:
        return pd.DataFrame(columns=["order_id", "late_delivery", "delay_days", "review_score", "low_review"])
    required = {"order_id", "review_score"}
    if not required.issubset(reviews):
        raise ValueError(f"Review CSV missing columns: {sorted(required - set(reviews.columns))}")
    scores = reviews[["order_id", "review_score"]].copy()
    scores["review_score"] = pd.to_numeric(scores["review_score"], errors="coerce")
    scores = scores[scores["review_score"].between(1, 5)]
    # Preserve earlier dashboard convention: average multiple valid reviews per order.
    scores = scores.groupby("order_id", as_index=False)["review_score"].mean()
    required_dates = {"order_delivered_customer_date", "order_estimated_delivery_date"}
    if not required_dates.issubset(mart):
        raise ValueError(f"Order mart missing columns: {sorted(required_dates - set(mart.columns))}")
    eligible = mart.loc[
        mart["late_delivery"].notna(),
        ["order_id", "late_delivery", "order_delivered_customer_date", "order_estimated_delivery_date"],
    ].copy()
    delivered = pd.to_datetime(eligible["order_delivered_customer_date"], errors="coerce").dt.normalize()
    estimated = pd.to_datetime(eligible["order_estimated_delivery_date"], errors="coerce").dt.normalize()
    eligible["delay_days"] = (delivered - estimated).dt.days.clip(lower=0)
    joined = eligible.merge(scores, on="order_id", how="inner", validate="one_to_one")
    joined["low_review"] = joined["review_score"].le(2)
    return joined


def review_severity(mart: pd.DataFrame, reviews: pd.DataFrame | None) -> pd.DataFrame:
    """Historical low-rating association by days later than the promised calendar date."""
    columns = ["delay_group", "reviewed_orders", "low_review_count", "share_of_low_reviews_pct", "mean_review_score", "low_review_rate_pct"]
    joined = _reviewed_deliveries(mart, reviews)
    if joined.empty:
        return pd.DataFrame(columns=columns)
    joined["delay_group"] = pd.cut(
        joined["delay_days"], bins=[-1, 0, 3, 7, float("inf")],
        labels=["On time / early", "1–3 days late", "4–7 days late", "8+ days late"],
    )
    out = joined.groupby("delay_group", observed=True).agg(
        reviewed_orders=("order_id", "nunique"),
        low_review_count=("low_review", "sum"),
        mean_review_score=("review_score", "mean"),
        low_review_rate_pct=("low_review", "mean"),
    ).reset_index()
    total_low_reviews = int(out["low_review_count"].sum())
    out["low_review_count"] = out["low_review_count"].astype(int)
    out["share_of_low_reviews_pct"] = (
        100 * out["low_review_count"] / total_low_reviews
    ).round(2) if total_low_reviews else 0.0
    out["mean_review_score"] = out["mean_review_score"].round(2)
    out["low_review_rate_pct"] = (out["low_review_rate_pct"] * 100).round(2)
    return out[columns]


def business_findings(
    mart: pd.DataFrame,
    sellers: pd.DataFrame,
    regions: pd.DataFrame,
    reviews_by_delivery: pd.DataFrame,
    *,
    min_seller_deliveries: int = 100,
    min_region_deliveries: int = 100,
) -> list[str]:
    """Deterministic, descriptive findings only; no generated explanations or causal claims."""
    findings = []
    benchmark = delivery_benchmark(mart)
    if benchmark is not None:
        findings.append(f"Overall late-delivery benchmark: {benchmark:.2%} of eligible deliveries.")
    if not reviews_by_delivery.empty:
        r = reviews_by_delivery.set_index("delivery_group")
        if {"Late", "On time"}.issubset(r.index):
            gap = float(r.loc["Late", "low_review_rate_pct"] - r.loc["On time", "low_review_rate_pct"])
            findings.append(f"Low-review prevalence is {gap:.2f} percentage points higher for late deliveries (historical association).")
    for label, data, threshold in [
        ("seller", sellers, min_seller_deliveries),
        ("customer state", regions, min_region_deliveries),
    ]:
        if benchmark is None or data.empty:
            continue
        ranked = benchmark_performance(data, benchmark)
        ranked = ranked[ranked["eligible_deliveries"] >= threshold]
        ranked = ranked[ranked["excess_late_deliveries"] > 0].sort_values("excess_late_deliveries", ascending=False)
        if not ranked.empty:
            top = ranked.iloc[0]
            identifier = top["seller_id"] if label == "seller" else top["customer_state"]
            findings.append(
                f"Highest excess late-delivery exposure among eligible {label}s: {identifier} "
                f"({top['late_deliveries']:,} late / {top['eligible_deliveries']:,} eligible; "
                f"{top['excess_late_deliveries']:+.1f} versus baseline expectation). "
                "Investigate; this does not establish responsibility."
            )
    return findings
