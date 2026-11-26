"""Experimental purchase-time features and honest diagnostic comparison.

Run: python -m olist_logistics.customer_review_improvements --raw-dir data/raw
Does not overwrite the established customer-review model or its reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .customer_review_model import (
    FEATURES, NUMERIC, CATEGORICAL, TARGET, chronological_split,
    evaluate_ranked, make_review_cohort, _models,
)
from .pipeline import build_order_mart, load_sources

EXTRA_NUMERIC = ["freight_share", "mean_item_value", "is_weekend", "days_until_promised_date",
                 "seller_prior_orders_mean", "seller_prior_orders_max"]
EXTRA_CATEGORICAL = ["primary_product_category"]
EXTENDED_FEATURES = FEATURES + EXTRA_NUMERIC + EXTRA_CATEGORICAL


def seller_prior_counts(mart: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Past order *counts*, not future orders or historical review/delivery labels.

    Orders with identical timestamps get the same strictly-earlier count.
    Includes ALL historical placed orders in the reference log, not just reviewed orders.
    """
    if not {"order_id", "order_purchase_timestamp"}.issubset(mart.columns):
        raise ValueError("mart lacks order id or purchase timestamp")
    if not {"order_id", "seller_id"}.issubset(items.columns):
        raise ValueError("items lacks order id or seller id")
    orders = mart[["order_id", "order_purchase_timestamp"]].copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    links = items[["order_id", "seller_id"]].dropna().drop_duplicates()
    links = links.merge(orders, on="order_id", how="inner", validate="many_to_one")
    links = links.dropna(subset=["order_purchase_timestamp"])
    if links.empty:
        return pd.DataFrame(columns=["order_id", "seller_prior_orders_mean", "seller_prior_orders_max"])
    # timestamp groups ensure same-time orders cannot be counted as prior events.
    ordered = (links.groupby(["seller_id", "order_purchase_timestamp"], as_index=False)
               .agg(orders_at_timestamp=("order_id", "nunique"))
               .sort_values(["seller_id", "order_purchase_timestamp"]))
    ordered["prior_orders"] = (ordered.groupby("seller_id")["orders_at_timestamp"].cumsum()
                               - ordered["orders_at_timestamp"])
    links = links.merge(ordered[["seller_id", "order_purchase_timestamp", "prior_orders"]],
                        on=["seller_id", "order_purchase_timestamp"], how="left", validate="many_to_one")
    return (links.groupby("order_id", as_index=False)
            .agg(seller_prior_orders_mean=("prior_orders", "mean"),
                 seller_prior_orders_max=("prior_orders", "max")))


def primary_category(items: pd.DataFrame, products: pd.DataFrame | None) -> pd.DataFrame:
    """One stable category per order, selected by most item value (ties alphabetical)."""
    if products is None:
        return pd.DataFrame(columns=["order_id", "primary_product_category"])
    required_items = {"order_id", "product_id", "price"}
    required_products = {"product_id", "product_category_name"}
    if not required_items.issubset(items) or not required_products.issubset(products):
        raise ValueError("Product category inputs have missing required columns")
    product_map = products[["product_id", "product_category_name"]].drop_duplicates("product_id")
    part = items[["order_id", "product_id", "price"]].merge(product_map, on="product_id", how="left", validate="many_to_one")
    part["price"] = pd.to_numeric(part["price"], errors="coerce").fillna(0)
    part["product_category_name"] = part["product_category_name"].fillna("unknown").astype(str)
    grouped = part.groupby(["order_id", "product_category_name"], as_index=False)["price"].sum()
    grouped = grouped.sort_values(["order_id", "price", "product_category_name"], ascending=[True, False, True])
    return (grouped.drop_duplicates("order_id")[["order_id", "product_category_name"]]
            .rename(columns={"product_category_name": "primary_product_category"}))


def build_enriched_cohort(mart: pd.DataFrame, items: pd.DataFrame,
                          reviews: pd.DataFrame, products: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add only fields available at purchase (assuming posted promise is known)."""
    baseline = make_review_cohort(mart, reviews)
    attrs = mart[["order_id", "order_estimated_delivery_date", "order_purchase_timestamp"]].copy()
    attrs["order_estimated_delivery_date"] = pd.to_datetime(attrs["order_estimated_delivery_date"], errors="coerce")
    attrs["order_purchase_timestamp"] = pd.to_datetime(attrs["order_purchase_timestamp"], errors="coerce")
    attrs["days_until_promised_date"] = (
        attrs["order_estimated_delivery_date"].dt.normalize()
        - attrs["order_purchase_timestamp"].dt.normalize()
    ).dt.days.astype(float)
    attrs = attrs[["order_id", "days_until_promised_date"]]
    extended = baseline.merge(attrs, on="order_id", how="left", validate="one_to_one")
    extended = extended.merge(seller_prior_counts(mart, items), on="order_id", how="left", validate="one_to_one")
    extended = extended.merge(primary_category(items, products), on="order_id", how="left", validate="one_to_one")
    extended["primary_product_category"] = extended["primary_product_category"].fillna("unknown")
    total = extended["total_item_value"].fillna(0) + extended["total_freight_value"].fillna(0)
    extended["freight_share"] = (extended["total_freight_value"] / total.replace(0, np.nan)).clip(0, 1)
    extended["mean_item_value"] = extended["total_item_value"] / extended["item_count"].replace(0, np.nan)
    extended["is_weekend"] = extended["purchase_weekday"].isin([5, 6]).astype(float)
    return extended[["order_id", "order_purchase_timestamp", *EXTENDED_FEATURES, TARGET]]


def extended_estimator(model_type: str) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, NUMERIC + EXTRA_NUMERIC),
                              ("cat", categorical, CATEGORICAL + EXTRA_CATEGORICAL)])
    if model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=160, min_samples_leaf=12,
                                      max_features="sqrt", n_jobs=-1, random_state=42,
                                      class_weight="balanced_subsample")
    else:
        raise ValueError("Unknown model type")
    return Pipeline([("prep", prep), ("model", clf)])


def diagnostic_groups(frame: pd.DataFrame, scores: np.ndarray) -> list[dict]:
    """Test-set descriptive diagnostics; never use these for model selection."""
    data = frame[[TARGET, "total_item_value", "total_freight_value", "customer_state"]].copy()
    data["score"] = scores
    data["alert"] = False
    k = max(1, int(np.ceil(len(data) * 0.1)))
    data.iloc[np.argsort(-scores, kind="stable")[:k], data.columns.get_loc("alert")] = True
    data["value_band"] = pd.cut(data["total_item_value"], bins=[-np.inf, 50, 150, np.inf],
                                 labels=["<=50 BRL", "50–150 BRL", ">150 BRL"])
    rows = []
    for name, col in [("item_value", "value_band"), ("customer_state", "customer_state")]:
        for key, group in data.groupby(col, observed=True, dropna=False):
            if len(group) < 100:
                continue
            subset = group[group["alert"]]
            rows.append({"dimension": name, "group": str(key), "orders": len(group),
                         "low_review_rate": round(float(group[TARGET].mean()), 4),
                         "alerts": len(subset), "alert_precision": round(float(subset[TARGET].mean()), 4)
                         if len(subset) else None,
                         "missed_low_reviews": int(((group[TARGET] == 1) & ~group["alert"]).sum())})
    return rows


def compare_models(cohort: pd.DataFrame) -> dict:
    """Compare original and enriched feature sets on identical chronological rows.

    Selection is on validation AP only; the test set is diagnostic and previously
    exposed by Phase 2B, not an untouched evaluation for this iteration.
    """
    train, val, test = chronological_split(cohort)
    if train[TARGET].nunique() != 2:
        raise ValueError("Training window needs both classes")
    candidates = {
        "baseline_logistic": (_models()["logistic_regression"], FEATURES),
        "baseline_forest": (_models()["random_forest"], FEATURES),
        "enriched_logistic": (extended_estimator("logistic_regression"), EXTENDED_FEATURES),
        "enriched_forest": (extended_estimator("random_forest"), EXTENDED_FEATURES),
    }
    trained, validation = {}, {}
    for name, (model, fields) in candidates.items():
        model.fit(train[fields], train[TARGET])
        trained[name] = (model, fields)
        validation[name] = evaluate_ranked(val[TARGET], model.predict_proba(val[fields])[:, 1])
    selected = max(candidates, key=lambda key: (validation[key]["average_precision"] or -1))
    test_results = {}
    for name, (model, fields) in trained.items():
        scores = model.predict_proba(test[fields])[:, 1]
        test_results[name] = evaluate_ranked(test[TARGET], scores)
    model, fields = trained[selected]
    scores = model.predict_proba(test[fields])[:, 1]
    return {
        "selected_on_validation": selected,
        "selection_metric": "validation average precision",
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "validation": validation,
        "test_exploratory": test_results,
        "test_diagnostics": diagnostic_groups(test, scores),
        "features_enriched": EXTENDED_FEATURES,
        "notes": [
            "The Phase 2B test period was previously inspected. Treat these holdout comparisons as exploratory, not a fresh untouched confirmation.",
            "Seller history uses only counts from strictly earlier order timestamps (no review or delivery labels).",
            "Delivery promise is assumed available when purchase is placed. Verify real deployment timing.",
            "Only delivered reviewed orders have targets, creating selection bias.",
            "Scores are not verified as calibrated probabilities; diagnostics are descriptive, not causal.",
        ],
    }


def run(raw_dir: Path, report_path: Path) -> dict:
    orders, items, customers = load_sources(raw_dir)
    reviews_file = raw_dir / "olist_order_reviews_dataset.csv"
    if not reviews_file.is_file():
        raise FileNotFoundError(f"Missing {reviews_file}")
    products_file = raw_dir / "olist_products_dataset.csv"
    products = pd.read_csv(products_file) if products_file.is_file() else None
    mart = build_order_mart(orders, items, customers)
    reviews = pd.read_csv(reviews_file)
    cohort = build_enriched_cohort(mart, items, reviews, products)
    result = compare_models(cohort)
    result["product_category_source"] = "products CSV" if products is not None else "unknown (products CSV missing)"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental improvement analysis for low-review risk model")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--report", type=Path, default=Path("reports/customer_review_improvements.json"))
    args = parser.parse_args()
    result = run(args.raw_dir, args.report)
    print("Validation winner:", result["selected_on_validation"])
    print("Validation average precision:", {k: v["average_precision"] for k, v in result["validation"].items()})
    print("Exploratory test metrics:", json.dumps(result["test_exploratory"], indent=2))
    print("Saved diagnostics:", args.report)


if __name__ == "__main__":
    main()
