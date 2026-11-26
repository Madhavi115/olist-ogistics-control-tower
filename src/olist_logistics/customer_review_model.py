"""Purchase-time prediction of low customer-review ratings for historical Olist orders.

Usage: python -m olist_logistics.customer_review_model --raw-dir data/raw

Research prototype: cohort includes delivered orders with a valid review and
purchase timestamp. This creates selection bias and is NOT a causal model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .pipeline import build_order_mart, load_sources

NUMERIC = ["item_count", "seller_count", "total_item_value", "total_freight_value", "purchase_hour", "purchase_weekday", "purchase_month_number"]
CATEGORICAL = ["customer_state"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "low_review"


def make_review_cohort(mart: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """One record per delivered, reviewed order, using only purchase-time model features.

    Actual delivery dates, delivery statuses, review scores, review timestamps,
    and seller history computed from later orders must never enter FEATURES.
    """
    required = {"order_id", "review_score"}
    if not required.issubset(reviews):
        raise ValueError(f"Reviews missing columns: {sorted(required - set(reviews))}")
    mart_fields = {"order_id", "order_status", "order_purchase_timestamp", *NUMERIC[:4], "customer_state"}
    if not mart_fields.issubset(mart):
        raise ValueError(f"Order mart missing columns: {sorted(mart_fields - set(mart))}")
    if mart["order_id"].duplicated().any():
        raise ValueError("Order mart must contain exactly one row per order")

    score = reviews[["order_id", "review_score"]].copy()
    score["review_score"] = pd.to_numeric(score["review_score"], errors="coerce")
    score = score.loc[score["review_score"].between(1, 5)]
    # Match the existing Phase 1B business analysis policy for duplicate reviews.
    score = score.groupby("order_id", as_index=False)["review_score"].mean()

    base = mart.loc[mart["order_status"].eq("delivered"),
                    ["order_id", "order_purchase_timestamp", "customer_state", *NUMERIC[:4]]].copy()
    base["order_purchase_timestamp"] = pd.to_datetime(base["order_purchase_timestamp"], errors="coerce")
    base = base.dropna(subset=["order_purchase_timestamp"])
    base = base.merge(score, on="order_id", how="inner", validate="one_to_one")
    if base.empty:
        raise ValueError("No delivered orders with valid reviews and purchase dates")
    stamp = base["order_purchase_timestamp"]
    base["purchase_hour"] = stamp.dt.hour.astype(float)
    base["purchase_weekday"] = stamp.dt.dayofweek.astype(float)
    base["purchase_month_number"] = stamp.dt.month.astype(float)
    base[TARGET] = (base["review_score"] <= 2).astype(int)
    # No future outcomes retained in the model-design matrix.
    return base[["order_id", "order_purchase_timestamp", *FEATURES, TARGET]].sort_values(
        ["order_purchase_timestamp", "order_id"], kind="stable"
    ).reset_index(drop=True)


def chronological_split(cohort: pd.DataFrame, train_share: float = 0.60,
                        validation_share: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / validation / untouched test in chronological purchase order."""
    if not (0 < train_share < 1 and 0 < validation_share < 1 and train_share + validation_share < 1):
        raise ValueError("Invalid chronological split proportions")
    ordered = cohort.sort_values(["order_purchase_timestamp", "order_id"], kind="stable")
    n = len(ordered)
    i, j = int(n * train_share), int(n * (train_share + validation_share))
    if i < 1 or j <= i or j >= n:
        raise ValueError("Not enough orders for train/validation/test split")
    return ordered.iloc[:i].copy(), ordered.iloc[i:j].copy(), ordered.iloc[j:].copy()


def _preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", num, NUMERIC), ("state", cat, CATEGORICAL)])


def _models() -> dict[str, Pipeline | DummyClassifier]:
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline([("prep", _preprocessor()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))]),
        "random_forest": Pipeline([("prep", _preprocessor()),
            ("model", RandomForestClassifier(n_estimators=160, min_samples_leaf=12,
                                              max_features="sqrt", n_jobs=-1, random_state=42,
                                              class_weight="balanced_subsample"))]),
    }


def evaluate_ranked(y_true: pd.Series | np.ndarray, scores: np.ndarray,
                    top_fraction: float = 0.10) -> dict:
    """Evaluate discrimination and a fixed review-team alert capacity."""
    labels = np.asarray(y_true, dtype=int)
    predictions = np.asarray(scores, dtype=float)
    if len(labels) != len(predictions) or not len(labels):
        raise ValueError("Labels and scores must have same nonzero length")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be within (0, 1]")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(predictions).all():
        raise ValueError("Invalid binary labels or non-finite scores")
    k = max(1, int(np.ceil(len(labels) * top_fraction)))
    # Stable tie-handling: preserve input order for equal scores.
    selected = np.argsort(-predictions, kind="stable")[:k]
    found = int(labels[selected].sum())
    positives = int(labels.sum())
    prevalence = float(labels.mean())
    return {
        "rows": int(len(labels)), "low_reviews": positives, "prevalence": round(prevalence, 6),
        "roc_auc": round(float(roc_auc_score(labels, predictions)), 6) if len(np.unique(labels)) == 2 else None,
        "average_precision": round(float(average_precision_score(labels, predictions)), 6) if positives else None,
        "alert_count": k, "detected_low_reviews": found,
        "precision_at_10_pct": round(found / k, 6),
        "recall_at_10_pct": round(found / positives, 6) if positives else None,
        "lift_at_10_pct": round((found / k) / prevalence, 6) if prevalence else None,
    }


def train_and_evaluate(cohort: pd.DataFrame) -> tuple[Pipeline | DummyClassifier, dict]:
    train, val, test = chronological_split(cohort)
    # Both classes must exist in training for meaningful classification.
    if train[TARGET].nunique() != 2:
        raise ValueError("Training window must have both low and non-low reviews")
    metrics = {"prediction_time": "order purchase", "label": "mean valid review score <= 2",
               "cohort": "delivered orders with a valid review and purchase timestamp",
               "features": FEATURES, "selection_metric": "validation average precision",
               "alert_budget": "top 10% of scored orders",
               "splits": {"train": len(train), "validation": len(val), "test": len(test)},
               "date_ranges": {name: {"start": str(frame["order_purchase_timestamp"].min()),
                                      "end": str(frame["order_purchase_timestamp"].max())}
                               for name, frame in [("train", train), ("validation", val), ("test", test)]}}
    fitted = {}
    validations = {}
    for name, model in _models().items():
        model.fit(train[FEATURES], train[TARGET])
        score = model.predict_proba(val[FEATURES])[:, 1]
        validations[name] = evaluate_ranked(val[TARGET], score)
        fitted[name] = model
    # Dummy is a reference, not a candidate for model selection.
    candidate_names = ["logistic_regression", "random_forest"]
    winner = max(candidate_names, key=lambda name: (
        validations[name]["average_precision"] if validations[name]["average_precision"] is not None else -1,
        -candidate_names.index(name),
    ))
    metrics["validation"] = validations
    metrics["selected_model"] = winner
    # Report test metrics once; do not use these values to switch models.
    metrics["test"] = {
        name: evaluate_ranked(test[TARGET], model.predict_proba(test[FEATURES])[:, 1])
        for name, model in fitted.items()
    }
    metrics["limitations"] = [
        "Historical associations, not causal explanations.",
        "Training only on delivered reviewed orders introduces selection bias.",
        "Scores are model outputs and are not validated as calibrated probabilities.",
        "Order-item totals are assumed available at purchase; validate this for any production system.",
        "Olist is a historical dataset; no current customer or shipment status is available.",
    ]
    return fitted[winner], metrics


def run(raw_dir: Path, artifact_dir: Path, report_path: Path) -> dict:
    orders, items, customers = load_sources(raw_dir)
    reviews_file = raw_dir / "olist_order_reviews_dataset.csv"
    if not reviews_file.is_file():
        raise FileNotFoundError(f"Missing review CSV: {reviews_file}")
    reviews = pd.read_csv(reviews_file)
    cohort = make_review_cohort(build_order_mart(orders, items, customers), reviews)
    model, metrics = train_and_evaluate(cohort)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "target": TARGET,
                 "selected_model": metrics["selected_model"]}, artifact_dir / "customer_review_model.joblib")
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and assess Olist low-review risk model")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--report", type=Path, default=Path("reports/customer_review_metrics.json"))
    args = parser.parse_args()
    result = run(args.raw_dir, args.artifact_dir, args.report)
    print(f"Selected on validation: {result['selected_model']}")
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
