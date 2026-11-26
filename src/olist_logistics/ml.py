
"""Train and evaluate an Olist delivery-delay prediction model."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .pipeline import load_sources

NUMERIC_FEATURES = [
    "item_count",
    "total_price",
    "total_freight",
    "promised_days",
    "purchase_month",
    "purchase_weekday",
]

CATEGORICAL_FEATURES = ["customer_state"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_training_data(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Create historical features and actual late-delivery labels."""

    orders = orders.copy()

    date_columns = [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ]

    for column in date_columns:
        orders[column] = pd.to_datetime(
            orders[column], errors="coerce"
        )

    # Each order can have multiple item rows.
    item_features = (
        items.groupby("order_id")
        .agg(
            item_count=("order_id", "size"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
        )
        .reset_index()
    )

    data = orders.merge(
        item_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    data = data.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    # Only completed deliveries have a known outcome.
    data = data[
        (data["order_status"] == "delivered")
        & data["order_purchase_timestamp"].notna()
        & data["order_estimated_delivery_date"].notna()
        & data["order_delivered_customer_date"].notna()
    ].copy()

    purchase = data["order_purchase_timestamp"]
    estimate = data["order_estimated_delivery_date"]
    #actual = data["order_delivered_customer_date"]

    data["promised_days"] = (
        estimate - purchase
    ).dt.total_seconds() / 86400

    # Exclude invalid promises earlier than the purchase.
    data = data[data["promised_days"] >= 0].copy()

    # Late means delivered on a calendar date AFTER the promise.
    data["is_late"] = (
        data["order_delivered_customer_date"].dt.normalize()
        > data["order_estimated_delivery_date"].dt.normalize()
    ).astype(int)

    data["purchase_month"] = purchase.dt.month
    data["purchase_weekday"] = purchase.dt.dayofweek

    for column in ["item_count", "total_price", "total_freight"]:
        data[column] = data[column].fillna(0)

    data["customer_state"] = (
        data["customer_state"].fillna("UNKNOWN")
    )

    return data.sort_values(
        "order_purchase_timestamp"
    ).reset_index(drop=True)


def build_model() -> Pipeline:
    """Build preprocessing and classification steps."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_evaluate(data: pd.DataFrame):
    """Train on earlier orders and evaluate on later orders."""

    if len(data) < 20:
        raise ValueError("Not enough delivered orders for training.")

    # Chronological split: earliest 80% vs latest 20%.
    split_index = int(len(data) * 0.8)

    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    x_train = train[FEATURES]
    y_train = train["is_late"]

    x_test = test[FEATURES]
    y_test = test["is_late"]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Both training and test sets need late and on-time orders."
        )

    model = build_model()
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "LogisticRegression",
        "total_eligible_orders": int(len(data)),
        "training_orders": int(len(train)),
        "test_orders": int(len(test)),
        "train_late_rate": float(y_train.mean()),
        "test_late_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "decision_threshold": 0.5,
    }

    results = pd.DataFrame(
        {
            "order_id": test["order_id"].to_numpy(),
            "actual_is_late": y_test.to_numpy(),
            "predicted_is_late": predictions,
            "delay_probability": probabilities,
        }
    )

    return model, metrics, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Olist delivery-delay baseline"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts")
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports")
    )
    args = parser.parse_args()

    print("Loading Olist datasets...")
    orders, items, customers = load_sources(args.raw_dir)

    print("Preparing machine-learning features...")
    data = prepare_training_data(orders, items, customers)

    print(f"Eligible delivered orders: {len(data):,}")

    print("Training and evaluating the model...")
    model, metrics, predictions = train_and_evaluate(data)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_dir / "delay_model.joblib"
    metrics_path = args.report_dir / "ml_metrics.json"
    predictions_path = (
        args.report_dir / "ml_holdout_predictions.csv"
    )

    joblib.dump(model, model_path)

    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    predictions.to_csv(predictions_path, index=False)

    print("\nML evaluation results:")
    print(json.dumps(metrics, indent=2))

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
