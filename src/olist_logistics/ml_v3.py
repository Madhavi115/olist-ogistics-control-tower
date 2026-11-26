
"""Compare Logistic Regression and Random Forest for delay risk."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .ml import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    build_model,
    prepare_training_data,
)
from .pipeline import load_sources


def build_random_forest():
    """Build a Random Forest with preprocessing."""

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_ranking(y_true, probabilities, top_fraction=0.10):
    """Evaluate scores and a fixed-size investigation budget."""

    y = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    budget = max(
        1, int(np.ceil(len(probabilities) * top_fraction))
    )

    # Select exactly the highest-scoring orders.
    ranking = np.argsort(-probabilities, kind="stable")
    selected_indices = ranking[:budget]

    alerts = np.zeros(len(y), dtype=int)
    alerts[selected_indices] = 1

    prevalence = float(y.mean())
    precision = float(
        precision_score(y, alerts, zero_division=0)
    )

    return {
        "orders": int(len(y)),
        "late_orders": int(y.sum()),
        "late_rate": prevalence,
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "alert_budget": int(budget),
        "detected_late_orders": int(
            ((alerts == 1) & (y == 1)).sum()
        ),
        "precision_at_10_percent": precision,
        "recall_at_10_percent": float(
            recall_score(y, alerts, zero_division=0)
        ),
        "lift_at_10_percent": (
            float(precision / prevalence)
            if prevalence > 0
            else 0.0
        ),
    }


def train_compare_models(data):
    """Select a model on validation data, then test it."""

    n = len(data)

    if n < 100:
        raise ValueError("Not enough orders for training.")

    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    train = data.iloc[:train_end]
    validation = data.iloc[train_end:validation_end]
    test = data.iloc[validation_end:]

    for name, frame in [
        ("training", train),
        ("validation", validation),
        ("test", test),
    ]:
        if frame["is_late"].nunique() != 2:
            raise ValueError(
                f"{name} data needs both outcome classes."
            )

    candidates = {
        "logistic_regression": build_model(),
        "random_forest": build_random_forest(),
    }

    trained_models = {}
    validation_metrics = {}

    print("\nTraining candidate models...")

    for name, model in candidates.items():
        print(f"Training {name}...")

        model.fit(
            train[FEATURES],
            train["is_late"],
        )

        probabilities = model.predict_proba(
            validation[FEATURES]
        )[:, 1]

        validation_metrics[name] = evaluate_ranking(
            validation["is_late"], probabilities
        )

        trained_models[name] = model

    # Model selection uses validation only.
    selected_name = max(
        validation_metrics,
        key=lambda name: validation_metrics[name][
            "average_precision"
        ],
    )

    print(f"\nSelected model: {selected_name}")

    # Compare holdout performance after model selection.
    # Do not use test results to change the selected model.
    test_metrics = {}
    selected_predictions = None

    for name, model in trained_models.items():
        probabilities = model.predict_proba(
            test[FEATURES]
        )[:, 1]

        test_metrics[name] = evaluate_ranking(
            test["is_late"], probabilities
        )

        if name == selected_name:
            ranking = np.argsort(
                -probabilities, kind="stable"
            )

            budget = max(
                1, int(np.ceil(len(test) * 0.10))
            )

            alerts = np.zeros(len(test), dtype=int)
            alerts[ranking[:budget]] = 1

            selected_predictions = pd.DataFrame(
                {
                    "order_id": test["order_id"].to_numpy(),
                    "actual_is_late": (
                        test["is_late"].to_numpy()
                    ),
                    "risk_score": probabilities,
                    "top_10_percent_alert": alerts,
                }
            )

    report = {
        "experiment": "model_v3_comparison",
        "selected_model": selected_name,
        "selection_metric": "validation_average_precision",
        "training_orders": int(len(train)),
        "validation_orders": int(len(validation)),
        "test_orders": int(len(test)),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    return (
        trained_models[selected_name],
        report,
        selected_predictions,
    )


def main():
    print("Loading Olist data...")

    orders, items, customers = load_sources(
        Path("data/raw")
    )

    data = prepare_training_data(
        orders, items, customers
    )

    model, report, predictions = train_compare_models(
        data
    )

    model_dir = Path("artifacts")
    report_dir = Path("reports")

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        model_dir / "delay_model_v3.joblib",
    )

    (report_dir / "ml_v3_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    predictions.to_csv(
        report_dir / "ml_v3_predictions.csv",
        index=False,
    )

    print("\nMODEL V3 RESULTS:")
    print(json.dumps(report, indent=2))

    print("\nModel and reports saved successfully.")


if __name__ == "__main__":
    main()
