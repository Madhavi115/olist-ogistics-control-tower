
"""Version 2: chronological validation and delay-alert tuning."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .ml import FEATURES, build_model, prepare_training_data
from .pipeline import load_sources


def evaluate(y_true, probabilities, threshold):
    """Calculate classification metrics at an alert threshold."""

    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "alerts": int(predictions.sum()),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def choose_threshold(y_validation, validation_probabilities):
    """Choose the threshold with the best validation F1-score."""

    candidates = np.arange(0.05, 0.951, 0.05)
    scores = []

    for threshold in candidates:
        result = evaluate(
            y_validation, validation_probabilities, threshold
        )
        scores.append(result)

    # Prefer better F1, then precision when scores tie.
    best = max(
        scores,
        key=lambda row: (row["f1"], row["precision"]),
    )

    return best["threshold"], scores


def run_experiment(data):
    """Train, tune on validation, then evaluate on holdout test."""

    if len(data) < 100:
        raise ValueError("Not enough orders for this experiment.")

    n = len(data)
    train_end = int(n * 0.6)
    validation_end = int(n * 0.8)

    train = data.iloc[:train_end]
    validation = data.iloc[train_end:validation_end]
    test = data.iloc[validation_end:]

    y_train = train["is_late"]
    y_validation = validation["is_late"]
    y_test = test["is_late"]

    for name, labels in [
        ("training", y_train),
        ("validation", y_validation),
        ("test", y_test),
    ]:
        if labels.nunique() < 2:
            raise ValueError(
                f"{name} data needs both late and on-time orders."
            )

    model = build_model()
    model.fit(train[FEATURES], y_train)

    validation_probabilities = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    best_threshold, threshold_scores = choose_threshold(
        y_validation, validation_probabilities
    )

    # The test set is evaluated AFTER threshold selection.
    test_probabilities = model.predict_proba(
        test[FEATURES]
    )[:, 1]

    baseline_results = evaluate(y_test, test_probabilities, 0.5)
    tuned_results = evaluate(
        y_test, test_probabilities, best_threshold
    )

    # Also measure an operational alert budget:
    # investigate the top 10% highest-risk test orders.
    alert_budget = max(1, int(np.ceil(len(test) * 0.10)))
    top_indices = np.argsort(-test_probabilities)[:alert_budget]

    top_alerts = np.zeros(len(test), dtype=int)
    top_alerts[top_indices] = 1

    budget_results = {
        "alert_budget": int(alert_budget),
        "precision": float(
            precision_score(y_test, top_alerts, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, top_alerts, zero_division=0)
        ),
        "detected_late_orders": int(
            ((top_alerts == 1) & (y_test.to_numpy() == 1)).sum()
        ),
    }

    report = {
        "model": "LogisticRegression",
        "split": "chronological_60_20_20",
        "training_orders": int(len(train)),
        "validation_orders": int(len(validation)),
        "test_orders": int(len(test)),
        "training_late_rate": float(y_train.mean()),
        "validation_late_rate": float(y_validation.mean()),
        "test_late_rate": float(y_test.mean()),
        "validation_selected_threshold": best_threshold,
        "test_roc_auc": float(
            roc_auc_score(y_test, test_probabilities)
        ),
        "test_average_precision": float(
            average_precision_score(y_test, test_probabilities)
        ),
        "test_default_threshold": baseline_results,
        "test_tuned_threshold": tuned_results,
        "test_top_10_percent_alert_budget": budget_results,
        "test_all_on_time_accuracy": float(1 - y_test.mean()),
    }

    predictions = pd.DataFrame(
        {
            "order_id": test["order_id"].to_numpy(),
            "actual_is_late": y_test.to_numpy(),
            "delay_probability": test_probabilities,
            "alert_default": (
                test_probabilities >= 0.5
            ).astype(int),
            "alert_tuned": (
                test_probabilities >= best_threshold
            ).astype(int),
            "alert_top_10_percent": top_alerts,
        }
    )

    return model, report, predictions, threshold_scores


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Olist ML V2"
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

    print("Loading Olist orders...")
    orders, items, customers = load_sources(args.raw_dir)

    print("Preparing historical features...")
    data = prepare_training_data(orders, items, customers)

    print("Training Model V2...")
    model, report, predictions, threshold_scores = (
        run_experiment(data)
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.model_dir / "delay_model_v2.joblib")

    (args.report_dir / "ml_v2_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    predictions.to_csv(
        args.report_dir / "ml_v2_predictions.csv",
        index=False,
    )

    pd.DataFrame(threshold_scores).to_csv(
        args.report_dir / "ml_v2_validation_thresholds.csv",
        index=False,
    )

    print("\nMODEL V2 RESULTS")
    print(json.dumps(report, indent=2))
    print("\nSaved model, metrics, predictions, and threshold analysis.")


if __name__ == "__main__":
    main()
