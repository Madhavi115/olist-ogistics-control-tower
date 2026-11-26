
"""FastAPI prediction service for Olist shipment delay risk."""

from datetime import datetime
from functools import lru_cache
from math import ceil
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/delay_model_v3.joblib")

app = FastAPI(
    title="Olist Logistics AI Control Tower",
    description="Predict delivery-delay risk using historical order patterns.",
    version="0.1.0",
)


class OrderInput(BaseModel):
    order_id: str = Field(min_length=1)
    purchase_timestamp: datetime
    estimated_delivery_date: datetime
    item_count: int = Field(ge=1)
    total_price: float = Field(ge=0)
    total_freight: float = Field(ge=0)
    customer_state: str = Field(min_length=2, max_length=2)


class PredictionOutput(BaseModel):
    order_id: str
    delay_risk_score: float
    model: str


@lru_cache(maxsize=1)
def load_model():
    """Load the trusted, locally trained model once."""
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    """Check model availability."""
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Trained model file not found.",
        )

    try:
        load_model()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Trained model could not be loaded.",
        ) from exc

    return {"status": "ready", "model": "random_forest"}


@app.post("/predict", response_model=PredictionOutput)
def predict(order: OrderInput):
    """Return a model risk score for one shipment."""

    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Train Model V3 before requesting predictions.",
        )

    try:
        promised_days = (
            order.estimated_delivery_date
            - order.purchase_timestamp
        ).total_seconds() / 86400
    except TypeError as exc:
        raise HTTPException(
            status_code=422,
            detail="Use consistent time zones for both dates.",
        ) from exc

    if promised_days < 0:
        raise HTTPException(
            status_code=422,
            detail="Promised delivery must not precede purchase.",
        )

    features = pd.DataFrame(
        [
            {
                "item_count": order.item_count,
                "total_price": order.total_price,
                "total_freight": order.total_freight,
                "promised_days": promised_days,
                "purchase_month": order.purchase_timestamp.month,
                "purchase_weekday": (
                    order.purchase_timestamp.weekday()
                ),
                "customer_state": order.customer_state.upper(),
            }
        ]
    )

    try:
        model = load_model()
        risk_score = float(
            model.predict_proba(features)[0, 1]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Model prediction unavailable.",
        ) from exc

    return PredictionOutput(
        order_id=order.order_id,
        delay_risk_score=round(risk_score, 4),
        model="random_forest_v3",
    )

# ----------------------------------------------------------
# Batch shipment risk prediction
# ----------------------------------------------------------


class BatchPredictionInput(BaseModel):
    orders: list[OrderInput] = Field(
        min_length=1,
        max_length=1000,
    )
    top_fraction: float = Field(
        default=0.10,
        gt=0,
        le=1,
    )


class RankedShipment(BaseModel):
    rank: int
    order_id: str
    delay_risk_score: float
    priority_investigation: bool


class BatchPredictionOutput(BaseModel):
    model: str
    total_orders: int
    priority_count: int
    top_fraction: float
    ranked_shipments: list[RankedShipment]


def prepare_batch_features(
    orders: list[OrderInput],
) -> pd.DataFrame:
    """Convert incoming orders to the trained feature schema."""

    feature_rows = []

    for order in orders:
        try:
            promised_days = (
                order.estimated_delivery_date
                - order.purchase_timestamp
            ).total_seconds() / 86400
        except TypeError as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Order {order.order_id}: "
                    "use consistent time zones."
                ),
            ) from exc

        if promised_days < 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Order {order.order_id}: "
                    "promised delivery precedes purchase."
                ),
            )

        feature_rows.append(
            {
                "item_count": order.item_count,
                "total_price": order.total_price,
                "total_freight": order.total_freight,
                "promised_days": promised_days,
                "purchase_month": (
                    order.purchase_timestamp.month
                ),
                "purchase_weekday": (
                    order.purchase_timestamp.weekday()
                ),
                "customer_state": (
                    order.customer_state.upper()
                ),
            }
        )

    return pd.DataFrame(feature_rows)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
)
def predict_batch(payload: BatchPredictionInput):
    """Score orders together and rank highest-risk orders first."""

    orders = payload.orders

    order_ids = [order.order_id for order in orders]

    if len(order_ids) != len(set(order_ids)):
        raise HTTPException(
            status_code=422,
            detail="Order IDs must be unique within a batch.",
        )

    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Trained model file not found.",
        )

    features = prepare_batch_features(orders)

    try:
        model = load_model()

        # One model call scores the entire batch.
        probabilities = model.predict_proba(
            features
        )[:, 1]
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Batch model prediction unavailable.",
        ) from exc

    # Sort by score descending. Original order breaks ties.
    sorted_indices = sorted(
        range(len(orders)),
        key=lambda index: (-float(probabilities[index]), index),
    )

    priority_count = ceil(
        len(orders) * payload.top_fraction
    )

    ranked_shipments = []

    for rank, index in enumerate(sorted_indices, start=1):
        ranked_shipments.append(
            RankedShipment(
                rank=rank,
                order_id=orders[index].order_id,
                delay_risk_score=round(
                    float(probabilities[index]), 4
                ),
                priority_investigation=(
                    rank <= priority_count
                ),
            )
        )

    return BatchPredictionOutput(
        model="random_forest_v3",
        total_orders=len(orders),
        priority_count=priority_count,
        top_fraction=payload.top_fraction,
        ranked_shipments=ranked_shipments,
    )


class InvestigationOutput(BaseModel):
    ranking: dict
    verified_findings: list[dict]
    verified_report: str
    ai_status: str
    ai_analysis: dict | None


@app.post("/investigate", response_model=InvestigationOutput)
def investigate(payload: BatchPredictionInput):
    """Score a batch and run a local LLM evidence-grounded investigation."""
    from .investigation_service import WORKFLOW

    orders = [order.model_dump(mode="json") for order in payload.orders]
    if len({order["order_id"] for order in orders}) != len(orders):
        raise HTTPException(status_code=422, detail="Duplicate order IDs.")
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Trained model not found.")

    # Synchronous FastAPI route: the server executes the graph in a threadpool.
    # No HTTP call back to this same server is made.
    try:
        result = WORKFLOW.invoke(
            {"orders": orders, "top_fraction": payload.top_fraction}
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Investigation service unavailable."
        ) from exc

    return InvestigationOutput(
        ranking=result["ranking"],
        verified_findings=result["findings"],
        verified_report=result["verified_report"],
        ai_status=result["ai_status"],
        ai_analysis=result["structured_analysis"],
    )
