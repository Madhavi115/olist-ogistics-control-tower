"""Pure validation, API client, and display helpers used by the Streamlit app."""

from __future__ import annotations

import csv
import io
from typing import Any

import pandas as pd
import requests

REQUIRED_COLUMNS = (
    "order_id",
    "purchase_timestamp",
    "estimated_delivery_date",
    "item_count",
    "total_price",
    "total_freight",
    "customer_state",
)


class DashboardError(ValueError):
    """Friendly error raised for bad input or unexpected API responses."""


def orders_from_csv(csv_text: str) -> list[dict[str, Any]]:
    """Parse a CSV upload into validated order JSON; never silently drop rows."""
    try:
        reader = csv.DictReader(io.StringIO(csv_text, newline=""))
        if not reader.fieldnames:
            raise DashboardError("CSV must contain a header row.")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise DashboardError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    except csv.Error as exc:
        raise DashboardError(f"Unable to read CSV: {exc}") from exc

    if not 1 <= len(rows) <= 1000:
        raise DashboardError("Upload between 1 and 1,000 shipments.")

    order_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            order_id = row["order_id"].strip()
            state = row["customer_state"].strip().upper()
            purchase = row["purchase_timestamp"].strip()
            estimate = row["estimated_delivery_date"].strip()
            item_count = int(row["item_count"])
            total_price = float(row["total_price"])
            total_freight = float(row["total_freight"])
            if not order_id or order_id in order_ids:
                raise ValueError("Missing or duplicate order_id")
            if len(state) != 2 or not state.isalpha():
                raise ValueError("customer_state must be a two-letter code")
            if item_count < 1 or total_price < 0 or total_freight < 0:
                raise ValueError("item count must be positive; amounts nonnegative")
            start = pd.Timestamp(purchase)
            end = pd.Timestamp(estimate)
            if pd.isna(start) or pd.isna(end) or end < start:
                raise ValueError("invalid delivery date interval")
            order_ids.add(order_id)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise DashboardError(f"CSV row {line_number}: {exc}") from exc
        results.append(
            {
                "order_id": order_id,
                "purchase_timestamp": purchase,
                "estimated_delivery_date": estimate,
                "item_count": item_count,
                "total_price": total_price,
                "total_freight": total_freight,
                "customer_state": state,
            }
        )
    return results


def validate_investigation(result: dict[str, Any]) -> pd.DataFrame:
    """Turn a real investigation API response into a sorted leaderboard."""
    try:
        ranking = result["ranking"]
        rows = ranking["ranked_shipments"]
        total = int(ranking["total_orders"])
        priority_count = int(ranking["priority_count"])
        if not isinstance(rows, list) or len(rows) != total or total < 1:
            raise ValueError("Unexpected number of shipments")
        frame = pd.DataFrame(rows)
        expected = {
            "rank", "order_id", "delay_risk_score", "priority_investigation"
        }
        if not expected.issubset(frame.columns):
            raise ValueError("Ranking is missing required columns")
        frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
        frame["delay_risk_score"] = pd.to_numeric(
            frame["delay_risk_score"], errors="raise"
        )
        if (frame["delay_risk_score"] < 0).any() or (
            frame["delay_risk_score"] > 1
        ).any():
            raise ValueError("Risk scores are outside [0, 1]")
        if frame["rank"].tolist() != list(range(1, total + 1)):
            raise ValueError("Ranks are not consecutive")
        if frame["order_id"].duplicated().any():
            raise ValueError("Duplicate order identifiers in results")
        if not frame["delay_risk_score"].is_monotonic_decreasing:
            raise ValueError("The ranking is not ordered from highest to lowest")
        if frame["priority_investigation"].map(type).ne(bool).any():
            raise ValueError("Priority flags must be booleans")
        if int(frame["priority_investigation"].sum()) != priority_count:
            raise ValueError("Priority flag count does not match the summary")
    except (KeyError, TypeError, ValueError) as exc:
        raise DashboardError(f"Unexpected /investigate response: {exc}") from exc
    return frame


def request_health(base_url: str, timeout: int = 5) -> dict[str, Any]:
    """Read the backend health endpoint; raise on network/server problems."""
    response = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
    response.raise_for_status()
    return response.json()


def request_investigation(
    base_url: str,
    orders: list[dict[str, Any]],
    top_fraction: float = 0.1,
    timeout: int = 180,
) -> dict[str, Any]:
    """Call the existing ML + LangGraph + Qwen3 endpoint once."""
    if not orders or len(orders) > 1000:
        raise DashboardError("Choose between 1 and 1,000 shipments")
    if not 0 < top_fraction <= 1:
        raise DashboardError("Top fraction must be in (0, 1]")
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/investigate",
            json={"orders": orders, "top_fraction": top_fraction},
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        validate_investigation(result)
        return result
    except requests.exceptions.Timeout as exc:
        raise DashboardError(
            "Investigation timed out. Check that Ollama and FastAPI are running."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise DashboardError(
            "Cannot reach FastAPI. Start Uvicorn and check the server URL."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        detail = response.text[:350]
        raise DashboardError(f"API returned HTTP {response.status_code}: {detail}") from exc
    except (ValueError, TypeError) as exc:
        if isinstance(exc, DashboardError):
            raise
        raise DashboardError(f"API returned invalid JSON: {exc}") from exc


def format_ai_analysis(result: dict[str, Any]) -> dict[str, Any]:
    """Return only the validated report fields, or a clear fallback."""
    if result.get("ai_status") != "ok" or not isinstance(
        result.get("ai_analysis"), dict
    ):
        return {
            "assessment": "Local AI analysis unavailable; see verified findings.",
            "unknown_information": [],
            "recommended_actions": [],
            "available": False,
        }
    analysis = result["ai_analysis"]
    return {
        "assessment": str(analysis.get("assessment", "")),
        "unknown_information": list(analysis.get("unknown_information") or []),
        "recommended_actions": list(analysis.get("recommended_actions") or []),
        "available": True,
    }
