# Olist Logistics AI Control Tower — Streamlit Dashboard

## What is included

- `dashboard/app.py` — the Streamlit dashboard UI
- `dashboard/dashboard_core.py` — CSV validation, API calls, and ranking validation
- `dashboard/__init__.py` — Python package marker
- `data/sample_orders.csv` — the same five historic sample orders used in the project
- `tests/test_dashboard_core.py` — seven isolated tests (no running backend needed)
- `requirements-dashboard.txt` — dashboard dependencies

This is an **add-on** to your existing repository. It does **not** replace your FastAPI app, trained model, or investigation service. It expects your local FastAPI endpoint `POST /investigate` to return the response format you already demonstrated.

## Installation

1. Extract this ZIP and **copy the included `dashboard/`, `data/sample_orders.csv`, `tests/test_dashboard_core.py`, and `requirements-dashboard.txt` into the root of your existing `olist-logistics-control-tower` repository**. Merge folders rather than replacing existing folders. Do not overwrite your existing data, tests, or API.
2. Activate the existing virtual environment:

   ```bash
   source .venv/bin/activate
   python -m pip install -r requirements-dashboard.txt
   ```

3. Ensure Ollama is running (`ollama list` should show your `qwen3:4b` model). Leave the Ollama application running.
4. **Terminal 1: start/keep FastAPI running** (if port 8000 is already in use, you probably already have it running; verify with `curl http://127.0.0.1:8000/health`):

   ```bash
   python -m uvicorn olist_logistics.api:app --reload
   ```

5. **Terminal 2: start the dashboard from the project root**:

   ```bash
   source .venv/bin/activate
   python -m streamlit run dashboard/app.py
   ```

6. Open `http://localhost:8501` in Chrome. Use the five demo shipments and click **Run AI investigation**. Expect 5 orders, 1 priority, and `ORDER-004` ranked first for the historical sample if your V3 model is unchanged. Actual scores come from the backend, not from the dashboard.

## Tests

```bash
python -m pytest tests/test_dashboard_core.py -q
python -m pytest -q
python -m ruff check dashboard tests/test_dashboard_core.py
```

The isolated tests mock HTTP requests, use sample CSV text, and validate fallback behavior. They do not require Ollama, FastAPI, or a trained Random Forest to execute. The full-suite command may include existing project tests.

## CSV format

Required columns: `order_id`, `purchase_timestamp`, `estimated_delivery_date`, `item_count`, `total_price`, `total_freight`, `customer_state`. Upload up to 1,000 orders. See `data/sample_orders.csv`. Enter order features as they would be known at the moment of prediction; do not include actual delivery outcomes. This dashboard is an experimental prototype, **not a live shipment tracking product**.

## Operational notes

- The investigation endpoint can take longer while your Mac runs Qwen3. The UI allows 180 seconds per request and shows failures without generating fake model results.
- The ML performance tab reads your local `reports/ml_v3_metrics.json` produced by `python -m olist_logistics.ml_v3`. If missing, the tab explains how to create it.
- Risk scores are uncalibrated. Priority flags denote rank within the submitted batch rather than objectively high probability of delay.
- The LLM report is a generated narrative and may contain errors; deterministic evidence is shown separately.
- Avoid uploading customer data or binding the API to public network interfaces without proper authentication and security controls.
- If port 8501 is busy, run `python -m streamlit run dashboard/app.py --server.port 8502`.
