# Olist Logistics AI Control Tower — Phase 1

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue)](#testing)

A GitHub-ready **Phase 1 data engineering project** for a future ML-powered,
agentic logistics decision-support platform. The delivered code currently builds
and validates a historical **one-row-per-order** analytical table, produces KPIs,
and includes automated tests. It does **not** yet contain the ML model, agents,
API, dashboard, or production deployment (planned for later phases).

## Architecture

```text
Kaggle Olist CSVs -> schema/key validation -> item-level aggregation
             -> order + customer joins -> historical order mart (Parquet)
             -> KPI report (JSON)
```

## Dataset

Download the original **Brazilian E-Commerce Public Dataset by Olist**:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Put these 3 files in `data/raw/`:

```text
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_customers_dataset.csv
```

The nine-file dataset is free to access, but its license is **CC BY-NC-SA 4.0**.
Do not commit the raw data or redistribute it in this repository. Cite Olist.

## Quick start (local / VS Code)

Prerequisites: Python 3.10+ and Git.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e '.[dev]'
olist-pipeline --raw-dir data/raw --out-dir data/processed --report-dir reports
```

Outputs:

- `data/processed/order_mart.parquet`: historical analytics, one row per order.
- `reports/phase1_kpis.json`: actual report computed from your downloaded CSVs.

**The real dataset is not included**, so repository tests use synthetic fixtures.

## Testing

```bash
ruff check .
pytest -q
```

CI runs lint and tests on pushes to `main` and pull requests. It does not need
Kaggle credentials or external data, because tests use tiny synthetic samples.

## Business definitions

- Delivery delay: actual delivery calendar date is after promised calendar date.
- Eligible denominator: delivered orders with valid purchase, actual delivery,
  and estimated delivery timestamps, and nonnegative delivery duration.
- Undelivered/canceled orders: excluded from late-rate denominator; still counted
  among total orders.
- Multiple items per order: aggregated *before* joining to orders to avoid row
  duplication and inaccurate metrics.

See [data contract](docs/data_contract.md) and [team workflow](docs/team_workflow.md).

## Roadmap

- **Phase 1 (included):** ingestion, schema checks, one-row-per-order mart, KPI report, tests, CI.
- **Phase 2:** leakage-safe purchase-time features.
- **Phase 3:** classification baseline, time-based evaluation, calibrated model.
- **Phase 4:** PostgreSQL, FastAPI, authentication and persistence.
- **Phase 5:** LangGraph investigation agents with controlled SQL tools.
- **Phase 6:** human-in-the-loop Streamlit dashboard.
- **Phase 7:** Docker Compose, monitoring and deployment hardening.
- **Phase 8:** evaluation, demos and portfolio.

## Limitations

This is not a live supply chain feed. Carrier identity, alternative-carrier quotes,
tracking scans and actual intervention outcomes are unavailable. No actual risk
reduction or savings should be claimed without a valid evaluation.
