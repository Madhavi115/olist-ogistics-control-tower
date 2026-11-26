"""Interactive Olist Logistics AI Control Tower dashboard.

Start from the project root with:
    python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from dashboard.dashboard_core import (
    DashboardError,
    format_ai_analysis,
    orders_from_csv,
    request_health,
    request_investigation,
    validate_investigation,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample_orders.csv"
METRICS_FILE = ROOT / "reports" / "ml_v3_metrics.json"

st.set_page_config(
    page_title="Olist | Logistics AI Control Tower",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {padding-top: 1.5rem; max-width: 1450px;}
    [data-testid="stMetric"] {border: 1px solid #dce5e9; padding: 1rem;
                              border-radius: 12px; background: #f8fafc;}
    .subtle-note {color: #64748b; font-size: 0.88rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📦 Olist Logistics AI Control Tower")
st.caption(
    "Risk-based shipment prioritization · Random Forest V3 · "
    "FastAPI · LangGraph · local Qwen3"
)
st.info(
    "Research prototype using historical Olist patterns. Scores are **not** "
    "calibrated probabilities. Priority means highest ranked within the current "
    "batch, not a confirmed delivery delay.",
    icon="ℹ️",
)

with st.sidebar:
    st.header("Control center")
    api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000")
    source = st.radio("Shipment source", ["Five demo shipments", "Upload CSV"])
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload order features", type=["csv"])
        st.caption("At most 1,000 orders. Same columns as the sample CSV.")
    fraction = st.slider("Investigation budget (%)", 1, 100, 10, 1)
    st.download_button(
        "Download example CSV",
        data=SAMPLE_CSV.read_bytes(),
        file_name="sample_olist_orders.csv",
        mime="text/csv",
    )
    st.divider()
    if st.button("Check backend connection"):
        try:
            h = request_health(api_url)
            st.success(f"Backend ready · {h.get('model', 'unknown model')}")
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Backend unavailable: {exc}")

try:
    if source == "Five demo shipments":
        orders = orders_from_csv(SAMPLE_CSV.read_text(encoding="utf-8"))
    elif uploaded is not None:
        orders = orders_from_csv(uploaded.getvalue().decode("utf-8-sig"))
    else:
        orders = []
except (DashboardError, UnicodeDecodeError) as exc:
    st.error(f"Input error: {exc}")
    orders = []

with st.expander("Preview incoming shipments", expanded=False):
    if orders:
        st.dataframe(pd.DataFrame(orders), use_container_width=True, hide_index=True)
    else:
        st.caption("Upload a CSV to preview the orders.")

left, _ = st.columns([1, 4])
with left:
    run_clicked = st.button(
        "🔎 Run AI investigation",
        type="primary",
        use_container_width=True,
        disabled=not orders,
    )

if run_clicked:
    try:
        with st.spinner("Scoring shipments and running the local AI investigation..."):
            # No fake/mock scores: all figures below come from the real API response.
            st.session_state["investigation"] = request_investigation(
                api_url, orders, top_fraction=fraction / 100
            )
        st.success("Investigation finished.")
    except DashboardError as exc:
        st.session_state.pop("investigation", None)
        st.error(str(exc))

result = st.session_state.get("investigation")
if result:
    try:
        leaderboard = validate_investigation(result)
    except DashboardError as exc:
        st.error(str(exc))
        st.stop()

    ranking = result["ranking"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders evaluated", ranking["total_orders"])
    c2.metric("Priority investigations", ranking["priority_count"])
    c3.metric("Top-ranked order", leaderboard.iloc[0]["order_id"])
    c4.metric("Top risk score", f"{leaderboard.iloc[0]['delay_risk_score']:.4f}")

    tab_overview, tab_ai, tab_evidence, tab_metrics = st.tabs(
        ["📊 Risk ranking", "🧠 AI analyst", "📋 Verified evidence", "📈 ML performance"]
    )

    with tab_overview:
        st.subheader("Shipment risk leaderboard")
        display = leaderboard.rename(
            columns={
                "rank": "Rank",
                "order_id": "Order ID",
                "delay_risk_score": "Risk score",
                "priority_investigation": "Priority investigation",
            }
        )
        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Risk score": st.column_config.ProgressColumn(
                    "Risk score", min_value=0, max_value=1, format="%.4f"
                ),
                "Priority investigation": st.column_config.CheckboxColumn(
                    "Priority investigation"
                ),
            },
        )
        fig = px.bar(
            leaderboard,
            x="delay_risk_score",
            y="order_id",
            orientation="h",
            color="priority_investigation",
            color_discrete_map={True: "#d97706", False: "#2563eb"},
            labels={
                "delay_risk_score": "Relative model risk score",
                "order_id": "Order ID",
                "priority_investigation": "Priority",
            },
            title="Highest-ranked shipments first",
        )
        fig.update_layout(
            template="plotly_white",
            yaxis={"categoryorder": "array", "categoryarray": leaderboard["order_id"][::-1].tolist()},
            height=max(360, min(850, 55 * len(leaderboard))),
            legend_title_text="Priority",
        )
        fig.update_xaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download ranked shipments (CSV)",
            data=leaderboard.to_csv(index=False),
            file_name="ranked_shipments.csv",
            mime="text/csv",
        )

    with tab_ai:
        st.subheader("Local Qwen3 investigation")
        analysis = format_ai_analysis(result)
        if not analysis["available"]:
            st.warning("Ollama response unavailable or invalid. Verified evidence remains accessible.")
        else:
            st.markdown("**AI assessment (unverified narrative)**")
            st.write(analysis["assessment"])
            lhs, rhs = st.columns(2)
            with lhs:
                st.markdown("**Unknown information**")
                for item in analysis["unknown_information"]:
                    st.markdown(f"- {item}")
            with rhs:
                st.markdown("**Recommended operational checks**")
                for item in analysis["recommended_actions"]:
                    st.markdown(f"- {item}")
            st.caption("Review all AI language: no live carrier data is connected.")
        st.download_button(
            "Download investigation JSON",
            data=json.dumps(result, indent=2),
            file_name="olist_investigation.json",
            mime="application/json",
        )

    with tab_evidence:
        st.subheader("Verified deterministic findings")
        findings = result.get("verified_findings", [])
        if findings:
            selected_order = st.selectbox(
                "Priority shipment", [f["order_id"] for f in findings]
            )
            finding = next(f for f in findings if f["order_id"] == selected_order)
            st.json(finding)
        else:
            st.info("No verified priority findings in the response.")
        with st.expander("Full verified investigation report"):
            st.code(result.get("verified_report", ""), language="text")
        st.caption("The facts above come from the deterministic investigation node.")

    with tab_metrics:
        st.subheader("Historical model validation — V3 holdout")
        if METRICS_FILE.exists():
            try:
                metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
                test = metrics["test_metrics"]
                rf = test["random_forest"]
                logit = test["logistic_regression"]
                d1, d2, d3 = st.columns(3)
                d1.metric("Random Forest ROC-AUC", f"{rf['roc_auc']:.3f}")
                d2.metric("Random Forest AP", f"{rf['average_precision']:.3f}")
                d3.metric("Precision at top 10%", f"{rf['precision_at_10_percent']:.2%}")
                compare = pd.DataFrame(
                    [
                        {"Model": "Random Forest", "Detected delayed orders": rf["detected_late_orders"]},
                        {"Model": "Logistic Regression", "Detected delayed orders": logit["detected_late_orders"]},
                    ]
                )
                chart = px.bar(
                    compare, x="Model", y="Detected delayed orders",
                    text="Detected delayed orders", title="Top 10% alert budget on holdout orders",
                    color="Model", color_discrete_sequence=["#d97706", "#2563eb"],
                )
                chart.update_layout(template="plotly_white", showlegend=False)
                st.plotly_chart(chart, use_container_width=True)
                st.caption(
                    "Metrics come from historical held-out Olist orders, not from the uploaded batch. "
                    "The V3 score is not calibrated."
                )
            except (KeyError, ValueError, TypeError) as exc:
                st.warning(f"Could not read ML metrics: {exc}")
        else:
            st.info(
                "Run `python -m olist_logistics.ml_v3` to create "
                "`reports/ml_v3_metrics.json` and enable the performance tab."
            )
else:
    st.markdown("### Ready to investigate")
    st.write(
        "Select the five demo shipments or upload a CSV, then click **Run AI investigation**. "
        "The dashboard will call your existing `POST /investigate` endpoint."
    )
    if METRICS_FILE.exists():
        st.caption("Historical model metrics are available in reports/ml_v3_metrics.json.")
