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

# Historical business analysis is separate from the prediction and investigation API.
st.divider()
with st.expander("📊 Business analytics", expanded=False):
    st.caption("Historical Olist descriptive analysis; independent of the five demo shipments.")
    raw_dir_text = st.text_input(
        "Folder containing original Olist CSV files", value=str(ROOT / "data" / "raw"),
        help="Requires orders, items and customers CSV files; reviews CSV is optional.",
    )
    if st.button("Load business insights"):
        try:
            from olist_logistics.business_intelligence import (
                benchmark_performance,
                business_findings,
                business_summary,
                delivery_benchmark,
                read_business_data,
                regional_performance,
                review_severity,
                sales_month_coverage,
                satisfaction_by_delivery,
                seller_performance,
            )

            with st.spinner("Aggregating historical Olist data..."):
                hist_mart, hist_items, hist_reviews = read_business_data(Path(raw_dir_text).expanduser())
                baseline = delivery_benchmark(hist_mart)
                sellers = seller_performance(hist_mart, hist_items)
                regions = regional_performance(hist_mart)
                reviews = satisfaction_by_delivery(hist_mart, hist_reviews)
                st.session_state["business_data"] = {
                    "summary": business_summary(hist_mart),
                    "benchmark": baseline,
                    "monthly": sales_month_coverage(hist_mart),
                    "sellers": benchmark_performance(sellers, baseline),
                    "regions": benchmark_performance(regions, baseline),
                    "reviews": reviews,
                    "severity": review_severity(hist_mart, hist_reviews) if hist_reviews is not None else None,
                    "findings": business_findings(hist_mart, sellers, regions, reviews),
                }
            st.success("Historical business insights loaded.")
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            st.error(f"Could not load business insights: {exc}")

    if "business_data" in st.session_state:
        data = st.session_state["business_data"]
        summary = data["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Historical orders", f"{summary['orders']:,}")
        c2.metric("Item sales value", f"R$ {summary['item_value_brl']:,.0f}")
        c3.metric("Customer states", summary["customer_states"])
        c4.metric("Eligible late rate", f"{data['benchmark']:.2%}" if data["benchmark"] is not None else "N/A")
        st.caption("Item value excludes freight; it is not profit or confirmed recognized revenue.")

        st.subheader("Business findings")
        if data["findings"]:
            for finding in data["findings"]:
                st.markdown(f"- {finding}")
        else:
            st.info("Not enough eligible observations for a business summary.")
        st.caption("Descriptive findings suggest where to investigate; they do not prove causes.")

        sales_tab, seller_tab, region_tab, review_tab = st.tabs(
            ["📈 Sales trends", "🏪 Seller scorecards", "🗺️ Regional logistics", "⭐ Customer reviews"]
        )
        with sales_tab:
            df = data["monthly"]
            if df.empty:
                st.info("No purchase-month sales observations are available.")
            else:
                interior = df[df["month_status"] == "interior month"]
                chart_data = interior if not interior.empty else df
                st.plotly_chart(
                    px.line(chart_data, x="purchase_month", y="item_value_brl", markers=True,
                            title="Monthly item sales value (boundary months excluded when possible)"),
                    use_container_width=True,
                )
                boundary = df[df["month_status"] != "interior month"]
                if not boundary.empty:
                    st.warning("First/last observed months are excluded from the trend chart when possible. "
                               "They may have partial coverage; this is not proof that they are incomplete.")
                st.dataframe(df, hide_index=True, use_container_width=True)
        with seller_tab:
            df = data["sellers"]
            minimum = st.slider("Minimum eligible seller deliveries", 20, 500, 100, 10)
            eligible = df[df["eligible_deliveries"] >= minimum].sort_values(
                "excess_late_deliveries", ascending=False
            )
            st.caption("Ranked by observed late deliveries above a simple overall-rate benchmark. "
                       "Multi-seller orders count for each participating seller; this does not assign blame.")
            st.dataframe(eligible.head(100), hide_index=True, use_container_width=True)
            st.download_button("Download seller scorecards", df.to_csv(index=False),
                               "seller_scorecards.csv", "text/csv")
        with region_tab:
            df = data["regions"]
            minimum = st.slider("Minimum eligible state deliveries", 20, 1000, 100, 20)
            eligible = df[df["eligible_deliveries"] >= minimum].sort_values(
                "excess_late_deliveries", ascending=False
            )
            st.caption("Compare number of late deliveries and percentage: a high rate on few orders "
                       "is not always the largest customer impact.")
            if not eligible.empty:
                plot = eligible.head(15)
                fig = px.bar(plot, x="customer_state", y="late_deliveries",
                             hover_data=["eligible_deliveries", "late_rate_pct", "excess_late_deliveries"],
                             title="Late deliveries in states ranked by excess over baseline")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(eligible, hide_index=True, use_container_width=True)
        with review_tab:
            reviews = data["reviews"]
            severity = data["severity"]
            if severity is None:
                st.info("Add olist_order_reviews_dataset.csv for customer-review analysis.")
            elif severity.empty:
                st.info("No valid reviews matched eligible delivered orders.")
            else:
                st.plotly_chart(
                    px.bar(severity, x="delay_group", y="low_review_rate_pct",
                           hover_data=["reviewed_orders", "mean_review_score"],
                           title="Low customer ratings (1–2 stars) by lateness"),
                    use_container_width=True,
                )
                st.dataframe(severity, hide_index=True, use_container_width=True)
                st.markdown("**Customer impact: where low reviews occur**")
                total_low = int(severity["low_review_count"].sum())
                total_reviewed = int(severity["reviewed_orders"].sum())
                v1, v2 = st.columns(2)
                v1.metric("Reviewed eligible deliveries", f"{total_reviewed:,}")
                v2.metric("Orders with low reviews", f"{total_low:,}")
                st.plotly_chart(
                    px.bar(
                        severity, x="delay_group", y="low_review_count",
                        text="low_review_count",
                        hover_data=["share_of_low_reviews_pct", "reviewed_orders", "low_review_rate_pct"],
                        title="Number of low-reviewed orders by delivery lateness",
                        labels={"low_review_count": "Low-reviewed orders", "delay_group": "Delivery lateness"},
                    ),
                    use_container_width=True,
                )
                st.caption("Counts are calculated from individual reviewed orders, not estimated "
                           "from rounded percentages. On-time deliveries may account for many "
                           "negative reviews because their order volume is large.")
                st.markdown("**Late versus on-time summary**")
                st.dataframe(reviews, hide_index=True, use_container_width=True)
                st.caption("Association only. If an order has multiple valid reviews, its mean "
                           "review score is used. Reviews and actual delivery dates must not "
                           "be used as purchase-time prediction inputs.")

# Customer review prediction is a separate historical modeling use case.
# Do not score the five shipment demo orders with this model: they are a
# separate demonstration batch without observed customer-review outcomes.
st.divider()
with st.expander("⭐ Customer review prediction — model evaluation", expanded=False):
    st.caption("Purchase-time low-review prediction. Historical validation only; not live customer support.")
    review_metrics_file = ROOT / "reports" / "customer_review_metrics.json"
    if not review_metrics_file.is_file():
        st.info(
            "Train the review model first with `python -m olist_logistics.customer_review_model "
            "--raw-dir data/raw`. This will create the evaluation report."
        )
    else:
        try:
            review_metrics = json.loads(review_metrics_file.read_text(encoding="utf-8"))
            best_name = review_metrics["selected_model"]
            metrics = review_metrics["test"][best_name]
            st.write(f"**Validation-selected model:** {best_name.replace('_', ' ').title()}")
            st.caption("Chronological test period: " +
                       str(review_metrics["date_ranges"]["test"]["start"]) + " to " +
                       str(review_metrics["date_ranges"]["test"]["end"]))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Test orders", f"{metrics['rows']:,}")
            m2.metric("Low-review prevalence", f"{metrics['prevalence']:.1%}")
            m3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "N/A")
            m4.metric("Average precision", f"{metrics['average_precision']:.3f}" if metrics["average_precision"] is not None else "N/A")
            st.write(
                f"**Top-10% review budget:** flagged {metrics['alert_count']:,} orders and "
                f"identified {metrics['detected_low_reviews']:,} low-reviewed orders "
                f"(precision {metrics['precision_at_10_pct']:.1%})."
            )
            comparison = pd.DataFrame([
                {"Model": name.replace("_", " ").title(),
                 "Average precision": row["average_precision"],
                 "Low reviews in top 10%": row["detected_low_reviews"]}
                for name, row in review_metrics["test"].items()
            ])
            st.plotly_chart(
                px.bar(comparison, x="Model", y="Low reviews in top 10%",
                       title="Historical holdout: low-reviewed orders found within fixed alert budget"),
                use_container_width=True,
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.warning("This model is trained only on delivered, reviewed orders. It cannot establish "
                       "why a review is negative, and it has not been validated on all incoming orders.")
            st.caption("Scores are not verified calibrated probabilities. The test set was not "
                       "used for model selection.")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            st.error(f"Could not display customer review metrics: {exc}")
