"""Optional Streamlit multipage panel for experimental review model comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "customer_review_improvements.json"

st.set_page_config(page_title="Customer review model comparison", page_icon="📊", layout="wide")
st.title("Customer review model comparison")
st.caption("Purchase-time features · chronological validation · experimental analysis")
st.warning("The test window was previously inspected. Test comparisons here are exploratory, not an independent untouched evaluation.")

if not REPORT.exists():
    st.info("Run `python -m olist_logistics.customer_review_improvements --raw-dir data/raw` to generate the report.")
    st.stop()

try:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    validation = report["validation"]
    test = report["test_exploratory"]
    chosen = report["selected_on_validation"]
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"Could not load comparison report: {exc}")
    st.stop()

st.write(f"**Model selected on validation:** `{chosen}`")
rows = []
for model in validation:
    rows.append({"Model": model, "Validation AP": validation[model]["average_precision"],
                 "Validation ROC-AUC": validation[model]["roc_auc"],
                 "Test AP (exploratory)": test[model]["average_precision"],
                 "Test ROC-AUC (exploratory)": test[model]["roc_auc"],
                 "Test low reviews found @ 10%": test[model]["detected_low_reviews"]})
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)
st.plotly_chart(px.bar(df, x="Model", y="Validation AP", title="Model selection: validation average precision"), use_container_width=True)
st.subheader("Error analysis (selected model; exploratory test data)")
diag = pd.DataFrame(report.get("test_diagnostics", []))
if not diag.empty:
    dimensions = diag["dimension"].unique().tolist()
    dimension = st.selectbox("Group results by", dimensions)
    st.dataframe(diag.loc[diag["dimension"].eq(dimension)], hide_index=True, use_container_width=True)
else:
    st.info("No groups satisfy the minimum sample size for this report.")
with st.expander("Assumptions and limitations"):
    for item in report.get("notes", []):
        st.write("-", item)
