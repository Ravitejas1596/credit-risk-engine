from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import shap
import streamlit as st

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.model import CreditRiskEngine  # type: ignore  # noqa: E402


API_URL = os.environ.get("CREDIT_API_URL", "").strip() or None
_engine: Optional[CreditRiskEngine] = None


def _score(payload: Dict[str, Any]) -> Dict[str, Any]:
    # If an explicit API URL is configured, use the FastAPI service.
    if API_URL:
        r = requests.post(f"{API_URL}/score", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    # Otherwise, run the model in-process (useful for Hugging Face Spaces).
    global _engine
    if _engine is None:
        artifacts_path = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"
        _engine = CreditRiskEngine(artifacts_path)

    res = _engine.score(payload, top_k=8)
    return {
        "probability_default": res.probability,
        "decision": res.decision,
        "top_factors": res.top_factors,
        "shap": {
            "base_value": res.base_value,
            "values": res.shap_values,
            "feature_names": res.feature_names,
            "x_values": res.x_values,
        },
    }


st.set_page_config(page_title="Credit Risk Engine", layout="wide")
st.title("Credit Risk Engine")
st.caption("XGBoost + calibrated probabilities + SHAP explainability")

with st.sidebar:
    st.subheader("Backend mode")
    if API_URL:
        st.write(f"Calling FastAPI at `{API_URL}`")
    else:
        st.write("Using in-process model inside Streamlit")

col1, col2, col3 = st.columns(3)

with col1:
    util = st.number_input("Revolving Utilization", min_value=0.0, value=0.3, step=0.01)
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    d3059 = st.number_input("30-59 days past due (count)", min_value=0, max_value=98, value=0, step=1)
    d6089 = st.number_input("60-89 days past due (count)", min_value=0, max_value=98, value=0, step=1)

with col2:
    d90 = st.number_input("90+ days late (count)", min_value=0, max_value=98, value=0, step=1)
    debt_ratio = st.number_input("Debt ratio", min_value=0.0, value=0.25, step=0.01)
    income = st.number_input("Monthly income", min_value=0.0, value=6000.0, step=100.0)

with col3:
    open_lines = st.number_input("Open credit lines/loans", min_value=0, value=8, step=1)
    re_loans = st.number_input("Real estate loans/lines", min_value=0, value=1, step=1)
    deps = st.number_input("Dependents", min_value=0, value=0, step=1)

payload = {
    "RevolvingUtilizationOfUnsecuredLines": float(util),
    "age": int(age),
    "NumberOfTime30-59DaysPastDueNotWorse": int(d3059),
    "DebtRatio": float(debt_ratio),
    "MonthlyIncome": float(income),
    "NumberOfOpenCreditLinesAndLoans": int(open_lines),
    "NumberOfTimes90DaysLate": int(d90),
    "NumberRealEstateLoansOrLines": int(re_loans),
    "NumberOfTime60-89DaysPastDueNotWorse": int(d6089),
    "NumberOfDependents": int(deps),
}

run = st.button("Score credit risk", type="primary")

if run:
    with st.spinner("Scoring..."):
        out = _score(payload)

    p = out["probability_default"]
    decision = out["decision"]
    st.subheader(f"Decision: {decision}")
    st.metric("Probability of default", f"{p:.3f}")

    st.divider()
    st.subheader("Why this decision (SHAP)")

    shap_blob = out["shap"]
    values = np.array(shap_blob["values"], dtype=float)
    base = float(shap_blob["base_value"])
    names = list(shap_blob["feature_names"])
    x_values = np.array(shap_blob.get("x_values", []), dtype=float)

    exp = shap.Explanation(
        values=values,
        base_values=base,
        data=x_values if x_values.size else None,
        feature_names=names,
    )

    fig = plt.figure(figsize=(10, 4))
    shap.plots.waterfall(exp, max_display=12, show=False)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Top drivers")
    st.dataframe(out["top_factors"], use_container_width=True)

