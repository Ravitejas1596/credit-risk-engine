from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap


def _add_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror the feature engineering used during training (scripts/train.py._add_features)
    so that the ColumnTransformer sees the expected columns.
    """
    df = df.copy()

    late_30_59 = df.get("NumberOfTime30-59DaysPastDueNotWorse")
    late_60_89 = df.get("NumberOfTime60-89DaysPastDueNotWorse")
    late_90 = df.get("NumberOfTimes90DaysLate")
    if late_30_59 is not None and late_60_89 is not None and late_90 is not None:
        df["delinquency_total"] = late_30_59.fillna(0) + late_60_89.fillna(0) + late_90.fillna(0)
        df["severe_delinquency"] = late_60_89.fillna(0) + 2 * late_90.fillna(0)

    if "MonthlyIncome" in df.columns and "DebtRatio" in df.columns:
        df["debt_burden"] = df["DebtRatio"] * df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    if "NumberOfOpenCreditLinesAndLoans" in df.columns and "NumberRealEstateLoansOrLines" in df.columns:
        df["real_estate_share"] = (
            df["NumberRealEstateLoansOrLines"].fillna(0) / (df["NumberOfOpenCreditLinesAndLoans"].fillna(0) + 1.0)
        )

    return df


@dataclass(frozen=True)
class ScoreResult:
    probability: float
    decision: str
    top_factors: List[Dict[str, Any]]
    shap_values: List[float]
    base_value: float
    feature_names: List[str]
    x_values: List[float]


class CreditRiskEngine:
    def __init__(self, artifacts_path: Path):
        payload = joblib.load(artifacts_path)
        self.pre = payload["preprocessor"]
        self.xgb = payload["xgb_model"]
        self.cal = payload["calibrator"]
        self.feature_names = list(payload.get("feature_names") or [])

        # TreeExplainer over the underlying XGB model (pre-calibration)
        self.explainer = shap.TreeExplainer(self.xgb)

    def _decision(self, p: float) -> str:
        return "APPROVE" if p < 0.35 else "REVIEW" if p < 0.6 else "DECLINE"

    def score(self, features: Dict[str, Any], top_k: int = 8) -> ScoreResult:
        df = pd.DataFrame([features])
        df = _add_features_for_inference(df)
        X = self.pre.transform(df)
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)
        x_row = np.array(X_dense).reshape(-1)

        p = float(self.cal.predict_proba(X)[:, 1][0])

        # SHAP values for model log-odds space; still useful as "drivers"
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals_1d = np.array(shap_vals).reshape(-1)

        base = self.explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.array(base).reshape(-1)[0])
        else:
            base = float(base)

        names = self.feature_names or [f"f{i}" for i in range(len(shap_vals_1d))]
        order = np.argsort(np.abs(shap_vals_1d))[::-1][:top_k]

        top = []
        for idx in order:
            top.append(
                {
                    "feature": names[int(idx)],
                    "contribution": float(shap_vals_1d[int(idx)]),
                    "abs_contribution": float(abs(shap_vals_1d[int(idx)])),
                    "value": float(x_row[int(idx)]) if int(idx) < len(x_row) else None,
                }
            )

        return ScoreResult(
            probability=p,
            decision=self._decision(p),
            top_factors=top,
            shap_values=[float(v) for v in shap_vals_1d.tolist()],
            base_value=base,
            feature_names=names,
            x_values=[float(v) for v in x_row.tolist()],
        )

