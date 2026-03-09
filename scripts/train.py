from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier


TARGET = "SeriousDlqin2yrs"


def _clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 5)
    if "DebtRatio" in df.columns:
        df["DebtRatio"] = df["DebtRatio"].clip(0, 10)
    if "MonthlyIncome" in df.columns:
        df["MonthlyIncome"] = df["MonthlyIncome"].clip(0, df["MonthlyIncome"].quantile(0.99))
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Payment history signals
    late_30_59 = df.get("NumberOfTime30-59DaysPastDueNotWorse")
    late_60_89 = df.get("NumberOfTime60-89DaysPastDueNotWorse")
    late_90 = df.get("NumberOfTimes90DaysLate")
    if late_30_59 is not None and late_60_89 is not None and late_90 is not None:
        df["delinquency_total"] = late_30_59.fillna(0) + late_60_89.fillna(0) + late_90.fillna(0)
        df["severe_delinquency"] = late_60_89.fillna(0) + 2 * late_90.fillna(0)

    # Debt / capacity signals
    if "MonthlyIncome" in df.columns and "DebtRatio" in df.columns:
        df["debt_burden"] = df["DebtRatio"] * df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

    # Credit mix / experience
    if "NumberOfOpenCreditLinesAndLoans" in df.columns and "NumberRealEstateLoansOrLines" in df.columns:
        df["real_estate_share"] = (
            df["NumberRealEstateLoansOrLines"].fillna(0) / (df["NumberOfOpenCreditLinesAndLoans"].fillna(0) + 1.0)
        )

    return df


def load_training_frame(raw_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    for col in ["Unnamed: 0", "Id", "ID"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column {TARGET}")
    df = _clip_outliers(df)
    df = _add_features(df)
    return df


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    num_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, feature_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@dataclass
class TrainedArtifacts:
    preprocessor: ColumnTransformer
    model: XGBClassifier
    calibrator: CalibratedClassifierCV
    feature_names: list[str]


def train(df: pd.DataFrame, seed: int) -> Tuple[TrainedArtifacts, Dict[str, Any]]:
    y = df[TARGET].astype(int).to_numpy()
    X = df.drop(columns=[TARGET])

    feature_cols = list(X.columns)
    pre = build_preprocessor(feature_cols)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    xgb = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.0,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=seed,
    )

    # Fit preprocessing + model
    Xtr = pre.fit_transform(X_train)
    xgb.fit(Xtr, y_train)

    # Calibrate on validation split (prefit)
    Xv = pre.transform(X_val)
    calibrator = CalibratedClassifierCV(xgb, method="isotonic", cv="prefit")
    calibrator.fit(Xv, y_val)

    # Evaluate
    Xt = pre.transform(X_test)
    proba = calibrator.predict_proba(Xt)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "n_test": int(len(y_test)),
        "seed": int(seed),
    }

    # Feature names (after preprocessor)
    try:
        fn = list(pre.get_feature_names_out())
    except Exception:
        fn = feature_cols

    artifacts = TrainedArtifacts(
        preprocessor=pre,
        model=xgb,
        calibrator=calibrator,
        feature_names=fn,
    )
    return artifacts, metrics


def save(artifacts: TrainedArtifacts, metrics: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "preprocessor": artifacts.preprocessor,
        "xgb_model": artifacts.model,
        "calibrator": artifacts.calibrator,
        "feature_names": artifacts.feature_names,
    }
    joblib.dump(payload, out_dir / "model.joblib")

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_csv",
        type=str,
        default=str(Path("data/raw/cs-training.csv")),
    )
    parser.add_argument("--out_dir", type=str, default=str(Path("artifacts")))
    args = parser.parse_args()

    df = load_training_frame(Path(args.train_csv))
    artifacts, metrics = train(df, seed=args.seed)
    save(artifacts, metrics, Path(args.out_dir))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
