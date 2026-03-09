from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scripts.train import TARGET, load_training_frame


def _group_metrics(y: np.ndarray, p: np.ndarray, group: pd.Series) -> pd.DataFrame:
    rows = []
    group = group.reset_index(drop=True)
    for g in group.dropna().unique().tolist():
        mask = (group == g).to_numpy()
        n = int(mask.sum())
        if n < 200:
            continue
        yy = y[mask]
        pp = p[mask]
        auc = float("nan")
        if len(np.unique(yy)) == 2:
            auc = float(roc_auc_score(yy, pp))
        rows.append(
            {
                "group": str(g),
                "n": n,
                "default_rate": float(np.mean(yy)),
                "mean_score": float(np.mean(pp)),
                "roc_auc": auc,
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_csv = repo_root / "data" / "raw" / "cs-training.csv"
    artifacts_path = repo_root / "artifacts" / "model.joblib"
    out_dir = repo_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_frame(raw_csv)

    y = df[TARGET].astype(int).to_numpy()
    X = df.drop(columns=[TARGET])

    # Mirror the train.py split strategy
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    X_test = X_test.reset_index(drop=True)

    payload = joblib.load(artifacts_path)
    pre = payload["preprocessor"]
    cal = payload["calibrator"]

    Xt = pre.transform(X_test)
    p = cal.predict_proba(Xt)[:, 1]

    # Proxy groupings (dataset does not contain protected attributes)
    age = X_test["age"] if "age" in X_test.columns else pd.Series([np.nan] * len(X_test))
    age_bin = pd.cut(age, bins=[18, 25, 35, 45, 55, 65, 100], include_lowest=True)

    deps = X_test.get("NumberOfDependents")
    deps_bin = None
    if deps is not None:
        deps_bin = pd.cut(deps.fillna(-1), bins=[-2, -0.5, 0.5, 2.5, 10_000], labels=["missing", "0", "1-2", "3+"])

    age_table = _group_metrics(y_test, p, age_bin.astype(str))
    deps_table = _group_metrics(y_test, p, deps_bin.astype(str)) if deps_bin is not None else pd.DataFrame()

    md = []
    md.append("## Bias audit (proxy groups)\n")
    md.append(
        "This dataset does not include protected attributes. The following slices are **proxies** intended to spot obvious performance skews.\n"
    )
    md.append("### Slice: age bins\n")
    if not age_table.empty:
        md.append(age_table.to_markdown(index=False))
        md.append("\n")
    else:
        md.append("_Not enough data per bin._\n")

    md.append("### Slice: dependents (missing / 0 / 1-2 / 3+)\n")
    if not deps_table.empty:
        md.append(deps_table.to_markdown(index=False))
        md.append("\n")
    else:
        md.append("_Not available._\n")

    out_path = out_dir / "bias_audit.md"
    out_path.write_text("\n".join(md))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

