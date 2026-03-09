## Model Card — Credit Risk Engine

### Model details
- **Task**: binary classification (financial distress in next 2 years)
- **Model**: XGBoost (`XGBClassifier`) + isotonic probability calibration
- **Explainability**: SHAP TreeExplainer (feature contributions in model space)

### Intended use
- Educational/demo credit-risk scoring with transparent drivers.
- Not for real lending decisions.

### Training data
- Source: Kaggle “Give Me Some Credit” competition dataset (user must accept Kaggle rules).
- Target: `SeriousDlqin2yrs`

### Metrics (holdout split)
See `artifacts/metrics.json`.

### Limitations
- Dataset is historical/competition-focused; may not represent any current portfolio.
- Sensitive attributes are not provided; fairness can only be approximated via available proxies.
- SHAP values are for the underlying tree model (pre-calibration); calibrated probability is used for the final score.

