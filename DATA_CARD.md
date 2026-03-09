## Data Card — Give Me Some Credit

### Source
- Kaggle competition dataset: “Give Me Some Credit”
- Files used: `cs-training.csv`

### Rows / columns
- Row count depends on the downloaded dataset version.
- Columns are numeric credit/behavior signals (e.g., utilization, delinquency counts, income, debt ratio).

### Target
- `SeriousDlqin2yrs` (1 = financial distress / 90+ days past due within 2 years)

### Known data issues / preprocessing
- Missing values (e.g., `MonthlyIncome`, `NumberOfDependents`) are imputed with median.
- Extreme outliers are clipped for stability (`DebtRatio`, utilization, income).

### License / access
- You must accept the Kaggle competition rules to download the data.
- This repository does **not** redistribute the dataset.

