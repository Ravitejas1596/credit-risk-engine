---
title: credit-risk-engine
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

## Credit Risk Engine (XGBoost + SHAP) — FastAPI + Streamlit

End-to-end credit scoring demo trained on Kaggle's **Give Me Some Credit** dataset.

### What you get

- Train a calibrated credit-risk model (XGBoost + probability calibration)
- Explain each decision with SHAP (waterfall plot + top drivers)
- FastAPI inference service (`/score`)
- Streamlit UI that calls the API
- Docker image compatible with Hugging Face Spaces (no external infra)

### Local setup

Requires Python **3.11+**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download data (Kaggle)

1. Place your Kaggle credentials at `~/.kaggle/kaggle.json` (recommended), or copy `kaggle.json` into the repo root (it is gitignored).
2. Ensure you've accepted the competition rules on Kaggle.

```bash
python -m scripts.download_data
```

### Train

```bash
python -m scripts.train --seed 42
```

This writes:

- `artifacts/model.joblib`
- `artifacts/metrics.json`

### Run locally

Terminal 1:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0
```

### Deploy (Hugging Face Spaces)

- Create a **Docker Space**
- Push this repo content to that Space
- The Streamlit app listens on port **7860**

