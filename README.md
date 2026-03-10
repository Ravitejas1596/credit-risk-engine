# 🧾 Credit Risk Engine

> *A loan decision isn't just a number — it's an explanation someone deserves.*

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet?style=flat-square)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Open%20on%20HuggingFace-yellow?style=for-the-badge)](https://huggingface.co/spaces/ravitejas1596/credit-risk-engine)

---

## 💡 Why I Built This

Every year, millions of loan applications get rejected.

Most applicants never find out exactly why. The bank says "we reviewed
your application and unfortunately..." — and that's it. A black box
made a decision that affects someone's life, and no one explained it.

That bothered me.

Credit scoring is one of the oldest and most consequential applications
of machine learning. But most implementations stop at the model —
a probability score, a threshold, approved or denied. What's missing
is the *why*: which specific factors in this person's financial profile
drove the decision, and by how much.

I built this project to explore what a **transparent, explainable
credit scoring system** actually looks like in practice — one where
every decision comes with a clear breakdown of the factors behind it.
Not a black box. An engine with a window.

Trained on Kaggle's *Give Me Some Credit* dataset — a real-world
credit bureau dataset with 150k borrowers and actual default outcomes.

---

## ✨ What It Does

### 📋 Credit Score + Risk Decision
Submit a borrower profile and get an instant default probability with
a calibrated risk tier — not just a raw score, but a properly
calibrated probability you can actually trust.

- Real-time scoring via FastAPI `/score` endpoint
- Probability calibration (Platt scaling) — raw XGBoost scores adjusted
  to reflect true default rates
- Risk tiers: Low / Medium / High / Very High
- Each tier maps to an actionable lending decision

### 🔍 SHAP Explainability — Why This Decision?
Every score comes with a full SHAP breakdown showing exactly which
features pushed the probability up or down, and by how much.

- Waterfall plot — visual decomposition of the score
- Top positive drivers: "what's working against this borrower"
- Top negative drivers: "what's working in their favour"
- Plain-English feature labels — not raw column names

---

## 🧠 How It Works
```
Raw Borrower Data (Give Me Some Credit)
              │
              ▼
┌─────────────────────────────┐
│      Feature Engineering    │
│  - Debt-to-income ratio     │
│  - Utilization rate bands   │
│  - Delinquency indicators   │
│  - Age + credit line counts │
│  - Missing value imputation │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│     XGBoost Classifier      │
│  - scale_pos_weight for     │
│    class imbalance          │
│  - Early stopping on AUC    │
│  - Platt scaling calibration│
└──────────┬──────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────────┐
│  Score  │  │  SHAP Explainer  │
│  0.0–1.0│  │  Waterfall plot  │
│  + tier │  │  Top drivers     │
└─────────┘  └──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   FastAPI /score endpoint   │
│   + Streamlit UI            │
│   + Docker container        │
│   + HuggingFace Spaces      │
└─────────────────────────────┘
```

### Why Probability Calibration Matters

A raw XGBoost model might output 0.73 for a borrower — but that
doesn't mean a 73% chance of default. Uncalibrated models are often
overconfident. Platt scaling adjusts the output so that a score of
0.73 actually reflects approximately 73% observed default rate in
the training data. This matters enormously in credit decisions where
the score is used to set interest rates or loan limits.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Model** | XGBoost | Gradient boosted classifier for tabular credit data |
| **Calibration** | Platt Scaling (sklearn) | Converts raw scores to true probabilities |
| **Explainability** | SHAP | Per-applicant waterfall plots + feature drivers |
| **API** | FastAPI | High-performance `/score` inference endpoint |
| **UI** | Streamlit | Interactive borrower profile input + results |
| **Data** | Give Me Some Credit (Kaggle) | 150k real borrowers with default outcomes |
| **Containerization** | Docker | Single-image deployment (API + UI via supervisord) |
| **Hosting** | Hugging Face Spaces | Free live demo, no external infra needed |
| **Process Manager** | Supervisord | Runs FastAPI + Streamlit together in one container |

---

## 📊 Dataset

**Give Me Some Credit** — Kaggle Competition Dataset

| Property | Value |
|---|---|
| Total borrowers | ~150,000 |
| Default rate | ~6.7% (imbalanced) |
| Features | 10 financial indicators |
| Target | SeriousDlqin2yrs (90+ day delinquency) |
| Source | Real credit bureau data |

Key features: revolving utilization, age, number of open credit lines,
debt ratio, monthly income, number of dependents, past-due counts
(30/60/90 days), real estate loans.

See [DATA_CARD.md](DATA_CARD.md) for full dataset documentation.
See [MODEL_CARD.md](MODEL_CARD.md) for model performance and limitations.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Kaggle account with Give Me Some Credit competition rules accepted
- Kaggle API credentials at `~/.kaggle/kaggle.json`

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/Ravitejas1596/credit-risk-engine.git
cd credit-risk-engine

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Download Data
```bash
python -m scripts.download_data
```

> Accept the competition rules at
> kaggle.com/competitions/GiveMeSomeCredit before running.

### Train the Model
```bash
python -m scripts.train --seed 42
```

This writes two artifacts:
- `artifacts/model.joblib` — trained + calibrated XGBoost model
- `artifacts/metrics.json` — AUC, accuracy, calibration metrics

### Run Locally
```bash
# Terminal 1 — FastAPI inference server
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit UI
streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0
```

Open [http://localhost:7860](http://localhost:7860)

### Run with Docker
```bash
# Build image
docker build -t credit-risk-engine .

# Run container
docker run -p 7860:7860 credit-risk-engine
```

### Deploy to Hugging Face Spaces
```
1. Create a new Docker Space on huggingface.co/spaces
2. Push this repo to the Space repository
3. Streamlit UI serves on port 7860 automatically
4. No additional configuration needed
```

---

## 📁 Project Structure
```
credit-risk-engine/
├── app/
│   ├── api.py                 # FastAPI /score endpoint
│   └── streamlit_app.py       # Streamlit UI — score + SHAP view
│
├── scripts/
│   ├── download_data.py       # Kaggle data downloader
│   └── train.py               # Training + calibration pipeline
│
├── artifacts/                 # Saved model + metrics (post-training)
├── reports/                   # EDA + evaluation reports
│
├── DATA_CARD.md               # Dataset documentation
├── MODEL_CARD.md              # Model performance + limitations
├── Dockerfile                 # Docker build (HuggingFace compatible)
├── supervisord.conf           # Runs FastAPI + Streamlit together
└── requirements.txt
```

---

## 🌐 Live Demo

**Try it → [https://huggingface.co/spaces/ravitejas1596/credit-risk-engine](https://huggingface.co/spaces/ravitejas1596/credit-risk-engine)**

> Enter a borrower profile, get an instant risk score, and see exactly
> which factors drove the decision — no setup needed.

---

## 🗺️ Roadmap

- [x] XGBoost baseline with feature engineering
- [x] Probability calibration (Platt scaling)
- [x] SHAP waterfall plots + top feature drivers
- [x] FastAPI inference endpoint
- [x] Docker + HuggingFace Spaces deployment
- [ ] LightGBM / logistic regression comparison
- [ ] Scorecard-style output (points-based, like FICO)
- [ ] Fairness audit — bias across age and income groups
- [ ] Batch scoring endpoint for portfolio analysis
- [ ] REST API authentication + rate limiting

---

## ⚠️ Limitations

- Trained on a single dataset — performance varies across lending markets
- Calibration is approximate — not a substitute for actuarial validation
- Model does not account for macroeconomic conditions or time drift
- See [MODEL_CARD.md](MODEL_CARD.md) for full details

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Kaggle + FICO](https://www.kaggle.com/competitions/GiveMeSomeCredit) — for the dataset
- [XGBoost](https://xgboost.readthedocs.io) — for the model
- [SHAP](https://shap.readthedocs.io) — for explainability
- [Hugging Face](https://huggingface.co) — for free model hosting

---

<p align="center">
  Built for anyone who believes a declined loan application
  deserves more than a form letter.
</p>
