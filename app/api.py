from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.model import CreditRiskEngine


ARTIFACTS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"

app = FastAPI(title="Credit Risk Engine", version="1.0.0")
engine: Optional[CreditRiskEngine] = None


class CreditProfile(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0)
    age: int = Field(..., ge=18, le=100)
    NumberOfTime30-59DaysPastDueNotWorse: int = Field(..., ge=0, le=98)
    DebtRatio: float = Field(..., ge=0)
    MonthlyIncome: float = Field(..., ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(..., ge=0, le=98)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    NumberOfTime60-89DaysPastDueNotWorse: int = Field(..., ge=0, le=98)
    NumberOfDependents: int = Field(..., ge=0)


@app.on_event("startup")
def _load() -> None:
    global engine
    if ARTIFACTS_PATH.exists():
        engine = CreditRiskEngine(ARTIFACTS_PATH)
    else:
        engine = None


@app.get("/health")
def health() -> Dict[str, Any]:
    ok = ARTIFACTS_PATH.exists()
    return {"ok": ok, "artifacts_path": str(ARTIFACTS_PATH)}


@app.post("/score")
def score(profile: CreditProfile) -> Dict[str, Any]:
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train first (python -m scripts.train) to create artifacts/model.joblib.",
        )
    res = engine.score(profile.model_dump(), top_k=8)
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

