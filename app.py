from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib

from feature_extractor import extract_features

MODEL_PATH = "fraud_model.pkl"
pipe = joblib.load(MODEL_PATH)

app = FastAPI(title="Secure Bharat - Fraud Call Detector", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    numbers: List[str]

class PredictCallBody(BaseModel):
    caller: str
    transcript: Optional[str] = None

SUSPICIOUS_PHRASES = [
    "kyc", "otp", "block your account", "lottery", "refund", "income tax",
    "upi", "pan card", "bank verification", "verification code",
    "install app", "remote access", "teamviewer", "anydesk",
    "prize", "seized", "customs", "package", "credit card", "urgent",
]

def risk_from_proba(p: float) -> str:
    if p >= 0.75:
        return "high"
    if p >= 0.45:
        return "medium"
    return "low"

@app.get("/")
def root():
    return {"message": "Secure Bharat - Fraud Call API is running. See /docs, /health, /predict, /predict_call."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(number: str = Query(..., description="Phone number, e.g., +911234567890")):
    feats = extract_features(number)
    df = pd.DataFrame([feats])
    proba = float(pipe.predict_proba(df)[0][1])
    label = "fraud" if proba >= 0.5 else "legit"
    risk = risk_from_proba(proba)
    return {"phone_number": number, "label": label, "risk": risk, "confidence": proba}

@app.post("/predict_batch")
def predict_batch(req: PredictRequest):
    rows = [extract_features(n) for n in req.numbers]
    df = pd.DataFrame(rows)
    probas = pipe.predict_proba(df)[:, 1]
    results = []
    for i, p in enumerate(probas):
        label = "fraud" if p >= 0.5 else "legit"
        risk = risk_from_proba(float(p))
        results.append({"phone_number": req.numbers[i], "label": label, "risk": risk, "confidence": float(p)})
    return {"results": results}

@app.post("/predict_call")
def predict_call(body: PredictCallBody):
    feats = extract_features(body.caller)
    df = pd.DataFrame([feats])
    p_num = float(pipe.predict_proba(df)[0][1])

    boost = 0.0
    if body.transcript:
        t = body.transcript.lower()
        hits = sum(1 for kw in SUSPICIOUS_PHRASES if kw in t)
        boost = min(0.07 * hits, 0.35)

    p_final = max(0.0, min(1.0, p_num + boost))
    label = "fraud" if p_final >= 0.5 else "legit"
    risk = risk_from_proba(p_final)

    return {
        "caller": body.caller,
        "label": label,
        "confidence": p_final,
        "risk": risk,
        "components": {"p_number": p_num, "transcript_boost": boost, "hits_capped": boost/0.07 if boost else 0}
    }
