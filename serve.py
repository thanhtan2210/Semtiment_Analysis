from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent / "models" / "sentiment_pipeline.joblib"

app = FastAPI(title="Sentiment Analysis API", version="0.1.0")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: Optional[float] = None


@app.on_event("startup")
def load_model():
    global pipe
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Please run train.py first.")
    pipe = joblib.load(MODEL_PATH)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Pipeline(TfidfVectorizer -> LogisticRegression)
    label = pipe.predict([req.text])[0]
    score = None
    # If classifier supports predict_proba, return the probability of predicted label
    clf = pipe.named_steps.get("clf")
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(
            pipe.named_steps["tfidf"].transform([req.text]))
        idx = clf.classes_.tolist().index(label)
        score = float(proba[0][idx])
    return PredictResponse(label=str(label), score=score)


# Run with: uvicorn serve:app --reload --port 8000
