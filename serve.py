import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import joblib
from pathlib import Path

# Cấu hình
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "sentiment_db"
COLLECTION_NAME = "results"
MODEL_PATH = Path(__file__).parent / "models" / "sentiment_pipeline.joblib"

app = FastAPI(title="Real-time Sentiment Analysis API", version="0.2.0")

# Kết nối MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

class SentimentRecord(BaseModel):
    text: str
    user: str
    platform: str
    timestamp: str
    sentiment: str

class Stats(BaseModel):
    sentiment_counts: dict

@app.get("/")
def read_root():
    return {"message": "Welcome to Real-time Sentiment Analysis API"}

@app.get("/latest", response_model=List[SentimentRecord])
def get_latest(limit: int = 10):
    """Lấy các bản ghi mới nhất từ MongoDB"""
    records = list(collection.find().sort("_id", -1).limit(limit))
    for r in records:
        r["_id"] = str(r["_id"]) # Convert ObjectId to string
    return records

@app.get("/stats", response_model=Stats)
def get_stats():
    """Thống kê số lượng cảm xúc"""
    pipeline = [
        {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}}
    ]
    results = list(collection.aggregate(pipeline))
    counts = {r["_id"]: r["count"] for r in results}
    return {"sentiment_counts": counts}

# Giữ lại tính năng dự đoán đơn lẻ nếu cần
@app.post("/predict")
def predict(text: str):
    if not MODEL_PATH.exists():
        return {"error": "Model not found. Run train.py first."}
    
    pipe = joblib.load(MODEL_PATH)
    label = pipe.predict([text])[0]
    return {"text": text, "prediction": str(label)}
