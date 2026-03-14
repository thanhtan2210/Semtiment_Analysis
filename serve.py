import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import joblib
from pathlib import Path

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "sentiment_db"
COLLECTION_NAME = "results"
MODEL_PATH = Path(__file__).parent / "models" / "sentiment_pipeline.joblib"

app = FastAPI(title="Real-time Sentiment Analysis API", version="0.2.0")

# Kết nối MongoDB
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    # Check connection
    client.server_info()
    logger.info(f"Connected to MongoDB at {MONGO_URI}")
except Exception as e:
    logger.error(f"Could not connect to MongoDB: {e}")

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
    return {"status": "online", "message": "Real-time Sentiment Analysis API is running"}

@app.get("/latest", response_model=List[SentimentRecord])
def get_latest(limit: int = 10):
    """Lấy các bản ghi mới nhất từ MongoDB"""
    try:
        records = list(collection.find().sort("_id", -1).limit(limit))
        for r in records:
            r["_id"] = str(r["_id"])
        logger.info(f"Retrieved {len(records)} latest records.")
        return records
    except Exception as e:
        logger.error(f"Error fetching records: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/stats", response_model=Stats)
def get_stats():
    """Thống kê số lượng cảm xúc"""
    try:
        pipeline = [
            {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}}
        ]
        results = list(collection.aggregate(pipeline))
        counts = {r["_id"]: r["count"] for r in results}
        logger.info("Aggregated sentiment statistics.")
        return {"sentiment_counts": counts}
    except Exception as e:
        logger.error(f"Error aggregating stats: {e}")
        raise HTTPException(status_code=500, detail="Aggregation error")

@app.post("/predict")
def predict(text: str):
    if not MODEL_PATH.exists():
        logger.warning("Predict endpoint called but model file is missing.")
        return {"error": "Model not found. Run train.py first."}
    
    try:
        pipe = joblib.load(MODEL_PATH)
        label = pipe.predict([text])[0]
        logger.info(f"Single prediction made for text: {text[:30]}...")
        return {"text": text, "prediction": str(label)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}
