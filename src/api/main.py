import os
import sys
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
import joblib

# Setup path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import config, logger as common_logger

logger = common_logger.get_logger(__name__)

app = FastAPI(title="Real-time Sentiment Analysis API", version="0.2.0")

# MongoDB Connection
try:
    client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[config.DB_NAME]
    collection = db[config.COLLECTION_NAME]
    client.server_info()
    logger.info(f"Connected to MongoDB at {config.MONGO_URI}")
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
    try:
        records = list(collection.find().sort("_id", -1).limit(limit))
        for r in records:
            r["_id"] = str(r["_id"])
        return records
    except Exception as e:
        logger.error(f"Error fetching records: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/stats", response_model=Stats)
def get_stats():
    try:
        pipeline = [{"$group": {"_id": "$sentiment", "count": {"$sum": 1}}}]
        results = list(collection.aggregate(pipeline))
        counts = {r["_id"]: r["count"] for r in results}
        return {"sentiment_counts": counts}
    except Exception as e:
        logger.error(f"Error aggregating stats: {e}")
        raise HTTPException(status_code=500, detail="Aggregation error")

@app.post("/predict")
def predict(text: str):
    if not os.path.exists(config.MODEL_PATH):
        return {"error": "Model not found. Run train.py first."}
    
    try:
        pipe = joblib.load(config.MODEL_PATH)
        label = pipe.predict([text])[0]
        return {"text": text, "prediction": str(label)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}
