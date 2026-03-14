import time
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Setup path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.common import config

def benchmark():
    print("Starting Performance Benchmark: Scalar UDF vs Pandas UDF...")
    try:
        model = joblib.load(config.MODEL_PATH)
    except Exception as e:
        print(f"Error: Could not load model: {e}")
        return

    # Create a dummy dataset of 5,000 records
    sample_texts = [
        "I love this product!", 
        "This is terrible, I hate it.", 
        "It was just okay, nothing special.", 
        "Feeling very happy today!", 
        "The weather is quite gloomy."
    ]
    data = sample_texts * 1000
    df = pd.DataFrame(data, columns=["text"])
    
    # 1. Benchmark Scalar (Looping)
    start_time = time.time()
    scalar_preds = [model.predict([t])[0] for t in df["text"]]
    scalar_duration = time.time() - start_time
    print(f"Scalar (Row-by-Row) Processing: {scalar_duration:.4f} seconds")

    # 2. Benchmark Vectorized (Pandas-style)
    start_time = time.time()
    # model.predict is vectorized in sklearn
    pandas_preds = model.predict(df["text"].tolist())
    pandas_duration = time.time() - start_time
    print(f"Pandas (Vectorized) Processing: {pandas_duration:.4f} seconds")

    speedup = scalar_duration / pandas_duration
    print(f"\nConclusion: Vectorized processing is {speedup:.2f}x faster!")
    
    return scalar_duration, pandas_duration, speedup

if __name__ == "__main__":
    benchmark()
