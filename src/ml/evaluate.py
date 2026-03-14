import argparse
import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Setup path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import config, logger as common_logger

logger = common_logger.get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment model")
    parser.add_argument("--input", type=str, default=config.CSV_DATA_PATH)
    parser.add_argument("--model-path", type=str, default=config.MODEL_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        logger.error("Model not found.")
        return

    df = pd.read_csv(args.input).dropna(how="any")
    # Assuming standard columns for simplicity in eval script
    X = df['Text'].astype(str)
    y = df['Sentiment']

    pipe = joblib.load(args.model_path)
    y_pred = pipe.predict(X)

    acc = accuracy_score(y, y_pred)
    logger.info(f"Evaluation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
