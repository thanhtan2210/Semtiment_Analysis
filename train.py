import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn


DEFAULT_TEXT_CANDIDATES = [
    "text",
    "review",
    "comment",
    "content",
    "sentence",
    "message",
]
DEFAULT_LABEL_CANDIDATES = [
    "label",
    "sentiment",
    "target",
    "class",
]


def detect_columns(df: pd.DataFrame, text_col: str | None, label_col: str | None) -> tuple[str, str]:
    cols_lower = {c.lower(): c for c in df.columns}
    if text_col:
        tc = cols_lower.get(text_col.lower(), text_col)
    else:
        tc = next((cols_lower.get(c)
                  for c in DEFAULT_TEXT_CANDIDATES if c in cols_lower), None)
    if label_col:
        lc = cols_lower.get(label_col.lower(), label_col)
    else:
        lc = next((cols_lower.get(c)
                  for c in DEFAULT_LABEL_CANDIDATES if c in cols_lower), None)
    if not tc or not lc:
        raise ValueError(
            f"Could not detect columns. Text: {text_col or DEFAULT_TEXT_CANDIDATES}, Label: {label_col or DEFAULT_LABEL_CANDIDATES}. Available: {list(df.columns)}"
        )
    return tc, lc


def normalize_labels(series: pd.Series) -> pd.Series:
    # Chuẩn hóa nhãn phổ biến: positive/negative/neutral
    mapping = {
        "pos": "positive",
        "positive": "positive",
        "neg": "negative",
        "negative": "negative",
        "neu": "neutral",
        "neutral": "neutral",
        "0": "negative",
        "1": "positive",
    }
    return series.astype(str).str.lower().map(lambda x: mapping.get(x, x))


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    strip_accents=None,
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=50000,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, n_jobs=None),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a simple sentiment analysis model (TF-IDF + LogisticRegression)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="sentimentdataset.csv",
        help="Path to input CSV (default: sentimentdataset.csv in this project)",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="Text column name. If omitted, auto-detects common names (text/review/comment/...)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label column name. If omitted, auto-detects common names (label/sentiment/target/...)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    parser.add_argument(
        "--normalize-labels",
        action="store_true",
        help="Normalize labels to positive/negative/neutral when possible",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/sentiment_pipeline.joblib",
        help="Output path to save model (default: models/sentiment_pipeline.joblib)",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="models/metrics.json",
        help="Output path to save metrics (default: models/metrics.json)",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: file:./mlruns)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sentiment-analysis",
        help="MLflow experiment name (default: sentiment-analysis)",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input data file not found: {in_path}")

    df = pd.read_csv(in_path)
    df = df.dropna(how="any")

    text_col, label_col = detect_columns(df, args.text_col, args.label_col)

    y = df[label_col]
    if args.normalize_labels:
        y = normalize_labels(y)

    X = df[text_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if y.nunique() > 1 else None
    )

    # Configure MLflow tracking (local file store under project directory)
    # Configure MLflow tracking (local file store under project directory by default)
    tracking_dir = Path(__file__).parent / "mlruns"
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    else:
        mlflow.set_tracking_uri(f"file:{tracking_dir}")
    mlflow.set_experiment(args.experiment_name)

    pipe = build_pipeline()

    # Prepare output paths for model/metrics besides MLflow artifacts
    out_model = Path(args.model_out)
    out_metrics = Path(args.metrics_out)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="tfidf-logreg"):
        # Log các tham số chính
        mlflow.log_params({
            "test_size": args.test_size,
            "random_state": args.random_state,
            "normalize_labels": bool(args.normalize_labels),
            "tfidf_ngram_min": 1,
            "tfidf_ngram_max": 2,
            "tfidf_max_features": 50000,
            "clf": "LogisticRegression",
            "clf_max_iter": 1000,
        })

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        report = classification_report(y_test, y_pred, output_dict=True)

        joblib.dump(pipe, out_model)
        with open(out_metrics, "w", encoding="utf-8") as f:
            json.dump({"accuracy": acc, "f1_weighted": f1,
                      "report": report}, f, ensure_ascii=False, indent=2)

        # Log metrics and artifacts to MLflow
        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})
        mlflow.log_artifact(str(out_metrics))
        # Log model (save a copy into MLflow artifacts)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

    print("Training completed.")
    print(f"Model: {out_model}")
    print(f"Metrics: {out_metrics}")
    print(f"Accuracy: {acc:.4f} | F1 (weighted): {f1:.4f}")


if __name__ == "__main__":
    main()
