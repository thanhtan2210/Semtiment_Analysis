import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved sentiment model")
    parser.add_argument(
        "--input",
        type=str,
        default="sentimentdataset.csv",
        help="Path to evaluation CSV",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/sentiment_pipeline.joblib",
        help="Path to saved model",
    )
    parser.add_argument("--text-col", type=str,
                        default=None, help="Text column name")
    parser.add_argument("--label-col", type=str,
                        default=None, help="Label column name")
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="models/metrics_eval.json",
        help="Output path to save evaluation metrics",
    )

    args = parser.parse_args()
    df = pd.read_csv(args.input).dropna(how="any")
    text_col, label_col = detect_columns(df, args.text_col, args.label_col)

    X = df[text_col].astype(str)
    y = df[label_col]

    pipe = joblib.load(args.model_path)
    y_pred = pipe.predict(X)

    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, average="weighted"))
    report = classification_report(y, y_pred, output_dict=True)

    out_metrics = Path(args.metrics_out)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1,
                  "report": report}, f, ensure_ascii=False, indent=2)

    print("Evaluation completed.")
    print(f"Accuracy: {acc:.4f} | F1 (weighted): {f1:.4f}")
    print(f"Metrics: {out_metrics}")


if __name__ == "__main__":
    main()
