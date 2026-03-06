# Sentiment Analysis Project

End-to-end sentiment analysis with a simple and reproducible pipeline:
- Train a TF‑IDF + LogisticRegression model from a CSV file
- Evaluate a saved model
- Serve predictions via a FastAPI endpoint
- Track params/metrics/artifacts with MLflow
- Optional Docker image (multi-stage, small runtime)

## Project Structure
- sentimentdataset.csv: sample dataset (CSV)
- train.py: training script (TF‑IDF + LogisticRegression)
- evaluate.py: evaluation script for a saved model
- serve.py: FastAPI service exposing `/predict`
- models/: saved model (`sentiment_pipeline.joblib`) and metrics (`metrics.json`)
- mlruns/: MLflow local tracking directory (auto-created)

## Requirements
- Python 3.10+
- Virtual environment recommended (workspace has `.venv`)

## Setup
```bash
pip install -r Semtiment_Analysis/requirements.txt
```

## Train
Auto-detects common text/label column names. Use flags to specify custom names if needed.
```bash
python Semtiment_Analysis/train.py --input Semtiment_Analysis/sentimentdataset.csv --normalize-labels
```
Options:
- `--text-col <text_column>`
- `--label-col <label_column>`
- `--test-size 0.2` (default 0.2)
- `--model-out Semtiment_Analysis/models/sentiment_pipeline.joblib`
- `--metrics-out Semtiment_Analysis/models/metrics.json`

## Evaluate
```bash
python Semtiment_Analysis/evaluate.py --input Semtiment_Analysis/sentimentdataset.csv --model-path Semtiment_Analysis/models/sentiment_pipeline.joblib
```

## MLflow (Try it)
Training logs params, metrics, and artifacts to `Semtiment_Analysis/mlruns`.
Start the UI:
```bash
mlflow ui --backend-store-uri "file:Semtiment_Analysis/mlruns" --port 5000
```
Open http://localhost:5000 to explore runs and artifacts.

## Serve API (FastAPI)
```bash
uvicorn Semtiment_Analysis.serve:app --reload --port 8000
```
Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```
Response:
```json
{
  "label": "positive",
  "score": 0.94
}
```

## Docker (multi-stage)
After training (model at `Semtiment_Analysis/models/sentiment_pipeline.joblib`), build a small image:
```bash
docker build -t sentiment-api -f Semtiment_Analysis/Dockerfile Semtiment_Analysis
docker run --rm -p 8000:8000 sentiment-api
```
Notes:
- The build bakes `models/` into the image. If model is missing, the API will fail to start.
- Alternatively mount the model from host:
```bash
docker run --rm -p 8000:8000 -v "$PWD/Semtiment_Analysis/models:/app/models" sentiment-api
```

## Docker Compose (MLflow + API)
You can run MLflow server and the API together locally using `docker-compose` (builds the API image and runs a lightweight MLflow server):

```bash
cd Semtiment_Analysis
docker-compose up --build
```

This will expose:
- MLflow UI: http://localhost:5000
- API: http://localhost:8000

The compose setup mounts `./mlruns` and `./models` into the containers so MLflow artifacts and the baked model are preserved on the host.

If you prefer to use a remote MLflow server, pass `--mlflow-uri` to `train.py` or set `MLFLOW_TRACKING_URI` in the environment before building/running the API.

## CI (GitHub Actions)
Workflow `.github/workflows/sentiment-ci.yml` lints with `black` & `ruff` and runs `pytest` for changes under `Semtiment_Analysis/`.