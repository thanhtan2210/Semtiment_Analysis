# Real-time Sentiment Analysis Pipeline

This project implements a professional end-to-end sentiment analysis pipeline using a modern Big Data stack:
- **Kafka**: Message broker for real-time data ingestion.
- **Spark Streaming**: Distributed processing engine for live sentiment inference.
- **MongoDB**: NoSQL database for scalable result storage.
- **FastAPI**: REST API for serving predictions and statistics.
- **Streamlit**: Real-time dashboard for visual insights.
- **MLflow**: Experiment tracking and model management.

## Project Structure
- `sentimentdataset.csv`: Dataset for training and simulation.
- `producer.py`: Kafka Producer for simulating real-time social media streams (includes Pydantic validation).
- `spark_processor.py`: Spark Streaming service using **Pandas UDFs** for optimized inference.
- `train.py`: ML pipeline using TF-IDF + Logistic Regression with MLflow integration.
- `serve.py`: FastAPI backend to retrieve latest results and statistics from MongoDB.
- `dashboard.py`: Streamlit dashboard for real-time visualization.
- `models/`: Directory for the serialized model and metrics.

## Requirements
- Python 3.10+
- Apache Kafka & MongoDB (managed via Docker)
- Apache Spark 3.5.0

## Setup
1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Infrastructure:
```bash
docker-compose up -d
```

## Usage Workflow

### 1. Training
Train the model on the provided dataset:
```bash
python train.py --input sentimentdataset.csv --normalize-labels
```

### 2. Live Pipeline
Start the streaming components in separate terminals:

- **Producer**: Stream data to Kafka
```bash
python producer.py
```

- **Spark Processor**: Process and analyze live data
```bash
python spark_processor.py
```

### 3. Monitoring
- **Dashboard**: View live insights
```bash
streamlit run dashboard.py
```

- **API**: Access raw data
```bash
uvicorn serve:app --reload --port 8000
```

## Key Features
- **Data Quality**: Pydantic validation at the source to prevent corrupted data entry.
- **Performance**: Vectorized Pandas UDFs in Spark for high-throughput inference.
- **Scalability**: Fully containerized infrastructure using Docker Compose.
- **Observability**: Structured logging and MLflow tracking for model lineage.
