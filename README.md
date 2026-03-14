# Real-time Sentiment Analysis Pipeline

This project implements a high-performance, professional end-to-end sentiment analysis pipeline designed for real-time social media monitoring. It utilizes a modern Big Data stack and follows Data Engineering best practices such as modular architecture, data validation, and vectorized processing.

## Architecture Overview

The system is built on a modular architecture to ensure scalability and maintainability:

1.  **Ingestion Layer (`src/ingestion/`)**: A Kafka Producer that simulates high-frequency social media streams. It uses **Pydantic** for schema validation to ensure data quality at the source.
2.  **Streaming Layer (`src/processing/`)**: An Apache Spark Streaming application that consumes data from Kafka. It leverages **Pandas UDFs** (Vectorized UDFs) for high-throughput sentiment inference.
3.  **Storage Layer**: Results are stored in **MongoDB** for rapid analytical queries and persistence.
4.  **Serving Layer (`src/api/`)**: A FastAPI service providing RESTful access to the latest sentiment insights and statistics.
5.  **Observation Layer (`src/dashboard/`)**: A real-time Streamlit dashboard for visual monitoring of user emotions and activity.
6.  **MLOps Layer (`src/ml/`)**: Model training and evaluation scripts with **MLflow** for experiment tracking and model lineage.

## Performance & Metrics

The pipeline has been optimized for low-latency and high-throughput processing. Below are the key performance indicators from our internal benchmarks:

### 1. Processing Speed (Inference)
By migrating from traditional Scalar UDFs to **Vectorized Pandas UDFs**, we achieved a massive speedup in our Spark processing stage:

| Inference Method | Time (5,000 records) | Throughput (records/sec) |
| :--- | :--- | :--- |
| Row-by-Row (Scalar) | ~70.13 seconds | ~71 rec/s |
| **Vectorized (Pandas)** | **~0.15 seconds** | **~33,300 rec/s** |
| **Improvement** | **478x Faster** | |

### 2. Model Performance
- **Model Type**: TF-IDF + Logistic Regression (Pipeline).
- **Target**: Multi-class emotion classification (over 30 distinct emotional labels).
- **Validation**: Accuracy logs and metrics are tracked automatically via **MLflow**.

## Project Structure

```text
Semtiment_Analysis/
├── data/                   # Raw data and CSV files
├── src/                    # Main source code
│   ├── common/             # Shared configuration and logging
│   ├── ingestion/          # Kafka Producer & Data validation
│   ├── processing/         # Spark Streaming & Vectorized inference
│   ├── api/                # FastAPI serving layer
│   ├── dashboard/          # Real-time Streamlit dashboard
│   └── ml/                 # Model training and evaluation
├── models/                 # Serialized model artifacts (.joblib)
├── mlruns/                 # MLflow experiment tracking data
├── docker-compose.yml      # Infrastructure orchestration (Kafka, Mongo, Spark)
├── requirements.txt        # Project dependencies
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Java 17+ (for local Spark execution)

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Start Infrastructure
Launch Kafka, Zookeeper, MongoDB, and Spark using Docker:
```bash
docker-compose up -d
```

### 3. Train the Model
```bash
make train
```

### 4. Run the Pipeline
In separate terminals, run the streaming components:
```bash
make producer    # Start streaming data to Kafka
make processor   # Start processing with Spark
```

### 5. Monitor Live Insights
- **Dashboard**: `make dashboard` (Access at http://localhost:8501)
- **API**: `make api` (Access at http://localhost:8000)
- **MLflow**: `mlflow ui` (Access at http://localhost:5000)

## Key Technical Features
- **Data Governance**: Pydantic models enforce strict schema validation, ensuring no "garbage" data enters the processing pipeline.
- **Performance Optimization**: Use of PyArrow and Pandas UDFs in Spark minimizes JVM-Python overhead.
- **Observability**: Centralized structured logging and automated MLflow tracking for full lifecycle monitoring.
- **Modular Design**: Clean separation of concerns allows for independent scaling of each pipeline stage.
