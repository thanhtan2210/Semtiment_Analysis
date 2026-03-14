import os
import joblib
import pandas as pd
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "social_media_stream"
MONGO_URI = "mongodb://localhost:27017/sentiment_db.results"
MODEL_PATH = "models/sentiment_pipeline.joblib"

# 2. Schema definition for Kafka data
schema = StructType([
    StructField("text", StringType()),
    StructField("user", StringType()),
    StructField("platform", StringType()),
    StructField("timestamp", StringType()),
    StructField("country", StringType())
])

def main():
    # 3. Initialize Spark Session with Kafka and MongoDB connectors
    logger.info("Starting Spark Session with Kafka & MongoDB connectors...")
    spark = SparkSession.builder \
        .appName("SentimentAnalysisStreaming") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.mongodb.output.uri", MONGO_URI) \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # 4. Load NLP model (scikit-learn pipeline)
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading pre-trained model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
    else:
        logger.error(f"Model not found at {MODEL_PATH}. Please run train.py first.")
        return

    # 5. Create Pandas UDF (Vectorized UDFs) for performance optimization
    @pandas_udf(StringType())
    def predict_sentiment_pandas_udf(text_series: pd.Series) -> pd.Series:
        clean_series = text_series.fillna("")
        predictions = model.predict(clean_series.tolist())
        return pd.Series(predictions)

    # 6. Read data stream from Kafka
    logger.info(f"Subscribing to Kafka topic: {KAFKA_TOPIC}...")
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # 7. Decode JSON data and apply NLP model
    json_df = df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    # Data Cleaning (filter out noise)
    clean_df = json_df.filter(
        (col("text").isNotNull()) & (col("text") != "")
    )

    results_df = clean_df.withColumn("sentiment", predict_sentiment_pandas_udf(col("text")))

    # 8. Write data to MongoDB
    logger.info("Initializing streaming query to MongoDB...")
    query = results_df.writeStream \
        .foreachBatch(lambda batch_df, batch_id: batch_df.write \
            .format("mongo") \
            .mode("append") \
            .save()) \
        .option("checkpointLocation", "checkpoints/sentiment_analysis") \
        .start()

    logger.info(f"Spark Streaming is active and processing '{KAFKA_TOPIC}'...")
    query.awaitTermination()

if __name__ == "__main__":
    main()
