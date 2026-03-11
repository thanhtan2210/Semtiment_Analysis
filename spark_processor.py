import os
import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, StringType

# 1. Cấu hình
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "social_media_stream"
MONGO_URI = "mongodb://localhost:27017/sentiment_db.results"
MODEL_PATH = "models/sentiment_pipeline.joblib"

# 2. Định nghĩa Schema cho dữ liệu Kafka
schema = StructType([
    StructField("text", StringType()),
    StructField("user", StringType()),
    StructField("platform", StringType()),
    StructField("timestamp", StringType()),
    StructField("country", StringType())
])

def main():
    # 3. Khởi tạo Spark Session với Kafka và MongoDB connector
    spark = SparkSession.builder \
        .appName("SentimentAnalysisStreaming") \
        .config("spark.mongodb.output.uri", MONGO_URI) \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # 4. Tải mô hình NLP (scikit-learn pipeline)
    if os.path.exists(MODEL_PATH):
        print(f"Loading pre-trained model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
    else:
        print("Model not found. Please run train.py first.")
        return

    # 5. Tạo UDF (User Defined Function) để Spark có thể chạy model scikit-learn
    def predict_sentiment(text):
        if text is None or text == "":
            return "Neutral"
        prediction = model.predict([text])[0]
        return str(prediction)

    sentiment_udf = udf(predict_sentiment, StringType())

    # 6. Đọc luồng dữ liệu từ Kafka
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # 7. Giải mã dữ liệu JSON và áp dụng mô hình NLP
    json_df = df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    results_df = json_df.withColumn("sentiment", sentiment_udf(col("text")))

    # 8. Ghi dữ liệu vào MongoDB
    query = results_df.writeStream \
        .foreachBatch(lambda batch_df, batch_id: batch_df.write \
            .format("mongo") \
            .mode("append") \
            .save()) \
        .start()

    print(f"Spark Streaming is running and processing '{KAFKA_TOPIC}'...")
    query.awaitTermination()

if __name__ == "__main__":
    main()
