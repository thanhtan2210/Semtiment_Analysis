import joblib
import pandas as pd
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType

def test_pandas_udf():
    # Python path configuration for Spark (Crucial for virtual environments)
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

    print(f"Using Python at: {python_path}")
    print("Initializing local Spark Session...")
    spark = SparkSession.builder \
        .appName("TestPandasUDF") \
        .master("local[1]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    try:
        model = joblib.load("models/sentiment_pipeline.joblib")
    except Exception as e:
        print("Error: Model not found. Please ensure you have run train.py.")
        return

    @pandas_udf(StringType())
    def predict_sentiment_pandas_udf(text_series: pd.Series) -> pd.Series:
        clean_series = text_series.fillna("")
        predictions = model.predict(clean_series.tolist())
        return pd.Series(predictions)

    # Create dummy DataFrame
    data = [("Today the weather is wonderful",), ("This product is terrible",), ("It's just okay",), ("",)]
    df = spark.createDataFrame(data, ["text"])

    print("\nOriginal Data:")
    df.show()

    # Apply Pandas UDF
    result_df = df.withColumn("sentiment", predict_sentiment_pandas_udf(col("text")))

    print("\nResult after using Pandas UDF:")
    result_df.show()

if __name__ == "__main__":
    test_pandas_udf()
