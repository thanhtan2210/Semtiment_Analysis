import joblib
import pandas as pd
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType

def test_pandas_udf():
    # Cấu hình đường dẫn Python cho Spark (Quan trọng khi dùng venv)
    python_path = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

    print(f"Sử dụng Python tại: {python_path}")
    print("Khởi tạo Spark Session cục bộ...")
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
        print("Lỗi: Không tìm thấy mô hình. Hãy chắc chắn bạn đã chạy train.py trước đó.")
        return

    @pandas_udf(StringType())
    def predict_sentiment_pandas_udf(text_series: pd.Series) -> pd.Series:
        clean_series = text_series.fillna("")
        predictions = model.predict(clean_series.tolist())
        return pd.Series(predictions)

    # Tạo DataFrame giả lập
    data = [("Hôm nay thời tiết thật tuyệt vời",), ("Sản phẩm này quá tệ",), ("Bình thường thôi",), ("",)]
    df = spark.createDataFrame(data, ["text"])

    print("\nDữ liệu gốc:")
    df.show()

    # Áp dụng Pandas UDF
    result_df = df.withColumn("sentiment", predict_sentiment_pandas_udf(col("text")))

    print("\nKết quả sau khi dùng Pandas UDF:")
    result_df.show()

if __name__ == "__main__":
    test_pandas_udf()
