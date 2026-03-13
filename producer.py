import json
import time
import pandas as pd
from kafka import KafkaProducer
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import logging

# Thiết lập logging chuyên nghiệp
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cấu hình Kafka
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092' 
TOPIC_NAME = 'social_media_stream'
CSV_FILE = Path(__file__).parent / "sentimentdataset.csv"

# Định nghĩa Schema dữ liệu (Data Governance)
class SentimentData(BaseModel):
    text: str = Field(..., min_length=1)
    user: str = Field(default="Unknown")
    platform: str = Field(default="Unknown")
    timestamp: Optional[str] = None
    country: str = Field(default="Unknown")

def get_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        return producer
    except Exception as e:
        logger.error(f"Error connecting to Kafka: {e}")
        return None

def main():
    if not CSV_FILE.exists():
        logger.error(f"File not found: {CSV_FILE}")
        return

    # Đọc dữ liệu
    df = pd.read_csv(CSV_FILE)
    logger.info(f"Successfully loaded {len(df)} records from CSV.")

    producer = get_producer()
    if not producer:
        return

    logger.info(f"Starting to stream data to Kafka topic: {TOPIC_NAME}...")

    success_count = 0
    error_count = 0

    for _, row in df.iterrows():
        try:
            # Thu thập và validate dữ liệu bằng Pydantic
            raw_data = {
                "text": str(row.get("Text", "")).strip(),
                "user": str(row.get("User", "")).strip(),
                "platform": str(row.get("Platform", "")).strip(),
                "timestamp": str(row.get("Timestamp", "")),
                "country": str(row.get("Country", "")).strip()
            }
            
            # Kiểm soát chất lượng (Quality Control)
            validated_data = SentimentData(**raw_data)
            
            # Gửi dữ liệu đã được làm sạch
            producer.send(TOPIC_NAME, value=validated_data.model_dump())
            success_count += 1
            
            if success_count % 50 == 0:
                logger.info(f"Sent {success_count} valid records...")
            
        except ValidationError as ve:
            error_count += 1
            logger.warning(f"Invalid record skipped: {ve.json()}")
        except Exception as e:
            logger.error(f"Error sending record: {e}")

        time.sleep(0.5)

    producer.flush()
    logger.info(f"Streaming finished. Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()
