import json
import time
import pandas as pd
from kafka import KafkaProducer
from pathlib import Path

# Cấu hình Kafka
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092' # Thay đổi thành 'kafka:9092' nếu chạy trong Docker
TOPIC_NAME = 'social_media_stream'
CSV_FILE = Path(__file__).parent / "sentimentdataset.csv"

def get_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        return producer
    except Exception as e:
        print(f"Error connecting to Kafka: {e}")
        return None

def main():
    if not CSV_FILE.exists():
        print(f"File not found: {CSV_FILE}")
        return

    # Đọc dữ liệu
    df = pd.read_csv(CSV_FILE)
    print(f"Successfully loaded {len(df)} records from CSV.")

    producer = get_producer()
    if not producer:
        return

    print(f"Starting to stream data to Kafka topic: {TOPIC_NAME}...")

    for _, row in df.iterrows():
        # Tạo payload dữ liệu
        data = {
            "text": str(row.get("Text", "")).strip(),
            "user": str(row.get("User", "")).strip(),
            "platform": str(row.get("Platform", "")).strip(),
            "timestamp": str(row.get("Timestamp", "")),
            "country": str(row.get("Country", "")).strip()
        }

        # Gửi dữ liệu
        producer.send(TOPIC_NAME, value=data)
        print(f"Sent: {data['text'][:50]}...")
        
        # Giả lập thời gian thực (0.5s mỗi tin nhắn)
        time.sleep(0.5)

    producer.flush()
    print("Streaming finished.")

if __name__ == "__main__":
    main()
