import streamlit as st
import pandas as pd
from pymongo import MongoClient
import time
import os
import plotly.express as px

# 1. Cấu hình MongoDB
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "sentiment_db"
COLLECTION_NAME = "results"

st.set_page_config(page_title="Real-time Sentiment Monitor", layout="wide")

st.title("📊 Real-time Sentiment Analysis Dashboard")
st.markdown("Hệ thống giám sát cảm xúc người dùng từ mạng xã hội (Kafka + Spark + MongoDB)")

# 2. Kết nối MongoDB
@st.cache_resource
def get_db_connection():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]

db_collection = get_db_connection()

# 3. Tạo layout cho Dashboard
col1, col2 = st.columns(2)
placeholder_stats = st.empty()
placeholder_charts = st.empty()

def load_data():
    data = list(db_collection.find().sort("_id", -1).limit(1000))
    if data:
        df = pd.DataFrame(data)
        df["_id"] = df["_id"].astype(str)
        return df
    return pd.DataFrame()

# 4. Vòng lặp cập nhật thời gian thực
while True:
    df = load_data()
    
    with placeholder_stats.container():
        if not df.empty:
            total_msgs = len(df)
            st.metric("Tổng số tin nhắn (1000 bản ghi mới nhất)", total_msgs)
            
            # Phân tách 2 cột cho các biểu đồ chính
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Cảm xúc phổ biến")
                sentiment_counts = df['sentiment'].value_counts()
                fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                st.subheader("Hoạt động theo quốc gia")
                country_counts = df['country'].value_counts().head(10)
                fig_bar = px.bar(x=country_counts.index, y=country_counts.values, labels={'x':'Quốc gia', 'y':'Số lượng'})
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("Dữ liệu chi tiết mới nhất")
            st.dataframe(df[['text', 'sentiment', 'country', 'platform']].head(10), use_container_width=True)
        else:
            st.warning("Đang chờ dữ liệu từ Spark Streaming... Hãy chạy producer.py và spark_processor.py")

    time.sleep(3) # Cập nhật mỗi 3 giây
