import streamlit as st
import pandas as pd
from pymongo import MongoClient
import time
import os
import plotly.express as px

# 1. MongoDB Configuration
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "sentiment_db"
COLLECTION_NAME = "results"

st.set_page_config(page_title="Real-time Sentiment Monitor", layout="wide")

st.title("📊 Real-time Sentiment Analysis Dashboard")
st.markdown("Monitoring user emotions from social media (Kafka + Spark + MongoDB)")

# 2. MongoDB Connection
@st.cache_resource
def get_db_connection():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]

db_collection = get_db_connection()

# 3. Dashboard Layout
placeholder_stats = st.empty()

def load_data():
    data = list(db_collection.find().sort("_id", -1).limit(1000))
    if data:
        df = pd.DataFrame(data)
        df["_id"] = df["_id"].astype(str)
        return df
    return pd.DataFrame()

# 4. Real-time Update Loop
while True:
    df = load_data()
    
    with placeholder_stats.container():
        if not df.empty:
            total_msgs = len(df)
            st.metric("Total Messages (Last 1000 records)", total_msgs)
            
            # Main charts in two columns
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Common Emotions")
                sentiment_counts = df['sentiment'].value_counts()
                fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                st.subheader("Activity by Country")
                country_counts = df['country'].value_counts().head(10)
                fig_bar = px.bar(x=country_counts.index, y=country_counts.values, labels={'x':'Country', 'y':'Count'})
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("Latest Detailed Data")
            st.dataframe(df[['text', 'sentiment', 'country', 'platform']].head(10), use_container_width=True)
        else:
            st.warning("Waiting for data from Spark Streaming... Please run producer.py and spark_processor.py")

    time.sleep(3) # Refresh every 3 seconds
