# Streamlit App: Stock Price and Sentiment Analysis Using FinBERT
import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Title and Introduction
st.title("Stock Price and Sentiment Analysis Using FinBERT")
st.markdown("""
**Author:** Layne Hunt  
This app demonstrates how stock price data and sentiment from online news can be analyzed using FinBERT and combined for further insights.
""")

# Load FinBERT model for sentiment analysis
@st.cache_resource
def load_finbert_model():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_analyzer

sentiment_analyzer = load_finbert_model()

# Step 1: Fetch Stock Price Data
st.subheader("Step 1: Fetch Stock Price Data")
ticker = st.text_input("Enter Stock Ticker:", "GOOG")
start_date_stock = st.date_input("Start Date:", pd.Timestamp("2023-10-22"))
end_date_stock = st.date_input("End Date:", pd.Timestamp("2024-12-03"))

if st.button("Fetch Stock Data"):
    try:
        stock_data = yf.download(ticker, start=start_date_stock, end=end_date_stock)
        stock_data = stock_data.reset_index()
        stock_data['date'] = stock_data['Date'].dt.date
        stock_data_simple = stock_data[['date', 'Adj Close', 'Volume']].rename(columns={'Adj Close': 'adjusts'})
        st.write("Stock Data Sample:", stock_data_simple.head())
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")

# Step 2: Fetch Sentiment Data from NewsAPI
st.subheader("Step 2: Fetch Sentiment Data")
news_api_key = st.text_input("Enter NewsAPI Key:", type="password")
query = st.text_input("Search Query (e.g., 'Google stock'):", f"{ticker} stock")
if st.button("Fetch Sentiment Data"):
    try:
        news_url = f'https://newsapi.org/v2/everything?q={query}&from={start_date_stock}&to={end_date_stock}&apiKey={news_api_key}'
        news_response = requests.get(news_url)
        news_data = news_response.json()

        if 'articles' not in news_data or not news_data['articles']:
            st.warning("No articles found or invalid API response.")
            news_daily_sentiment = pd.DataFrame({'date': [], 'news_sentiment': []})
        else:
            articles = [{'content': item['content'], 'date': item['publishedAt'][:10]} for item in news_data['articles']]
            news_df = pd.DataFrame(articles)
            news_df['sentiment'] = news_df['content'].apply(
                lambda x: sentiment_analyzer(x)[0]['label'] if isinstance(x, str) else None
            )
            news_df['sentiment_score'] = news_df['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0}).fillna(0)
            news_daily_sentiment = news_df.groupby('date')['sentiment_score'].mean().reset_index(name='news_sentiment')
            news_daily_sentiment['date'] = pd.to_datetime(news_daily_sentiment['date']).dt.date
            st.write("News Sentiment Data Sample:", news_daily_sentiment.head())
    except Exception as e:
        st.error(f"Error fetching or processing sentiment data: {e}")

# Step 3: Combine Stock and Sentiment Data
st.subheader("Step 3: Combine Stock and Sentiment Data")
if st.button("Combine Data"):
    try:
        combined_data = pd.merge(stock_data_simple, news_daily_sentiment, on='date', how='left').fillna(0)
        combined_data['average_sentiment'] = combined_data['news_sentiment']
        st.write("Combined Data Sample:", combined_data.head())
    except Exception as e:
        st.error(f"Error combining data: {e}")

# Step 4: Visualization
st.subheader("Step 4: Visualize Stock Prices and Sentiment")
if st.button("Visualize Data"):
    try:
        scaler = MinMaxScaler()
        combined_data['adjusts_normalized'] = scaler.fit_transform(combined_data[['adjusts']])
        combined_data['average_sentiment_normalized'] = scaler.fit_transform(combined_data[['average_sentiment']])

        plt.figure(figsize=(12, 6))
        plt.plot(combined_data['date'], combined_data['adjusts_normalized'], label='Normalized Stock Price')
        plt.plot(combined_data['date'], combined_data['average_sentiment_normalized'], label='Normalized Sentiment')
        plt.legend()
        plt.title('Normalized Stock Price vs Sentiment')
        plt.xlabel('Date')
        plt.ylabel('Normalized Values')
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error visualizing data: {e}")

# Predictive Modeling
st.subheader("Step 5: Predictive Modeling (Random Forest)")
if st.button("Run Predictive Model"):
    try:
        combined_data.dropna(inplace=True)
        features = ['adjusts', 'average_sentiment']
        X = combined_data[features]
        y = (combined_data['adjusts'].shift(-1) > combined_data['adjusts']).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Model Accuracy: {accuracy:.2f}")
    except Exception as e:
        st.error(f"Error in predictive modeling: {e}")