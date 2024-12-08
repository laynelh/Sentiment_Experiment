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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

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

# Initialize session state for stock and sentiment data
if "stock_data_simple" not in st.session_state:
    st.session_state.stock_data_simple = pd.DataFrame(columns=['date', 'adjusts', 'Volume'])
if "stock_data_fetched" not in st.session_state:
    st.session_state.stock_data_fetched = False
if "news_daily_sentiment" not in st.session_state:
    st.session_state.news_daily_sentiment = pd.DataFrame(columns=['date', 'news_sentiment'])
if "news_data_fetched" not in st.session_state:
    st.session_state.news_data_fetched = False

# Fetch Both Stock and Sentiment Data
st.subheader("Step 1: Fetch Stock and Sentiment Data")
ticker = st.text_input("Enter Stock Ticker:", "GOOG")
news_api_key = st.text_input("Enter NewsAPI Key:", type="password")

start_date = st.date_input("Start Date:", pd.Timestamp("2023-10-22"))
end_date = st.date_input("End Date:", pd.Timestamp("2024-12-08"))
start_date_news = st.date_input("News Start Date:", pd.Timestamp("2024-11-10"))
end_date_news = st.date_input("News End Date:", pd.Timestamp("2024-12-08"))

if st.button("Fetch Data"):
    try:
        # Fetch Stock Data
        st.write(f"Fetching stock data for ticker: {ticker}, Start: {start_date}, End: {end_date}")
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            st.warning("No stock data fetched. Please check the ticker symbol and date range.")
        else:
            stock_data = stock_data.reset_index()
            stock_data['date'] = stock_data['Date'].dt.date
            st.session_state.stock_data_simple = stock_data[['date', 'Adj Close', 'Volume']].rename(columns={'Adj Close': 'adjusts'})
            st.session_state.stock_data_fetched = True
            st.write("Stock Data Sample:", st.session_state.stock_data_simple.head())

        # Fetch Sentiment Data
        st.write(f"Fetching news data for query: '{ticker} stock'")
        news_url = f'https://newsapi.org/v2/everything?q={ticker} stock&from={start_date_news}&to={end_date_news}&apiKey={news_api_key}'
        news_response = requests.get(news_url)
        news_data = news_response.json()

        if 'articles' not in news_data or not news_data['articles']:
            st.warning("No articles found or invalid API response.")
            st.session_state.news_daily_sentiment = pd.DataFrame({'date': [], 'news_sentiment': []})
        else:
            articles = [{'content': item['content'], 'date': item['publishedAt'][:10]} for item in news_data['articles']]
            news_df = pd.DataFrame(articles)
            news_df['sentiment'] = news_df['content'].apply(
                lambda x: sentiment_analyzer(x)[0]['label'] if isinstance(x, str) else None
            )
            news_df['sentiment_score'] = news_df['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0}).fillna(0)
            st.session_state.news_daily_sentiment = news_df.groupby('date')['sentiment_score'].mean().reset_index(name='news_sentiment')
            st.session_state.news_data_fetched = True
            st.write("News Sentiment Data Sample:", st.session_state.news_daily_sentiment.head())

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        
# Combine Stock and Sentiment Data
# Combine Stock and Sentiment Data
st.subheader("Step 3: Combine Stock and Sentiment Data")
if st.button("Combine Data"):
    try:
        if not st.session_state.stock_data_fetched:
            st.warning("Stock data is missing. Please fetch stock data before combining.")
        elif not st.session_state.news_data_fetched:
            st.warning("Sentiment data is missing. Please fetch sentiment data before combining.")
        else:
            # Reset index and ensure single-level columns
            stock_data_simple = st.session_state.stock_data_simple.reset_index(drop=True)
            news_daily_sentiment = st.session_state.news_daily_sentiment.reset_index(drop=True)

            if isinstance(stock_data_simple.columns, pd.MultiIndex):
                stock_data_simple.columns = stock_data_simple.columns.get_level_values(0)

            # Ensure 'date' column is compatible
            stock_data_simple['date'] = pd.to_datetime(stock_data_simple['date']).dt.date
            news_daily_sentiment['date'] = pd.to_datetime(news_daily_sentiment['date']).dt.date

            # Merge the data
            combined_data = pd.merge(stock_data_simple, news_daily_sentiment, on='date', how='left').fillna(0)
            combined_data['average_sentiment'] = combined_data['news_sentiment']

            # Store combined_data in session state
            st.session_state.combined_data = combined_data
            st.write("Combined Data Sample:", combined_data.head())
    except Exception as e:
        st.error(f"Error combining data: {e}")
        
# Visualize Data
# Visualize Data
st.subheader("Step 4: Visualize Stock Prices and Sentiment")
if st.button("Visualize Data"):
    try:
        # Show date inputs for News Date Range with unique keys
        st.markdown("### Select News Date Range for Visualization")
        start_date_news = st.date_input("News Start Date:", pd.Timestamp("2024-11-10"), key="visual_start_date")
        end_date_news = st.date_input("News End Date:", pd.Timestamp("2024-12-08"), key="visual_end_date")

        if "combined_data" not in st.session_state or st.session_state.combined_data.empty:
            st.warning("Combined data is not available. Please combine the data first.")
        else:
            # Filter combined data based on selected date range
            combined_data = st.session_state.combined_data
            filtered_data = combined_data[
                (combined_data['date'] >= start_date_news) & (combined_data['date'] <= end_date_news)
            ]

            if filtered_data.empty:
                st.warning("No data available for the selected date range.")
            else:
                # Normalize data for visualization
                scaler = MinMaxScaler()
                filtered_data['adjusts_normalized'] = scaler.fit_transform(filtered_data[['adjusts']])
                filtered_data['average_sentiment_normalized'] = scaler.fit_transform(filtered_data[['average_sentiment']])

                # Plot data
                plt.figure(figsize=(12, 6))
                plt.plot(filtered_data['date'], filtered_data['adjusts_normalized'], label='Normalized Stock Price')
                plt.plot(filtered_data['date'], filtered_data['average_sentiment_normalized'], label='Normalized Sentiment')
                plt.legend()
                plt.title('Normalized Stock Price vs Sentiment')
                plt.xlabel('Date')
                plt.ylabel('Normalized Values')
                plt.xticks(rotation=45)
                plt.grid(True)
                st.pyplot(plt)
    except Exception as e:
        st.error(f"Error visualizing data: {e}")


# Predictive Modeling with Multiple Runs
st.subheader("Step 5: Predictive Modeling (Random Forest) with Repeated Evaluations")

# Slider to choose the number of evaluations
num_evaluations = st.slider("Number of Model Evaluations:", min_value=1, max_value=10, value=1, step=1)

if st.button("Run Predictive Model Multiple Times"):
    try:
        if "combined_data" not in st.session_state or st.session_state.combined_data.empty:
            st.warning("Combined data is not available. Please combine the data first.")
        else:
            combined_data = st.session_state.combined_data
            combined_data.dropna(inplace=True)

            # Prepare data for modeling
            features = ['adjusts', 'average_sentiment']
            X = combined_data[features]
            y = (combined_data['adjusts'].shift(-1) > combined_data['adjusts']).astype(int)

            # Store results across evaluations
            accuracy_list = []
            confusion_matrices = []

            for i in range(num_evaluations):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

                # Train Random Forest model
                model = RandomForestClassifier(n_estimators=100, random_state=i)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, predictions)
                accuracy_list.append(accuracy)

                # Confusion Matrix
                cm = confusion_matrix(y_test, predictions)
                confusion_matrices.append(cm)

            # Display Results
            st.write(f"Results of {num_evaluations} Evaluations:")
            st.write(f"Accuracy Scores: {accuracy_list}")
            st.write(f"Average Accuracy: {np.mean(accuracy_list):.2f}")

            # Display the last confusion matrix as an example
            st.write("Example Confusion Matrix (from last evaluation):")
            sns.heatmap(confusion_matrices[-1], annot=True, fmt="d", cmap="Blues", 
                        xticklabels=['Decrease', 'Increase'], yticklabels=['Decrease', 'Increase'])
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Error in predictive modeling: {e}")