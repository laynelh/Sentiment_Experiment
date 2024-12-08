# Sentiment Experiment

The **Sentiment Experiment** is a Streamlit-based web application designed to analyze sentiment using FinBERT. The following demonstrates how to set up and deploy a simple sentiment analysis tool locally.

---

## Requirements
- Python 3.10
- Streamlit
- TensorFlow (or compatible backend)

---

## Installation Instructions

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/laynelh/Sentiment_Experiment.git
cd Sentiment_Experiment
```

### 2. Create a Virtual Environment
Create a virtual environment to isolate dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
Install the required Python packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Running the Application

### Start the Streamlit App
Run the following command to start the Streamlit app:
```bash
streamlit run streamlit_app.py
```
### Access the App
Once the app starts, it will provide a local URL (e.g., `http://localhost:8501`) to access the app in your browser.

---

## Testing the FinBERT Model

A test file (`test.py`) is included to verify the FinBERT model functionality. Run the following command:
```bash
streamlit run test.py
```
