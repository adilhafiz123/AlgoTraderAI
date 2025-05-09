<p align="center">
  <img src="A_digital_graphic_features_the_title_\"AlgoTraderAI.png\"" alt="AlgoTraderAI" width="600"/>
</p>

# 🚀 AlgoTraderAI

An algorithmic trading system that combines historical price data 📈, momentum indicators 🌀, news sentiment analysis 📰, and machine learning predictions 🤖.

---

## 🌟 Features

- 🕰️ Historical price data fetching and momentum calculation  
- 🧠 News headline sentiment analysis using BERT-style models  
- 📊 Next-day price prediction using XGBoost  
- 📉 Interactive and saved plots for analysis  

---

## ⚙️ Installation

1. 📥 Clone the repository:
```bash
git clone https://github.com/yourusername/AlgoTraderAI.git
cd AlgoTraderAI
```

2. 📦 Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🖥️ Command Line Interface

Basic usage:
```bash
python -m src.main --ticker TSLA \
                  --start 2024-01-01 \
                  --end 2024-05-01 \
                  --window 10 \
                  --news-limit 100 \
                  --predict \
                  --lags 7 \
                  --output tesla.png
```

### ⚙️ Arguments

- `--ticker` 🏷️: Stock ticker symbol (required)  
- `--start` 📅: Start date in YYYY-MM-DD format (required)  
- `--end` 📅: End date in YYYY-MM-DD format (required)  
- `--window` 🔁: Momentum window size (default: 10)  
- `--news-limit` 📰: Max number of news headlines to analyse (default: 100)  
- `--predict` 🧠: Enable price prediction (optional)  
- `--lags` ⏳: Number of lag features for prediction (default: 7)  
- `--output` 🖼️: Output file path for plots (optional)  

---

### 🐍 Python API

You can also use AlgoTraderAI as a Python package:

```python
from src.data import fetch_data, calculate_momentum
from src.utils import fetch_headlines, analyse_sentiment
from src.visualization import plot_summary

# Fetch and process data
data_df = fetch_data("AAPL", "2024-01-01", "2024-03-01")
momentum = calculate_momentum(data_df["Adj Close"], window=10)

# Analyze sentiment
headlines = fetch_headlines("AAPL", limit=50)
sentiment_df = analyse_sentiment(headlines, "finiteautomata/bertweet-base-sentiment-analysis")

# Plot results
plot_summary(data_df, momentum, sentiment_df, "AAPL", window=10)
```

🔎 See the `examples/` directory for more usage examples.

---

## 🗂️ Project Structure

```
AlgoTraderAI/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── price_data.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── prediction.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── sentiment.py
│   └── visualization/
│       ├── __init__.py
│       └── plots.py
├── tests/
│   └── test_data.py
├── examples/
│   └── basic_analysis.py
├── docs/
├── requirements.txt
└── README.md
```

---

## 🛠️ Development

### ✅ Running Tests

```bash
pytest tests/
```

### 🧩 Adding New Features

1. Create appropriate module in `src/`  
2. Add tests in `tests/`  
3. Update documentation  
4. Add example usage in `examples/`  

---

## 📦 Dependencies

- `yfinance` 📈: Yahoo Finance data fetching  
- `pandas` 🐼: Data manipulation  
- `matplotlib` 📊: Plotting  
- `transformers` 🤖: Sentiment analysis  
- `torch` 🔥: Deep learning backend  
- `scikit-learn` 📘: Machine learning utilities  
- `xgboost` 🚀: Price prediction model  
