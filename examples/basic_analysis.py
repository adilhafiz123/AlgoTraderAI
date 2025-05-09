"""
Basic example of using AlgoTraderAI.
"""

from pathlib import Path
from src.data import fetch_data, calculate_momentum
from src.utils import fetch_headlines, analyse_sentiment, aggregate_sentiment
from src.visualization import plot_summary

def main():
    """Run a basic analysis example."""
    # Parameters
    ticker = "AAPL"
    start = "2024-01-01"
    end = "2024-03-01"
    window = 10
    news_limit = 50
    output = Path("apple_analysis.png")

    # Fetch and process price data
    data_df = fetch_data(ticker, start, end)
    momentum = calculate_momentum(data_df["Adj Close"], window)
    data_df["Momentum"] = momentum

    # Fetch and analyze news sentiment
    headlines = fetch_headlines(ticker, news_limit)
    sentiment_df = analyse_sentiment(headlines, "finiteautomata/bertweet-base-sentiment-analysis")
    daily_sentiment = aggregate_sentiment(sentiment_df)

    # Generate summary plot
    plot_summary(
        data_df,
        momentum,
        daily_sentiment,
        ticker,
        window,
        output,
    )

if __name__ == "__main__":
    main() 