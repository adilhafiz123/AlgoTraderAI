"""
Main script for AlgoTraderAI.
"""

import argparse
import logging
from pathlib import Path
from typing import Final

from .data import fetch_data, calculate_momentum
from .models import make_feature_matrix, train_test_split_ts, train_predict_model
from .utils import fetch_headlines, analyse_sentiment, aggregate_sentiment
from .visualization import plot_predictions, plot_summary

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AlgoTraderAI - Algorithmic Trading System")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--window", type=int, default=10, help="Momentum window size")
    parser.add_argument("--news-limit", type=int, default=100, help="Maximum number of news headlines")
    parser.add_argument("--predict", action="store_true", help="Enable price prediction")
    parser.add_argument("--lags", type=int, default=7, help="Number of lag features for prediction")
    parser.add_argument("--output", type=Path, help="Output file path for plots")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Fetch and process price data
    data_df = fetch_data(args.ticker, args.start, args.end)
    momentum = calculate_momentum(data_df["Adj Close"], args.window)
    data_df["Momentum"] = momentum

    # Fetch and analyze news sentiment
    headlines = fetch_headlines(args.ticker, args.news_limit)
    sentiment_df = analyse_sentiment(headlines, "finiteautomata/bertweet-base-sentiment-analysis")
    daily_sentiment = aggregate_sentiment(sentiment_df)

    # Generate summary plot
    plot_summary(
        data_df,
        momentum,
        daily_sentiment,
        args.ticker,
        args.window,
        args.output,
    )

    # Optional: Price prediction
    if args.predict:
        X, y = make_feature_matrix(data_df, daily_sentiment, args.lags)
        X_train, X_test, y_train, y_test = train_test_split_ts(X, y)
        predictions = train_predict_model(X_train, y_train, X_test)
        plot_predictions(y_test, predictions, args.ticker, args.output)

if __name__ == "__main__":
    main() 