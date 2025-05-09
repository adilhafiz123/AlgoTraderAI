"""
News and sentiment analysis utilities.
"""

import logging
from datetime import datetime
from importlib import import_module
from typing import Final, List

import pandas as pd
import yfinance as yf

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

def fetch_headlines(ticker: str, limit: int = 50) -> List[dict]:
    """Fetch news headlines for a given ticker."""
    LOGGER.info("Fetching up to %d news headlines for %s…", limit, ticker)
    try:
        news_items = yf.Ticker(ticker).news
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Unable to fetch news: %s", exc)
        news_items = []
    return news_items[:limit]

def _get_sentiment_pipeline(model_name: str):
    """Lazy‑load HuggingFace to avoid startup cost when sentiment is skipped."""
    transformers = import_module("transformers")
    LOGGER.debug("Loading model %s…", model_name)
    return transformers.pipeline(
        task="sentiment-analysis",
        model=model_name,
        truncation=True,
        device=-1,
    )

def analyse_sentiment(headlines: List[dict], model_name: str) -> pd.DataFrame:
    """Analyze sentiment of news headlines."""
    if not headlines:
        LOGGER.warning("No headlines supplied – skipping sentiment analysis.")
        return pd.DataFrame()

    titles = [item["title"] for item in headlines]
    published = [datetime.fromtimestamp(item["providerPublishTime"]) for item in headlines]

    pipe = _get_sentiment_pipeline(model_name)
    LOGGER.info("Running sentiment analysis on %d headlines…", len(titles))
    raw = pipe(titles)

    df = pd.DataFrame(
        {
            "datetime": published,
            "headline": titles,
            "label": [r["label"] for r in raw],
            "score": [r["score"] for r in raw],
        }
    )
    # signed sentiment: positive ⇒ +score, negative ⇒ −score
    df["signed_score"] = df.apply(
        lambda r: r["score"] if r["label"].upper().startswith("POS") else -r["score"],
        axis=1,
    )
    return df

def aggregate_sentiment(df: pd.DataFrame, freq: str = "D") -> pd.Series:
    """Aggregate sentiment scores to daily frequency."""
    if df.empty:
        return pd.Series(dtype=float)
    ser = df.set_index("datetime")["signed_score"].sort_index()
    return ser.resample(freq).mean() 