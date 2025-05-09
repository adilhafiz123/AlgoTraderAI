"""
Data fetching and processing module for price and momentum calculations.
"""

import logging
from typing import Final

import pandas as pd
import yfinance as yf

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical price data for a given ticker."""
    LOGGER.info("Fetching price data for %s (%s → %s)…", ticker, start, end)
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        LOGGER.warning("No data returned for %s – check symbol or dates.", ticker)
    return data

def calculate_momentum(close: pd.Series, window: int) -> pd.Series:
    """Calculate momentum over a given window."""
    if window <= 0:
        raise ValueError("window must be > 0")
    LOGGER.debug("Calculating %d‑day momentum…", window)
    return close.diff(window) 