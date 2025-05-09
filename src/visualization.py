"""
Visualization module for plotting price, momentum, sentiment, and predictions.
"""

import logging
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

def plot_predictions(
    y_test: pd.Series,
    preds: pd.Series,
    ticker: str,
    output: Path | None = None,
) -> None:
    """Plot actual vs predicted prices."""
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    LOGGER.info("Prediction MAE: %.4f | R²: %.4f", mae, r2)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test.index, y_test, label="Actual", linewidth=2)
    ax.plot(preds.index, preds, label="Predicted", linestyle="--")
    ax.set_title(f"{ticker} Next‑Day Close Prediction (Test set)")
    ax.legend()
    fig.tight_layout()

    if output:
        pred_path = output.with_suffix("_pred.png")
        fig.savefig(pred_path, dpi=144)
        LOGGER.info("Prediction plot saved to %s", pred_path.resolve())
    else:
        plt.show()
    plt.close(fig)

def plot_summary(
    data: pd.DataFrame,
    momentum: pd.Series,
    sentiment: pd.Series | None,
    ticker: str,
    window: int,
    output: Path | None = None,
) -> None:
    """Plot summary of price, momentum, and sentiment."""
    LOGGER.info("Generating summary plot…")
    nrows = 3 if sentiment is not None and not sentiment.empty else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 4 * nrows), sharex=True)

    ax_price = axes[0]
    ax_price.plot(data.index, data["Adj Close"], label="Adj Close")
    ax_price.set_title(f"{ticker} Adjusted Close Price")
    ax_price.legend(loc="upper left")

    ax_mom = axes[1]
    ax_mom.plot(momentum.index, momentum, label=f"Momentum ({window}d)", color="tab:orange")
    ax_mom.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax_mom.set_title(f"{ticker} Momentum ({window}-day)")
    ax_mom.legend(loc="upper left")

    if nrows == 3:
        ax_sent = axes[2]
        ax_sent.bar(sentiment.index, sentiment, label="Daily Sentiment", width=0.8)
        ax_sent.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax_sent.set_title(f"{ticker} Daily Sentiment")
        ax_sent.legend(loc="upper left")

    fig.tight_layout()
    
    if output:
        fig.savefig(output, dpi=144)
        LOGGER.info("Summary plot saved to %s", output.resolve())
    else:
        plt.show()
    plt.close(fig) 