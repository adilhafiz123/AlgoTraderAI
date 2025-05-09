"""
Price prediction module using XGBoost.
"""

import logging
from typing import Final, Tuple

import pandas as pd

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

def _import_xgb() -> "XGBRegressor":  # type: ignore[name-defined]
    """Import XGBRegressor with proper error handling."""
    try:
        from xgboost import XGBRegressor  # noqa: WPS433
    except ModuleNotFoundError as exc:  # pragma: no cover
        LOGGER.error("xgboost not installed – run: pip install xgboost scikit-learn")
        raise SystemExit(exc) from exc
    return XGBRegressor

def make_feature_matrix(
    data: pd.DataFrame,
    sentiment: pd.Series | None,
    lags: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create lag features + optional sentiment to predict next‑day Adj Close."""
    df = data[["Adj Close", "Momentum"]].copy()
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["Adj Close"].shift(i)
    if sentiment is not None and not sentiment.empty:
        df = df.join(sentiment.rename("sentiment"))
    df.dropna(inplace=True)
    feature_cols = [c for c in df.columns if c.startswith("lag_") or c in ("Momentum", "sentiment")]
    X = df[feature_cols]
    y = df["Adj Close"].shift(-1).loc[X.index]
    return X, y

def train_test_split_ts(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Split data into train and test sets maintaining time series order."""
    split_idx = int(len(X) * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

def train_predict_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.Series:
    """Train XGBoost model and make predictions."""
    XGBRegressor = _import_xgb()
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return pd.Series(model.predict(X_test), index=X_test.index, name="Predicted") 