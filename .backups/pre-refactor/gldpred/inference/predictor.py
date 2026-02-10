"""Trajectory prediction with uncertainty bands and price reconstruction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class TrajectoryForecast:
    """Structured output of a trajectory prediction."""

    dates: List                     # future dates or day-offsets
    returns_quantiles: np.ndarray   # (K, Q) predicted daily returns
    price_paths: np.ndarray         # (K+1, Q) implied prices
    quantiles: List[float]
    last_price: float
    last_date: object               # datetime-like


class TrajectoryPredictor:
    """Generate trajectory forecasts with uncertainty bands."""

    def __init__(self, trainer) -> None:
        self.trainer = trainer

    # ------------------------------------------------------------------
    # Single-sequence trajectory
    # ------------------------------------------------------------------
    def predict_trajectory(
        self,
        df_or_X,
        feature_names_or_price=None,
        seq_length_or_date=None,
        asset: str = "GLD",
        *,
        is_crypto: bool | None = None,
    ) -> TrajectoryForecast:
        """Predict future price trajectory.

        Can be called in two ways:

        1. ``predict_trajectory(X, last_price, last_date)``
           — X is a numpy array (N, seq, F).
        2. ``predict_trajectory(df, feature_names, seq_length, asset)``
           — df is a pandas DataFrame with OHLCV + features.
        """
        if isinstance(df_or_X, pd.DataFrame):
            df = df_or_X
            feature_names = feature_names_or_price
            seq_length = seq_length_or_date
            last_price = float(df["Close"].iloc[-1])
            last_date = df.index[-1]
            X_seq = df[feature_names].values[-seq_length:]
            X_seq = X_seq.reshape(1, seq_length, -1).astype(np.float32)
            if is_crypto is None:
                is_crypto = asset in ("BTC-USD",)
        else:
            X_seq = df_or_X
            last_price = float(feature_names_or_price)
            last_date = seq_length_or_date
            if is_crypto is None:
                is_crypto = False

        if X_seq.ndim == 3 and X_seq.shape[0] > 1:
            X_seq = X_seq[-1:]

        pred = self.trainer.predict(X_seq)   # (1, K, Q)
        returns_q = pred[0]               # (K, Q)
        K, Q = returns_q.shape
        quantiles = list(self.trainer.quantiles_tuple)

        # Future dates
        start = pd.Timestamp(last_date) + pd.Timedelta(days=1)
        if is_crypto:
            future_dates = pd.date_range(start=start, periods=K).tolist()
        else:
            future_dates = pd.bdate_range(start=start, periods=K).tolist()

        # Price path reconstruction
        price_paths = np.zeros((K + 1, Q))
        price_paths[0, :] = last_price
        for k in range(K):
            price_paths[k + 1, :] = price_paths[k, :] * (1 + returns_q[k, :])

        return TrajectoryForecast(
            dates=future_dates,
            returns_quantiles=returns_q,
            price_paths=price_paths,
            quantiles=quantiles,
            last_price=last_price,
            last_date=last_date,
        )

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------
    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """Predict on all samples — returns (N, K, Q)."""
        return self.trainer.predict(X)
