"""Feature engineering: technical indicators and multi-step sequence creation."""
from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineering:
    """Compute technical features and create training sequences."""

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    @staticmethod
    def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add 30+ technical indicators to an OHLCV DataFrame."""
        df = data.copy()

        # Daily returns
        df["returns"] = df["Close"].pct_change()

        # Moving averages (simple + exponential)
        for w in [5, 10, 20, 50, 200]:
            df[f"sma_{w}"] = df["Close"].rolling(w).mean()
            df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

        # Volatility
        df["volatility_5"] = df["returns"].rolling(5).std()
        df["volatility_20"] = df["returns"].rolling(20).std()

        # Momentum
        df["momentum_5"] = df["Close"] - df["Close"].shift(5)
        df["momentum_10"] = df["Close"] - df["Close"].shift(10)

        # RSI-14
        df["rsi_14"] = FeatureEngineering._compute_rsi(df["Close"], 14)

        # MACD
        exp12 = df["Close"].ewm(span=12, adjust=False).mean()
        exp26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp12 - exp26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # Price relative to SMAs
        df["price_to_sma_20"] = df["Close"] / df["sma_20"]
        df["price_to_sma_50"] = df["Close"] / df["sma_50"]
        df["price_to_sma_200"] = df["Close"] / df["sma_200"]

        # Volume
        df["volume_sma_20"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma_20"]

        # ATR (Average True Range)
        prev_close = df["Close"].shift(1)
        tr = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr_14"] / df["Close"]

        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f"close_lag_{lag}"] = df["Close"].shift(lag)
            df[f"returns_lag_{lag}"] = df["returns"].shift(lag)

        return df

    # ------------------------------------------------------------------
    # RSI helper
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    @staticmethod
    def select_features(
        data: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Select numeric feature columns, excluding raw OHLCV."""
        if feature_columns is None:
            exclude = {
                "Open", "High", "Low", "Close", "Volume",
                "Dividends", "Stock Splits",
            }
            feature_columns = [
                c for c in data.columns
                if c not in exclude
                and data[c].dtype in (np.float64, np.int64, np.float32)
            ]
        return data[feature_columns]

    # ------------------------------------------------------------------
    # Sequence creation (multi-step)
    # ------------------------------------------------------------------
    @staticmethod
    def create_sequences(
        features: pd.DataFrame | np.ndarray,
        daily_returns: pd.Series | np.ndarray,
        seq_length: int = 20,
        forecast_steps: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create input sequences and multi-step return targets.

        Args:
            features: (T, F) feature matrix.
            daily_returns: (T,) daily percentage returns.
            seq_length: lookback window (number of historical days).
            forecast_steps: K — number of future days to predict.

        Returns:
            X: (N, seq_length, F)
            y: (N, K)
        """
        feat = (
            features.values if hasattr(features, "values")
            else np.asarray(features)
        )
        rets = (
            daily_returns.values if hasattr(daily_returns, "values")
            else np.asarray(daily_returns)
        )

        X: list[np.ndarray] = []
        y: list[np.ndarray] = []
        for i in range(len(feat) - seq_length - forecast_steps + 1):
            X.append(feat[i : i + seq_length])
            y.append(rets[i + seq_length : i + seq_length + forecast_steps])

        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
        )
