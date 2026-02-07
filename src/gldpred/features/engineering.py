"""
Feature engineering module for creating technical indicators and features.
"""
import pandas as pd
import numpy as np


class FeatureEngineering:
    """Create features from raw price data."""

    @staticmethod
    def add_technical_indicators(data):
        """
        Add technical indicators to the dataframe.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with additional feature columns
        """
        df = data.copy()

        # Returns
        df['returns'] = df['Close'].pct_change()

        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()

        # Volatility
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()

        # Momentum indicators
        df['momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)

        # RSI (Relative Strength Index)
        df['rsi_14'] = FeatureEngineering._compute_rsi(df['Close'], 14)

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Price position relative to moving averages
        df['price_to_sma_20'] = df['Close'] / df['sma_20']
        df['price_to_sma_50'] = df['Close'] / df['sma_50']

        # Volume indicators
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

        return df

    @staticmethod
    def _compute_rsi(series, period=14):
        """Compute RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def select_features(data, feature_columns=None):
        """
        Select specific feature columns.

        Args:
            data: DataFrame with all features
            feature_columns: List of column names to select (None = auto-select)

        Returns:
            DataFrame with selected features
        """
        if feature_columns is None:
            # Auto-select numeric columns, excluding target variables
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            feature_columns = [col for col in data.columns
                               if col not in exclude_cols and data[col].dtype in [np.float64, np.int64]]

        return data[feature_columns]

    @staticmethod
    def create_sequences(features, targets, seq_length=20):
        """
        Create sequences for time series prediction.

        Args:
            features: Feature DataFrame
            targets: Target Series or DataFrame
            seq_length: Length of input sequences

        Returns:
            Tuple of (X, y) as numpy arrays
        """
        X, y = [], []

        features_array = features.values
        targets_array = targets.values if hasattr(targets, 'values') else targets

        for i in range(len(features_array) - seq_length):
            X.append(features_array[i:i + seq_length])
            y.append(targets_array[i + seq_length])

        return np.array(X), np.array(y)
