"""Tests for feature engineering and multi-step sequence creation."""
from __future__ import annotations

import numpy as np
import pytest

from gldpred.features import FeatureEngineering
from conftest import FORECAST_STEPS, SEQ_LENGTH


class TestTechnicalIndicators:
    def test_adds_columns(self, synthetic_ohlcv):
        eng = FeatureEngineering()
        df = eng.add_technical_indicators(synthetic_ohlcv)
        for col in ["sma_5", "sma_20", "sma_50", "sma_200", "rsi_14", "macd", "atr_pct"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_duplicate_columns(self, synthetic_ohlcv):
        eng = FeatureEngineering()
        df = eng.add_technical_indicators(synthetic_ohlcv)
        assert len(df.columns) == len(set(df.columns))

    def test_select_features_excludes_raw(self, synthetic_ohlcv):
        eng = FeatureEngineering()
        df = eng.add_technical_indicators(synthetic_ohlcv)
        selected = eng.select_features(df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col not in selected

    def test_select_features_non_empty(self, synthetic_ohlcv):
        eng = FeatureEngineering()
        df = eng.add_technical_indicators(synthetic_ohlcv)
        selected = eng.select_features(df)
        assert len(selected) > 10


class TestSequenceCreation:
    def test_output_shape(self, synthetic_ohlcv):
        eng = FeatureEngineering()
        df = eng.add_technical_indicators(synthetic_ohlcv)
        feat_df = eng.select_features(df)
        daily_ret = df["Close"].pct_change().fillna(0).values

        X, y = eng.create_sequences(
            feat_df.values,
            daily_ret,
            seq_length=SEQ_LENGTH,
            forecast_steps=FORECAST_STEPS,
        )
        assert X.ndim == 3
        assert X.shape[1] == SEQ_LENGTH
        assert X.shape[2] == feat_df.shape[1]
        assert y.ndim == 2
        assert y.shape[1] == FORECAST_STEPS
        assert X.shape[0] == y.shape[0]

    def test_different_forecast_steps(self, synthetic_ohlcv):
        eng = FeatureEngineering()
        df = eng.add_technical_indicators(synthetic_ohlcv)
        feat_df = eng.select_features(df)
        daily_ret = df["Close"].pct_change().fillna(0).values

        for k in [1, 3, 10]:
            X, y = eng.create_sequences(
                feat_df.values,
                daily_ret,
                seq_length=SEQ_LENGTH,
                forecast_steps=k,
            )
            assert y.shape[1] == k

    def test_no_nan_in_output(self, synthetic_ohlcv):
        eng = FeatureEngineering()
        df = eng.add_technical_indicators(synthetic_ohlcv)
        feat_df = eng.select_features(df)
        daily_ret = df["Close"].pct_change().fillna(0).values

        # drop NaN rows from features first
        feat_clean = feat_df.dropna()
        ret_clean = daily_ret[len(daily_ret) - len(feat_clean):]

        X, y = eng.create_sequences(
            feat_clean.values,
            ret_clean,
            seq_length=SEQ_LENGTH,
            forecast_steps=FORECAST_STEPS,
        )
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
