"""Tests for FeatureEngineering and data-related utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gldpred.features import FeatureEngineering


class TestFeatureEngineering:
    def test_add_technical_indicators(self, synthetic_ohlcv):
        fe = FeatureEngineering()
        result = fe.add_technical_indicators(synthetic_ohlcv)
        # Should add features beyond the original 7 columns
        assert result.shape[1] > synthetic_ohlcv.shape[1]
        assert len(result) == len(synthetic_ohlcv)

    def test_select_features_returns_numeric(self, synthetic_ohlcv):
        fe = FeatureEngineering()
        data = fe.add_technical_indicators(synthetic_ohlcv)
        features = fe.select_features(data)
        assert all(np.issubdtype(features[c].dtype, np.number) for c in features.columns)

    def test_create_sequences_shapes(self, synthetic_ohlcv):
        fe = FeatureEngineering()
        data = fe.add_technical_indicators(synthetic_ohlcv)
        features = fe.select_features(data).ffill().bfill()
        targets = pd.Series(np.random.randn(len(features)), index=features.index)
        seq_len = 20
        X, y = fe.create_sequences(features, targets, seq_length=seq_len)
        assert X.ndim == 3
        assert X.shape[1] == seq_len
        assert X.shape[2] == features.shape[1]
        assert y.ndim == 1
        assert len(y) == len(X)

    def test_create_sequences_multitask(self, synthetic_ohlcv):
        """Targets with 2 columns (multitask) should be preserved."""
        fe = FeatureEngineering()
        data = fe.add_technical_indicators(synthetic_ohlcv)
        features = fe.select_features(data).ffill().bfill()
        n = len(features)
        targets = np.column_stack([
            np.random.randn(n),
            np.random.randint(0, 2, n),
        ])
        X, y = fe.create_sequences(features, targets, seq_length=20)
        assert y.ndim == 2
        assert y.shape[1] == 2
