"""Shared fixtures and configuration for the v3.0 test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

# ── Reproducibility ───────────────────────────────────────────────────
SEED = 42
FORECAST_STEPS = 5
SEQ_LENGTH = 10
INPUT_SIZE = 8
BATCH_SIZE = 4
QUANTILES = (0.1, 0.5, 0.9)
NUM_QUANTILES = len(QUANTILES)


@pytest.fixture(autouse=True)
def _set_seed():
    """Set seeds for reproducibility in every test."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)


@pytest.fixture
def synthetic_X():
    """Synthetic input array (N, seq_len, features)."""
    N = 50
    return np.random.randn(N, SEQ_LENGTH, INPUT_SIZE).astype(np.float32)


@pytest.fixture
def synthetic_y():
    """Synthetic multi-step return targets (N, K)."""
    N = 50
    return np.random.randn(N, FORECAST_STEPS).astype(np.float32) * 0.01


@pytest.fixture
def synthetic_ohlcv():
    """Synthetic OHLCV DataFrame resembling market data."""
    N = 300
    dates = pd.bdate_range("2020-01-02", periods=N, freq="B")
    close = 150 + np.cumsum(np.random.randn(N) * 0.5)
    high = close + np.abs(np.random.randn(N)) * 0.5
    low = close - np.abs(np.random.randn(N)) * 0.5
    open_ = close + np.random.randn(N) * 0.3
    volume = np.random.randint(100_000, 1_000_000, N).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df
