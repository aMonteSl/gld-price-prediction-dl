"""Shared fixtures for all tests."""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure src/ is importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_SAMPLES = 120
SEQ_LEN = 20
N_FEATURES = 10
HIDDEN = 32
LAYERS = 2
BATCH = 16


@pytest.fixture(autouse=True)
def seed_everything():
    """Reproducible tests — seed Python, NumPy, and PyTorch."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_X() -> np.ndarray:
    """Random input sequences (N, seq_len, features)."""
    return np.random.randn(N_SAMPLES, SEQ_LEN, N_FEATURES).astype(np.float32)


@pytest.fixture
def synthetic_y_reg() -> np.ndarray:
    """Random regression targets (N,)."""
    return np.random.randn(N_SAMPLES).astype(np.float32)


@pytest.fixture
def synthetic_y_cls() -> np.ndarray:
    """Random binary classification targets (N,)."""
    return np.random.randint(0, 2, N_SAMPLES).astype(np.float32)


@pytest.fixture
def synthetic_y_mt(synthetic_y_reg, synthetic_y_cls) -> np.ndarray:
    """Stacked multitask targets (N, 2) — col 0 reg, col 1 cls."""
    return np.column_stack([synthetic_y_reg, synthetic_y_cls])


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with 250 rows (~ 1 year)."""
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.cumsum(np.random.randn(n) * 0.5) + 180
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.2,
            "High": close + np.abs(np.random.randn(n) * 0.5),
            "Low": close - np.abs(np.random.randn(n) * 0.5),
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
            "Dividends": np.zeros(n),
            "Stock Splits": np.zeros(n),
        },
        index=dates,
    )
