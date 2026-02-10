"""Tests for BacktestEngine, BacktestResult, and BacktestSummary.

Uses a mock model bundle (duck-typed) that returns predictable quantile
outputs so we can verify the engine's computation logic deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from gldpred.services.backtest_engine import (
    BacktestEngine,
    BacktestResult,
    BacktestSummary,
    _cumulative_return,
    _nearest_idx,
)


# ── Mock model bundle ───────────────────────────────────────────────────

class _MockBundle:
    """Minimal duck-typed replacement for ``ModelBundle``.

    ``predict(X)`` returns a constant per-step return for each quantile
    so that the engine's cumulative return and comparison logic can be
    tested deterministically.
    """

    def __init__(
        self,
        per_step_returns: Tuple[float, float, float] = (-0.005, 0.01, 0.02),
        forecast_steps: int = 5,
    ) -> None:
        # per_step_returns: (P10, P50, P90) daily return per step
        self._returns = per_step_returns
        self._K = forecast_steps
        self.quantiles_tuple = (0.1, 0.5, 0.9)
        self.config = {"seq_length": 5, "forecast_steps": forecast_steps}
        self.label = "mock_model"
        self.asset = "GLD"

    @property
    def quantiles(self) -> Tuple[float, ...]:
        return self.quantiles_tuple

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return (N, K, Q) with constant per-step returns."""
        N = X.shape[0]
        K = self._K
        Q = len(self._returns)
        out = np.zeros((N, K, Q), dtype=np.float32)
        for q_idx, r in enumerate(self._returns):
            out[:, :, q_idx] = r
        return out


# ── Helper fixtures ──────────────────────────────────────────────────────

def _make_df(n_rows: int = 100, start_price: float = 100.0) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame with deterministic Close prices."""
    dates = pd.bdate_range("2024-01-02", periods=n_rows)
    np.random.seed(42)
    # Small random walk so Close prices are realistic
    returns = np.random.normal(0.001, 0.01, n_rows)
    prices = start_price * np.cumprod(1 + returns)
    df = pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.005,
        "Low": prices * 0.995,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 5_000_000, n_rows),
    }, index=dates)
    # Add dummy features
    df["f1"] = np.random.randn(n_rows)
    df["f2"] = np.random.randn(n_rows)
    return df


# ══════════════════════════════════════════════════════════════════════════
# Helper tests
# ══════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_cumulative_return_zero(self):
        assert _cumulative_return(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_cumulative_return_positive(self):
        # (1.01)^3 - 1 ≈ 0.030301
        r = _cumulative_return(np.array([0.01, 0.01, 0.01]))
        assert r == pytest.approx(0.030301, abs=1e-6)

    def test_cumulative_return_negative(self):
        r = _cumulative_return(np.array([-0.01, -0.01]))
        assert r < 0

    def test_nearest_idx(self):
        q = [0.1, 0.5, 0.9]
        assert _nearest_idx(q, 0.1) == 0
        assert _nearest_idx(q, 0.5) == 1
        assert _nearest_idx(q, 0.9) == 2
        assert _nearest_idx(q, 0.4) == 1  # closest to 0.5


# ══════════════════════════════════════════════════════════════════════════
# run_single
# ══════════════════════════════════════════════════════════════════════════

class TestRunSingle:
    def test_basic(self):
        df = _make_df(50)
        bundle = _MockBundle(forecast_steps=5)
        engine = BacktestEngine()

        result = engine.run_single(
            df=df,
            bundle=bundle,
            feature_names=["f1", "f2"],
            backtest_idx=10,
            seq_length=5,
            forecast_steps=5,
            investment=10_000,
        )
        assert result is not None
        assert isinstance(result, BacktestResult)
        assert result.horizon == 5
        assert result.investment == 10_000

    def test_predicted_return_p50(self):
        df = _make_df(50)
        bundle = _MockBundle(per_step_returns=(-0.01, 0.02, 0.03), forecast_steps=5)
        engine = BacktestEngine()

        result = engine.run_single(
            df=df,
            bundle=bundle,
            feature_names=["f1", "f2"],
            backtest_idx=10,
            seq_length=5,
            forecast_steps=5,
        )
        # P50 cumulative: (1.02)^5 - 1 ≈ 0.10408
        expected_cum = (1.02 ** 5) - 1
        assert result.predicted_return_p50 == pytest.approx(expected_cum, abs=1e-4)

    def test_actual_return_from_close(self):
        df = _make_df(50)
        bundle = _MockBundle(forecast_steps=5)
        engine = BacktestEngine()

        result = engine.run_single(
            df=df,
            bundle=bundle,
            feature_names=["f1", "f2"],
            backtest_idx=10,
            seq_length=5,
            forecast_steps=5,
        )
        entry = float(df["Close"].iloc[10])
        exit_ = float(df["Close"].iloc[15])
        expected_actual = (exit_ - entry) / entry
        assert result.actual_return == pytest.approx(expected_actual, abs=1e-8)

    def test_out_of_bounds_start(self):
        df = _make_df(50)
        bundle = _MockBundle(forecast_steps=5)
        engine = BacktestEngine()
        # backtest_idx < seq_length → None
        result = engine.run_single(
            df=df, bundle=bundle, feature_names=["f1", "f2"],
            backtest_idx=2, seq_length=5, forecast_steps=5,
        )
        assert result is None

    def test_out_of_bounds_end(self):
        df = _make_df(50)
        bundle = _MockBundle(forecast_steps=50)
        engine = BacktestEngine()
        # backtest_idx + forecast_steps >= len(df) → None
        result = engine.run_single(
            df=df, bundle=bundle, feature_names=["f1", "f2"],
            backtest_idx=10, seq_length=5, forecast_steps=50,
        )
        assert result is None

    def test_within_band_flag(self):
        """If model predicts wide bands, actual should be within."""
        df = _make_df(50)
        # Very wide band: P10=-0.5, P90=+0.5 (cumulative ~±50%)
        bundle = _MockBundle(per_step_returns=(-0.1, 0.0, 0.1), forecast_steps=3)
        engine = BacktestEngine()

        result = engine.run_single(
            df=df, bundle=bundle, feature_names=["f1", "f2"],
            backtest_idx=10, seq_length=5, forecast_steps=3,
        )
        assert result.within_band is True

    def test_prediction_error_nonneg(self):
        df = _make_df(50)
        bundle = _MockBundle(forecast_steps=5)
        engine = BacktestEngine()
        result = engine.run_single(
            df=df, bundle=bundle, feature_names=["f1", "f2"],
            backtest_idx=10, seq_length=5, forecast_steps=5,
        )
        assert result.prediction_error >= 0


# ══════════════════════════════════════════════════════════════════════════
# run_walk_forward
# ══════════════════════════════════════════════════════════════════════════

class TestRunWalkForward:
    def test_produces_summary(self):
        df = _make_df(100)
        bundle = _MockBundle(forecast_steps=5)
        engine = BacktestEngine()

        summary = engine.run_walk_forward(
            df=df, bundle=bundle, feature_names=["f1", "f2"],
            start_idx=10, step=10, seq_length=5, forecast_steps=5,
        )
        assert isinstance(summary, BacktestSummary)
        assert summary.total_points > 0
        assert len(summary.results) == summary.total_points

    def test_step_controls_density(self):
        df = _make_df(100)
        bundle = _MockBundle(forecast_steps=5)
        engine = BacktestEngine()

        s_dense = engine.run_walk_forward(
            df=df, bundle=bundle, feature_names=["f1", "f2"],
            start_idx=10, step=5, seq_length=5, forecast_steps=5,
        )
        s_sparse = engine.run_walk_forward(
            df=df, bundle=bundle, feature_names=["f1", "f2"],
            start_idx=10, step=20, seq_length=5, forecast_steps=5,
        )
        assert s_dense.total_points > s_sparse.total_points

    def test_empty_range(self):
        df = _make_df(10)
        bundle = _MockBundle(forecast_steps=5)
        engine = BacktestEngine()

        summary = engine.run_walk_forward(
            df=df, bundle=bundle, feature_names=["f1", "f2"],
            start_idx=5, end_idx=5, step=1, seq_length=5, forecast_steps=5,
        )
        assert summary.total_points == 0
        assert summary.results == []
        assert summary.mae == 0.0

    def test_coverage_between_0_and_100(self):
        df = _make_df(100)
        bundle = _MockBundle(forecast_steps=5)
        engine = BacktestEngine()

        summary = engine.run_walk_forward(
            df=df, bundle=bundle, feature_names=["f1", "f2"],
            start_idx=10, step=10, seq_length=5, forecast_steps=5,
        )
        assert 0 <= summary.coverage <= 100

    def test_directional_accuracy_between_0_and_100(self):
        df = _make_df(100)
        bundle = _MockBundle(forecast_steps=5)
        engine = BacktestEngine()

        summary = engine.run_walk_forward(
            df=df, bundle=bundle, feature_names=["f1", "f2"],
            start_idx=10, step=10, seq_length=5, forecast_steps=5,
        )
        assert 0 <= summary.directional_accuracy <= 100

    def test_summary_bias_sign(self):
        """If model always predicts positive but actual is mixed, bias > 0."""
        df = _make_df(100)
        # Model always predicts +1% per step
        bundle = _MockBundle(per_step_returns=(0.005, 0.01, 0.015), forecast_steps=5)
        engine = BacktestEngine()

        summary = engine.run_walk_forward(
            df=df, bundle=bundle, feature_names=["f1", "f2"],
            start_idx=10, step=10, seq_length=5, forecast_steps=5,
        )
        # bias is defined as avg(predicted - actual); with always-positive
        # predictions and mixed actuals, bias should typically be positive.
        # (Random walk is centred near 0.001 so predictions of 5% will overshoot)
        assert summary.prediction_bias > 0
