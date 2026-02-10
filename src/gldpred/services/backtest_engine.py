"""Walk-forward backtesting engine.

Simulates what a model would have recommended at historical dates and
compares predicted returns against actual market outcomes.

Usage::

    from gldpred.services.backtest_engine import BacktestEngine, BacktestResult

    engine = BacktestEngine()
    results = engine.run_walk_forward(
        df=data_with_features,
        bundle=model_bundle,
        feature_names=feature_names,
        start_idx=200,
        end_idx=400,
        step=20,
        investment=10_000,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Result of a single simulated prediction at a historical date."""

    backtest_date: str              # ISO — the date inference was "run"
    horizon: int                    # forecast steps (K)
    entry_price: float              # Close on backtest_date
    investment: float

    # ── Predicted ─────────────────────────────────────────────────────
    predicted_return_p10: float     # cumulative P10 return at horizon
    predicted_return_p50: float     # cumulative P50 return (median)
    predicted_return_p90: float     # cumulative P90 return
    predicted_pnl: float            # P50 return × investment

    # ── Actual ────────────────────────────────────────────────────────
    actual_return: float            # cumulative actual return at horizon
    actual_pnl: float              # actual return × investment

    # ── Comparison ────────────────────────────────────────────────────
    prediction_error: float         # |predicted_p50 − actual|
    within_band: bool               # P10 ≤ actual ≤ P90


@dataclass
class BacktestSummary:
    """Aggregate statistics for a walk-forward backtest."""

    total_points: int
    avg_predicted_return: float
    avg_actual_return: float
    prediction_bias: float          # avg(predicted − actual)
    mae: float                      # mean absolute error
    coverage: float                 # % of actuals within P10–P90
    directional_accuracy: float     # % of correct sign predictions
    avg_predicted_pnl: float
    avg_actual_pnl: float
    results: List[BacktestResult] = field(default_factory=list)


class BacktestEngine:
    """Simulate historical model predictions and compare vs actuals.

    This engine does **not** retrain models — it evaluates previously
    trained models on historical data slices to measure how well they
    would have performed in a real-time setting.
    """

    def run_single(
        self,
        df: pd.DataFrame,
        bundle: Any,
        feature_names: List[str],
        backtest_idx: int,
        seq_length: int = 20,
        forecast_steps: int = 20,
        investment: float = 10_000,
    ) -> Optional[BacktestResult]:
        """Run a single backtest at a specific index in the DataFrame.

        Args:
            df: DataFrame with OHLCV + engineered features (full history).
            bundle: ``ModelBundle`` with ``.predict(X)`` → ``(N, K, Q)``.
            feature_names: Feature column names that match the bundle.
            backtest_idx: Row index in *df* to treat as "today".
                Must satisfy ``backtest_idx >= seq_length`` and
                ``backtest_idx + forecast_steps < len(df)``.
            seq_length: Lookback window for the model input.
            forecast_steps: K — number of future steps the model outputs.
            investment: Notional investment (for P&L).

        Returns:
            ``BacktestResult`` or ``None`` if the index is out of bounds.
        """
        n = len(df)
        if backtest_idx < seq_length:
            return None
        if backtest_idx + forecast_steps >= n:
            return None

        # Extract the input sequence ending at backtest_idx
        start = backtest_idx - seq_length
        X_raw = df[feature_names].values[start:backtest_idx]
        X_3d = X_raw.reshape(1, seq_length, -1).astype(np.float32)

        # Run inference → (1, K, Q) numpy
        pred = bundle.predict(X_3d)          # (1, K, Q)
        returns_q = pred[0]                  # (K, Q)

        quantiles = list(bundle.quantiles)
        q10_idx = _nearest_idx(quantiles, 0.1)
        q50_idx = _nearest_idx(quantiles, 0.5)
        q90_idx = _nearest_idx(quantiles, 0.9)

        # Cumulative returns at horizon end
        cum_p10 = float(_cumulative_return(returns_q[:, q10_idx]))
        cum_p50 = float(_cumulative_return(returns_q[:, q50_idx]))
        cum_p90 = float(_cumulative_return(returns_q[:, q90_idx]))

        # Actual cumulative return
        entry_price = float(df["Close"].iloc[backtest_idx])
        end_idx = backtest_idx + forecast_steps
        exit_price = float(df["Close"].iloc[end_idx])
        actual_return = (exit_price - entry_price) / entry_price

        backtest_date = str(df.index[backtest_idx])

        return BacktestResult(
            backtest_date=backtest_date,
            horizon=forecast_steps,
            entry_price=entry_price,
            investment=investment,
            predicted_return_p10=cum_p10,
            predicted_return_p50=cum_p50,
            predicted_return_p90=cum_p90,
            predicted_pnl=cum_p50 * investment,
            actual_return=actual_return,
            actual_pnl=actual_return * investment,
            prediction_error=abs(cum_p50 - actual_return),
            within_band=(cum_p10 <= actual_return <= cum_p90),
        )

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        bundle: Any,
        feature_names: List[str],
        start_idx: int,
        end_idx: Optional[int] = None,
        step: int = 20,
        seq_length: int = 20,
        forecast_steps: int = 20,
        investment: float = 10_000,
    ) -> BacktestSummary:
        """Walk-forward backtest across a date range.

        Iterates from *start_idx* to *end_idx* in steps of *step*,
        running ``run_single`` at each position, then aggregates.

        Args:
            df: Full DataFrame with features.
            bundle: ``ModelBundle``.
            feature_names: Feature columns.
            start_idx: First row index to test (must be ≥ seq_length).
            end_idx: Last row index (exclusive). Default: ``len(df) − forecast_steps``.
            step: Days between each backtest point.
            seq_length: Model lookback window.
            forecast_steps: K — forecast horizon.
            investment: Notional investment amount.

        Returns:
            ``BacktestSummary`` with aggregated statistics and result list.
        """
        n = len(df)
        if end_idx is None:
            end_idx = n - forecast_steps

        results: List[BacktestResult] = []
        idx = max(start_idx, seq_length)
        while idx < end_idx:
            r = self.run_single(
                df=df,
                bundle=bundle,
                feature_names=feature_names,
                backtest_idx=idx,
                seq_length=seq_length,
                forecast_steps=forecast_steps,
                investment=investment,
            )
            if r is not None:
                results.append(r)
            idx += step

        return _summarize(results, investment)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _cumulative_return(daily_returns: np.ndarray) -> float:
    """Compound daily returns into a cumulative return."""
    return float(np.prod(1 + daily_returns) - 1)


def _nearest_idx(quantiles: List[float], target: float) -> int:
    """Find the index of the quantile nearest to *target*."""
    return int(np.argmin([abs(q - target) for q in quantiles]))


def _summarize(results: List[BacktestResult], investment: float) -> BacktestSummary:
    """Aggregate a list of backtest results into summary statistics."""
    if not results:
        return BacktestSummary(
            total_points=0,
            avg_predicted_return=0.0,
            avg_actual_return=0.0,
            prediction_bias=0.0,
            mae=0.0,
            coverage=0.0,
            directional_accuracy=0.0,
            avg_predicted_pnl=0.0,
            avg_actual_pnl=0.0,
            results=[],
        )

    predicted = [r.predicted_return_p50 for r in results]
    actual = [r.actual_return for r in results]
    errors = [r.prediction_error for r in results]
    within = [r.within_band for r in results]

    n = len(results)
    dir_correct = sum(
        1 for p, a in zip(predicted, actual) if (p >= 0) == (a >= 0)
    )

    return BacktestSummary(
        total_points=n,
        avg_predicted_return=sum(predicted) / n,
        avg_actual_return=sum(actual) / n,
        prediction_bias=sum(p - a for p, a in zip(predicted, actual)) / n,
        mae=sum(errors) / n,
        coverage=sum(within) / n * 100,
        directional_accuracy=dir_correct / n * 100,
        avg_predicted_pnl=sum(r.predicted_pnl for r in results) / n,
        avg_actual_pnl=sum(r.actual_pnl for r in results) / n,
        results=results,
    )
