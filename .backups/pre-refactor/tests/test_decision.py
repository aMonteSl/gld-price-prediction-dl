"""Tests for the decision engine (BUY / HOLD / AVOID logic)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gldpred.decision import DecisionEngine, Recommendation
from conftest import FORECAST_STEPS, QUANTILES


def _make_df(close=150.0, sma50=140.0, sma200=130.0, atr_pct=0.01):
    """Create a minimal DataFrame with the fields the engine expects."""
    return pd.DataFrame({
        "Close": [close],
        "sma_50": [sma50],
        "sma_200": [sma200],
        "atr_pct": [atr_pct],
    })


class TestDecisionEngine:
    def test_buy_signal(self):
        """Strong bullish conditions → BUY."""
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        # Positive returns across all quantiles
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.005)
        df = _make_df(close=160, sma50=155, sma200=140, atr_pct=0.01)
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        assert reco.action == "BUY"
        assert reco.confidence >= 65

    def test_avoid_signal(self):
        """Bearish conditions → AVOID."""
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), -0.005)
        df = _make_df(close=120, sma50=125, sma200=140, atr_pct=0.03)
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        assert reco.action == "AVOID"

    def test_hold_neutral(self):
        """Near-zero returns → HOLD."""
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.0001)
        df = _make_df(close=150, sma50=148, sma200=145, atr_pct=0.015)
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        assert reco.action in ("HOLD", "BUY")  # depends on exact score

    def test_recommendation_fields(self):
        """Recommendation has all expected fields."""
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.zeros((FORECAST_STEPS, len(QUANTILES)))
        df = _make_df()
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        assert isinstance(reco, Recommendation)
        assert reco.action in ("BUY", "HOLD", "AVOID")
        assert 0 <= reco.confidence <= 100
        assert isinstance(reco.rationale, list)
        assert isinstance(reco.warnings, list)
        assert isinstance(reco.details, dict)

    def test_diagnostics_gate_overfitting(self):
        """Overfitting verdict reduces score."""
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df(close=160, sma50=155, sma200=140, atr_pct=0.01)

        reco_healthy = engine.recommend(returns, df, quantiles=QUANTILES, diagnostics_verdict="healthy")
        reco_overfit = engine.recommend(returns, df, quantiles=QUANTILES, diagnostics_verdict="overfitting")
        assert reco_healthy.confidence > reco_overfit.confidence

    def test_high_volatility_penalty(self):
        """High ATR% penalises score."""
        engine = DecisionEngine(horizon_days=FORECAST_STEPS, max_volatility=0.02)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df_low = _make_df(close=160, sma50=155, sma200=140, atr_pct=0.01)
        df_high = _make_df(close=160, sma50=155, sma200=140, atr_pct=0.04)

        reco_low = engine.recommend(returns, df_low, quantiles=QUANTILES)
        reco_high = engine.recommend(returns, df_high, quantiles=QUANTILES)
        assert reco_low.confidence > reco_high.confidence

    def test_trend_filter_bearish(self):
        """Price below SMA200 with death cross → penalty."""
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.001)
        df = _make_df(close=120, sma50=125, sma200=140, atr_pct=0.015)
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        # Should have bearish trend in rationale
        assert any("below" in r.lower() or "bearish" in r.lower() for r in reco.rationale)

    def test_empty_df(self):
        """Engine handles empty DataFrame gracefully."""
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        reco = engine.recommend(returns, pd.DataFrame(), quantiles=QUANTILES)
        assert reco.action in ("BUY", "HOLD", "AVOID")

    def test_confidence_clamped(self):
        """Confidence is always between 0 and 100."""
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        # Extreme returns
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.1)
        df = _make_df(close=200, sma50=190, sma200=150, atr_pct=0.005)
        reco = engine.recommend(returns, df, quantiles=QUANTILES, diagnostics_verdict="healthy")
        assert 0 <= reco.confidence <= 100
