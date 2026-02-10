"""Tests for DecisionPolicy — transparent scoring pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gldpred.core.policy import DecisionPolicy, PolicyResult, ScoreFactor


FORECAST_STEPS = 20
QUANTILES = (0.1, 0.5, 0.9)


def _make_df(close=150.0, sma50=140.0, sma200=130.0, atr_pct=0.01):
    """Minimal DataFrame for the engine."""
    return pd.DataFrame({
        "Close": [close],
        "sma_50": [sma50],
        "sma_200": [sma200],
        "atr_pct": [atr_pct],
    })


class TestDecisionPolicy:

    def test_evaluate_returns_policy_result(self):
        policy = DecisionPolicy(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df(close=160, sma50=155, sma200=140, atr_pct=0.01)
        result = policy.evaluate(returns, df, QUANTILES)
        assert isinstance(result, PolicyResult)
        assert result.action in ("BUY", "HOLD", "AVOID")
        assert 0 <= result.confidence <= 100

    def test_factors_are_populated(self):
        policy = DecisionPolicy(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.005)
        df = _make_df(close=160, sma50=155, sma200=140, atr_pct=0.01)
        result = policy.evaluate(returns, df, QUANTILES)
        assert len(result.factors) > 0
        # Must include base factor
        names = [f.name for f in result.factors]
        assert "base" in names
        assert "expected_return" in names

    def test_each_factor_has_labels(self):
        policy = DecisionPolicy()
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df()
        result = policy.evaluate(returns, df, QUANTILES)
        for f in result.factors:
            assert isinstance(f, ScoreFactor)
            assert f.label_en
            assert f.label_es
            assert f.max_possible > 0

    def test_factor_sentiments(self):
        policy = DecisionPolicy(horizon_days=FORECAST_STEPS)
        # Bullish
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.005)
        df = _make_df(close=160, sma50=155, sma200=140, atr_pct=0.01)
        result = policy.evaluate(returns, df, QUANTILES)
        er = next(f for f in result.factors if f.name == "expected_return")
        assert er.sentiment == "positive"
        assert er.contribution > 0

    def test_negative_sentiment_on_bearish(self):
        policy = DecisionPolicy(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), -0.005)
        df = _make_df(close=120, sma50=125, sma200=140, atr_pct=0.01)
        result = policy.evaluate(returns, df, QUANTILES)
        er = next(f for f in result.factors if f.name == "expected_return")
        assert er.sentiment == "negative"
        assert er.contribution < 0

    def test_recommendation_property(self):
        policy = DecisionPolicy(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df(close=160, sma50=155, sma200=140)
        result = policy.evaluate(returns, df, QUANTILES)
        rec = result.recommendation
        assert rec.action == result.action
        assert rec.confidence == result.confidence

    def test_market_regime_exposed(self):
        policy = DecisionPolicy(horizon_days=FORECAST_STEPS)
        returns = np.zeros((FORECAST_STEPS, len(QUANTILES)))
        df = _make_df()
        # Need at least 20 rows for regime detection
        df_long = pd.DataFrame({
            "Close": np.linspace(100, 150, 50),
            "sma_50": np.linspace(95, 145, 50),
            "sma_200": np.linspace(90, 130, 50),
            "atr_pct": [0.01] * 50,
        })
        result = policy.evaluate(returns, df_long, QUANTILES)
        assert result.market_regime in (
            "trending_up", "trending_down", "ranging",
            "high_volatility", "unknown",
        )

    def test_cumulative_return_and_spread(self):
        policy = DecisionPolicy(horizon_days=5)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.002)
        # P10 = -0.001, P50 = 0.002, P90 = 0.005
        returns[:, 0] = -0.001
        returns[:, 2] = 0.005
        df = _make_df()
        result = policy.evaluate(returns, df, QUANTILES)
        assert result.cumulative_return != 0.0
        assert result.uncertainty_spread > 0

    def test_diagnostics_factor_present_when_provided(self):
        policy = DecisionPolicy(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df(close=160, sma50=155, sma200=140)
        result = policy.evaluate(
            returns, df, QUANTILES, diagnostics_verdict="healthy",
        )
        names = [f.name for f in result.factors]
        assert "diagnostics" in names
        diag = next(f for f in result.factors if f.name == "diagnostics")
        assert diag.contribution > 0
        assert diag.sentiment == "positive"

    def test_avoid_on_bearish(self):
        policy = DecisionPolicy(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), -0.005)
        df = _make_df(close=120, sma50=125, sma200=140, atr_pct=0.03)
        result = policy.evaluate(returns, df, QUANTILES)
        assert result.action == "AVOID"

    def test_buy_on_bullish(self):
        policy = DecisionPolicy(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.005)
        df = _make_df(close=160, sma50=155, sma200=140, atr_pct=0.01)
        result = policy.evaluate(returns, df, QUANTILES)
        assert result.action == "BUY"

    def test_score_factors_sum_approximates_total(self):
        """Sum of factor contributions should be close to total score."""
        policy = DecisionPolicy(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df(close=160, sma50=155, sma200=140)
        result = policy.evaluate(returns, df, QUANTILES)
        factor_sum = sum(f.contribution for f in result.factors)
        # Score is clamped [0, 100], so factor_sum might differ at edges
        assert abs(factor_sum - result.total_score) < 5.0 or result.total_score in (0.0, 100.0)

    def test_risk_metrics_present(self):
        policy = DecisionPolicy(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df()
        result = policy.evaluate(returns, df, QUANTILES)
        assert hasattr(result.risk, "stop_loss_pct")
        assert hasattr(result.risk, "take_profit_pct")
        assert hasattr(result.risk, "risk_reward_ratio")
