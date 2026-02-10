"""Tests for PortfolioComparator and the enhanced DecisionEngine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gldpred.decision import (
    DecisionEngine,
    PortfolioComparator,
    Recommendation,
    RecommendationHistory,
    RiskMetrics,
)
from conftest import FORECAST_STEPS, QUANTILES


# ── helpers ──────────────────────────────────────────────────────────────

def _make_df(close=150.0, sma50=140.0, sma200=130.0, atr_pct=0.01, nrows=30):
    """Create a DataFrame with enough rows for regime detection."""
    closes = np.linspace(close * 0.95, close, nrows)
    return pd.DataFrame({
        "Close": closes,
        "sma_50": [sma50] * nrows,
        "sma_200": [sma200] * nrows,
        "atr_pct": [atr_pct] * nrows,
        "volatility_20": [0.01] * nrows,
    })


class _FakeForecast:
    """Minimal forecast stub for portfolio comparison."""

    def __init__(self, last_price=150.0, K=5, quantiles=(0.1, 0.5, 0.9)):
        self.last_price = last_price
        self.last_date = pd.Timestamp("2024-01-01")
        self.quantiles = list(quantiles)
        Q = len(quantiles)
        self.returns_quantiles = np.full((K, Q), 0.005)
        self.price_paths = np.ones((K + 1, Q)) * last_price
        for k in range(K):
            self.price_paths[k + 1, :] = self.price_paths[k, :] * (1 + self.returns_quantiles[k, :])
        self.dates = pd.bdate_range("2024-01-02", periods=K).tolist()


# ── Risk metrics tests ──────────────────────────────────────────────────

class TestRiskMetrics:
    def test_recommendation_has_risk(self):
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.005)
        df = _make_df(close=160, sma50=155, sma200=140)
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        assert isinstance(reco.risk, RiskMetrics)

    def test_stop_loss_is_negative(self):
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df()
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        assert reco.risk.stop_loss_pct <= 0

    def test_take_profit_is_positive(self):
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df()
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        assert reco.risk.take_profit_pct >= 0

    def test_risk_reward_positive(self):
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.005)
        df = _make_df()
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        assert reco.risk.risk_reward_ratio >= 0

    def test_volatility_regime_values(self):
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df(atr_pct=0.01)
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        assert reco.risk.volatility_regime in ("low", "normal", "high")


# ── Market regime tests ─────────────────────────────────────────────────

class TestMarketRegime:
    def test_regime_returned_in_details(self):
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df()
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        assert "market_regime" in reco.details

    def test_regime_trending_up(self):
        # Strongly uptrending prices
        closes = np.linspace(100, 200, 30)
        df = pd.DataFrame({
            "Close": closes,
            "sma_50": [140] * 30,
            "sma_200": [130] * 30,
            "atr_pct": [0.01] * 30,
            "volatility_20": [0.01] * 30,
        })
        regime = DecisionEngine._detect_regime(df)
        assert regime == "trending_up"

    def test_regime_trending_down(self):
        closes = np.linspace(200, 100, 30)
        df = pd.DataFrame({
            "Close": closes,
            "sma_50": [140] * 30,
            "sma_200": [130] * 30,
            "atr_pct": [0.01] * 30,
            "volatility_20": [0.01] * 30,
        })
        regime = DecisionEngine._detect_regime(df)
        assert regime == "trending_down"

    def test_regime_unknown_short_df(self):
        df = pd.DataFrame({"Close": [100.0, 101.0]})
        regime = DecisionEngine._detect_regime(df)
        assert regime == "unknown"


# ── Conflicting signals tests ───────────────────────────────────────────

class TestConflictingSignals:
    def test_positive_forecast_bearish_trend(self):
        penalty, msgs = DecisionEngine._check_conflicting_signals(
            cum_return=0.01, trend_score=-15, vol_score=0, spread=0.01,
        )
        assert penalty < 0
        assert len(msgs) > 0

    def test_no_conflict(self):
        penalty, msgs = DecisionEngine._check_conflicting_signals(
            cum_return=0.01, trend_score=10, vol_score=5, spread=0.005,
        )
        assert penalty == 0
        assert len(msgs) == 0


# ── Recommendation timestamp ────────────────────────────────────────────

class TestRecommendationTimestamp:
    def test_has_timestamp(self):
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        returns = np.full((FORECAST_STEPS, len(QUANTILES)), 0.003)
        df = _make_df()
        reco = engine.recommend(returns, df, quantiles=QUANTILES)
        assert hasattr(reco, "timestamp")
        assert len(reco.timestamp) > 0


# ── Recommendation history tests ────────────────────────────────────────

class TestRecommendationHistory:
    def test_add_and_get(self):
        history = RecommendationHistory()
        reco = Recommendation(action="BUY", confidence=80)
        history.add("GLD", reco)
        assert len(history) == 1
        assert history.get_history()[0]["asset"] == "GLD"

    def test_filter_by_asset(self):
        history = RecommendationHistory()
        history.add("GLD", Recommendation(action="BUY", confidence=80))
        history.add("SLV", Recommendation(action="HOLD", confidence=50))
        gld_only = history.get_history(asset="GLD")
        assert len(gld_only) == 1
        assert gld_only[0]["action"] == "BUY"

    def test_clear(self):
        history = RecommendationHistory()
        history.add("GLD", Recommendation(action="BUY", confidence=80))
        history.clear()
        assert len(history) == 0


# ── Portfolio comparator tests ──────────────────────────────────────────

class TestPortfolioComparator:
    def test_compare_returns_result(self):
        comparator = PortfolioComparator(horizon_days=5)
        forecasts = {
            "GLD": _FakeForecast(last_price=150),
            "SLV": _FakeForecast(last_price=25),
        }
        dfs = {
            "GLD": _make_df(close=150),
            "SLV": _make_df(close=25, sma50=24, sma200=22),
        }
        result = comparator.compare(forecasts, dfs, investment=1000)
        assert len(result.outcomes) == 2
        assert result.best_asset in ("GLD", "SLV")
        assert result.investment == 1000

    def test_outcomes_are_ranked(self):
        comparator = PortfolioComparator(horizon_days=5)
        forecasts = {
            "GLD": _FakeForecast(last_price=150),
            "SLV": _FakeForecast(last_price=25),
        }
        dfs = {
            "GLD": _make_df(close=150),
            "SLV": _make_df(close=25, sma50=24, sma200=22),
        }
        result = comparator.compare(forecasts, dfs, investment=1000)
        ranks = [o.rank for o in result.outcomes]
        assert ranks == [1, 2]

    def test_pnl_computed(self):
        comparator = PortfolioComparator(horizon_days=5)
        forecasts = {"GLD": _FakeForecast(last_price=100)}
        dfs = {"GLD": _make_df(close=100)}
        result = comparator.compare(forecasts, dfs, investment=1000)
        o = result.outcomes[0]
        assert o.shares == pytest.approx(10.0)
        assert o.investment == 1000

    def test_empty_forecasts(self):
        comparator = PortfolioComparator(horizon_days=5)
        result = comparator.compare({}, {}, investment=1000)
        assert len(result.outcomes) == 0
        assert result.best_asset == ""

    def test_summary_string(self):
        comparator = PortfolioComparator(horizon_days=5)
        forecasts = {"GLD": _FakeForecast(last_price=150)}
        dfs = {"GLD": _make_df(close=150)}
        result = comparator.compare(forecasts, dfs, investment=1000)
        assert "GLD" in result.summary
        assert "Investment" in result.summary

    def test_each_outcome_has_recommendation(self):
        comparator = PortfolioComparator(horizon_days=5)
        forecasts = {"GLD": _FakeForecast(last_price=150)}
        dfs = {"GLD": _make_df(close=150)}
        result = comparator.compare(forecasts, dfs, investment=1000)
        o = result.outcomes[0]
        assert isinstance(o.recommendation, Recommendation)
        assert o.recommendation.action in ("BUY", "HOLD", "AVOID")
