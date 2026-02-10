"""Tests for the action-plan engine (scenario_analyzer + action_planner)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gldpred.decision.action_planner import (
    ActionPlan,
    DayRecommendation,
    DecisionRationale,
    EntryWindow,
    ExitPoint,
    build_action_plan,
    summarize_action_plan,
)
from gldpred.decision.scenario_analyzer import (
    ScenarioAnalysis,
    ScenarioOutcome,
    analyze_scenarios,
)


# ── helpers ──────────────────────────────────────────────────────────

def _make_price_paths(
    entry: float,
    daily_returns_p50: list[float],
    spread: float = 0.01,
) -> np.ndarray:
    """Build a synthetic (K+1, 3) price_paths array.

    ``spread`` controls the P10→P50 and P50→P90 offset (per step).
    """
    K = len(daily_returns_p50)
    paths = np.zeros((K + 1, 3))
    paths[0, :] = entry
    for k in range(K):
        r = daily_returns_p50[k]
        paths[k + 1, 0] = paths[k, 0] * (1 + r - spread)   # P10
        paths[k + 1, 1] = paths[k, 1] * (1 + r)             # P50
        paths[k + 1, 2] = paths[k, 2] * (1 + r + spread)    # P90
    return paths


def _make_dates(K: int):
    start = pd.Timestamp("2026-01-05")
    return pd.bdate_range(start=start, periods=K).tolist()


QUANTILES = [0.1, 0.5, 0.9]


def _build(
    daily_returns: list[float],
    entry: float = 100.0,
    spread: float = 0.003,
    horizon: int | None = None,
    tp: float = 5.0,
    sl: float = 3.0,
    mer: float = 1.0,
    lam: float = 0.5,
    investment: float = 10_000.0,
    df: pd.DataFrame | None = None,
    asset: str = "GLD",
    model_id: str = "",
) -> ActionPlan:
    """Shorthand to build an action plan from daily returns."""
    K = len(daily_returns)
    paths = _make_price_paths(entry, daily_returns, spread=spread)
    dates = _make_dates(K)
    h = horizon if horizon is not None else K

    scenarios = analyze_scenarios(
        paths, QUANTILES, entry_price=entry, horizon=h, investment=investment,
    )
    return build_action_plan(
        paths, dates, QUANTILES, scenarios,
        horizon=h,
        take_profit_pct=tp,
        stop_loss_pct=sl,
        min_expected_return_pct=mer,
        risk_aversion_lambda=lam,
        entry_price=entry,
        df=df,
        model_id=model_id,
        asset=asset,
    )


# ======================================================================
# Scenario Analyzer
# ======================================================================

class TestAnalyzeScenarios:
    def test_basic(self):
        paths = np.array([
            [100.0, 100.0, 100.0],
            [97.0, 102.0, 106.0],
            [95.0, 105.0, 112.0],
        ])
        sa = analyze_scenarios(
            paths, QUANTILES, entry_price=100.0, horizon=2, investment=10_000.0,
        )
        assert isinstance(sa, ScenarioAnalysis)
        assert sa.base.return_pct == pytest.approx(5.0, abs=0.01)
        assert sa.pessimistic.return_pct == pytest.approx(-5.0, abs=0.01)
        assert sa.optimistic.return_pct == pytest.approx(12.0, abs=0.01)

    def test_value_impact(self):
        paths = np.array([
            [100.0, 100.0, 100.0],
            [90.0, 110.0, 120.0],
        ])
        sa = analyze_scenarios(
            paths, QUANTILES, entry_price=100.0, horizon=1, investment=5_000.0,
        )
        assert sa.base.value_impact == pytest.approx(500.0, abs=0.01)
        assert sa.pessimistic.value_impact == pytest.approx(-500.0, abs=0.01)
        assert sa.optimistic.value_impact == pytest.approx(1000.0, abs=0.01)

    def test_horizon_clamped(self):
        paths = _make_price_paths(100.0, [0.01] * 3, spread=0.001)
        sa = analyze_scenarios(
            paths, QUANTILES, entry_price=100.0, horizon=100,
        )
        assert sa.horizon == 3

    def test_price_path_length(self):
        paths = _make_price_paths(100.0, [0.01] * 5, spread=0.001)
        sa = analyze_scenarios(
            paths, QUANTILES, entry_price=100.0, horizon=5,
        )
        assert len(sa.base.price_path) == 6  # horizon + 1

    def test_frozen(self):
        paths = _make_price_paths(100.0, [0.01] * 3, spread=0.001)
        sa = analyze_scenarios(
            paths, QUANTILES, entry_price=100.0, horizon=3,
        )
        with pytest.raises(AttributeError):
            sa.investment = 999  # type: ignore[misc]


# ======================================================================
# Action Planner — Uptrend / BUY
# ======================================================================

class TestActionPlanUptrend:
    def test_buy_in_uptrend(self):
        """Steady uptrend → at least one BUY action."""
        plan = _build([0.01] * 10)
        assert any(d.action == "BUY" for d in plan.daily_actions)

    def test_overall_signal_buy(self):
        plan = _build([0.01] * 10)
        assert plan.overall_signal in ("BUY", "HOLD")

    def test_entry_window_found(self):
        plan = _build([0.01] * 10)
        assert plan.entry_window is not None
        assert plan.entry_window.start_day >= 1

    def test_best_exit_found(self):
        plan = _build([0.01] * 10)
        assert plan.best_exit is not None
        assert plan.best_exit.day >= 1

    def test_daily_actions_count(self):
        plan = _build([0.01] * 10)
        assert len(plan.daily_actions) == 10

    def test_narrative_not_empty(self):
        plan = _build([0.01] * 5)
        assert plan.narrative
        assert len(plan.narrative) > 5


# ======================================================================
# Action Planner — Downtrend / AVOID
# ======================================================================

class TestActionPlanDowntrend:
    def test_downtrend_mostly_avoid(self):
        """Steady downtrend → mostly AVOID."""
        plan = _build([-0.01] * 10)
        avoid_count = sum(1 for d in plan.daily_actions if d.action == "AVOID")
        assert avoid_count >= 5

    def test_overall_signal_avoid(self):
        plan = _build([-0.01] * 10)
        assert plan.overall_signal == "AVOID"

    def test_no_entry_window(self):
        plan = _build([-0.02] * 10, spread=0.005)
        assert plan.entry_window is None


# ======================================================================
# Action Planner — Take-Profit / Stop-Loss triggers
# ======================================================================

class TestTPandSL:
    def test_tp_triggers_sell(self):
        """Large returns → SELL when TP hit."""
        plan = _build([0.03] * 5, tp=5.0, sl=3.0)
        sell_days = [d for d in plan.daily_actions if d.action == "SELL"]
        assert len(sell_days) >= 1
        # After SELL, remaining should be AVOID
        if sell_days:
            sell_day = sell_days[0].day
            for d in plan.daily_actions:
                if d.day > sell_day:
                    assert d.action == "AVOID"

    def test_sl_triggers_sell(self):
        """Large drops → SELL when SL breached (P10 return)."""
        plan = _build([-0.015] * 5, spread=0.02, sl=3.0, mer=0.0)
        sell_or_avoid = [d for d in plan.daily_actions if d.action in ("SELL", "AVOID")]
        # Should have defensive actions
        assert len(sell_or_avoid) >= 1

    def test_closed_after_sell(self):
        """All days after SELL are AVOID (position closed)."""
        plan = _build([0.04] * 5, tp=5.0)
        found_sell = False
        for d in plan.daily_actions:
            if d.action == "SELL":
                found_sell = True
                continue
            if found_sell:
                assert d.action == "AVOID", (
                    f"Day {d.day} should be AVOID after SELL, got {d.action}"
                )


# ======================================================================
# Entry Window Detection
# ======================================================================

class TestEntryWindow:
    def test_entry_window_range(self):
        plan = _build([0.01] * 10)
        if plan.entry_window:
            ew = plan.entry_window
            assert ew.start_day <= ew.end_day
            assert ew.start_day >= 1
            assert ew.end_day <= 10

    def test_all_buy_days_in_window(self):
        """All BUY-classified days should be contiguous."""
        plan = _build([0.01] * 10)
        buy_days = [d.day for d in plan.daily_actions if d.action == "BUY"]
        if len(buy_days) > 1:
            # Check they form a contiguous run
            for i in range(1, len(buy_days)):
                assert buy_days[i] - buy_days[i - 1] == 1


# ======================================================================
# Best Exit
# ======================================================================

class TestBestExit:
    def test_best_exit_has_positive_return(self):
        plan = _build([0.01] * 10)
        if plan.best_exit:
            assert plan.best_exit.expected_return_pct > 0

    def test_best_exit_before_decline(self):
        """Rise-then-fall: best exit should be at or before the peak."""
        plan = _build(
            [0.005, 0.01, 0.015, 0.005, -0.005],
            tp=20.0, sl=20.0, mer=0.5,
        )
        if plan.best_exit:
            assert plan.best_exit.day <= 4

    def test_best_exit_rationale(self):
        plan = _build([0.01] * 5)
        if plan.best_exit:
            assert plan.best_exit.rationale
            assert "Day" in plan.best_exit.rationale


# ======================================================================
# Decision Rationale
# ======================================================================

class TestRationale:
    def test_rationale_fields(self):
        plan = _build([0.01] * 5)
        r = plan.rationale
        assert isinstance(r, DecisionRationale)
        assert r.trend_confirmation
        assert r.volatility_regime
        assert r.quantile_risk
        assert r.today_assessment

    def test_rationale_with_df(self):
        """When a DataFrame with SMA is provided, trend is richer."""
        N = 300
        close = 100.0 + np.cumsum(np.random.randn(N) * 0.5)
        df = pd.DataFrame({
            "Close": close,
            "sma_50": pd.Series(close).rolling(50).mean(),
            "sma_200": pd.Series(close).rolling(200).mean(),
            "atr_pct": np.abs(np.random.randn(N)) * 0.02,
        })
        plan = _build([0.01] * 5, df=df)
        assert "SMA" in plan.rationale.trend_confirmation or "No" in plan.rationale.trend_confirmation

    def test_rationale_no_df(self):
        plan = _build([0.01] * 5, df=None)
        assert "No market data" in plan.rationale.trend_confirmation


# ======================================================================
# Narrative
# ======================================================================

class TestNarrative:
    def test_uptrend_narrative(self):
        plan = _build([0.01] * 5)
        assert plan.narrative
        # Should mention buy and exit/hold
        assert any(w in plan.narrative.lower() for w in ("buy", "hold", "exit", "avoid"))

    def test_downtrend_narrative(self):
        plan = _build([-0.01] * 5)
        assert "avoid" in plan.narrative.lower() or "not favorable" in plan.narrative.lower()


# ======================================================================
# Overall Signal
# ======================================================================

class TestOverallSignal:
    def test_valid_signals(self):
        for returns in [[0.01] * 5, [-0.01] * 5, [0.001] * 5]:
            plan = _build(returns)
            assert plan.overall_signal in ("BUY", "HOLD", "SELL", "AVOID")
            assert 0 <= plan.overall_confidence <= 100

    def test_weak_return_avoid(self):
        plan = _build([-0.005] * 5, mer=3.0)
        assert plan.overall_signal == "AVOID"


# ======================================================================
# Summarize
# ======================================================================

class TestSummarize:
    def test_serialisable(self):
        plan = _build(
            [0.01] * 5, model_id="test-model", asset="GLD",
        )
        summary = summarize_action_plan(plan)
        assert isinstance(summary, dict)
        assert summary["overall_signal"] in ("BUY", "HOLD", "SELL", "AVOID")
        assert summary["asset"] == "GLD"
        assert summary["model_id"] == "test-model"
        assert "daily_plan" in summary
        # json-serialisable
        json.dumps(summary)

    def test_summary_fields(self):
        plan = _build([0.02] * 3)
        summary = summarize_action_plan(plan)
        assert "overall_signal" in summary
        assert "rationale" in summary
        assert "scenarios" in summary

        row = summary["daily_plan"][0]
        for key in ("day", "date", "action", "confidence", "rationale",
                     "price_p50", "ret_p50_pct", "risk_score"):
            assert key in row, f"Missing key '{key}' in daily plan"

    def test_entry_window_in_summary(self):
        plan = _build([0.01] * 10)
        summary = summarize_action_plan(plan)
        if plan.entry_window:
            assert "entry_window" in summary
            ew = summary["entry_window"]
            assert "start_day" in ew
            assert "end_day" in ew

    def test_best_exit_in_summary(self):
        plan = _build([0.01] * 10)
        summary = summarize_action_plan(plan)
        if plan.best_exit:
            assert "best_exit" in summary
            bx = summary["best_exit"]
            assert "day" in bx
            assert "expected_return_pct" in bx


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    def test_horizon_1(self):
        plan = _build([0.02], horizon=1)
        assert plan.horizon == 1
        assert len(plan.daily_actions) == 1

    def test_horizon_clamped(self):
        plan = _build([0.01] * 3, horizon=100)
        assert plan.horizon == 3

    def test_zero_spread(self):
        plan = _build([0.02] * 5, spread=0.0)
        assert plan.overall_signal in ("BUY", "HOLD", "SELL", "AVOID")

    def test_metadata_propagated(self):
        plan = _build(
            [0.01] * 3, model_id="abc123", asset="BTC-USD",
        )
        assert plan.model_id == "abc123"
        assert plan.asset == "BTC-USD"
        assert plan.params["take_profit_pct"] == 5.0

    def test_day_recommendations_have_prices(self):
        plan = _build([0.01] * 5)
        for d in plan.daily_actions:
            assert d.price_p10 > 0
            assert d.price_p50 > 0
            assert d.price_p90 > 0
            assert d.price_p10 <= d.price_p50 <= d.price_p90

    def test_confidence_in_range(self):
        plan = _build([0.01] * 5)
        for d in plan.daily_actions:
            assert 0 <= d.confidence <= 100

    def test_scenarios_attached(self):
        plan = _build([0.01] * 5)
        assert isinstance(plan.scenarios, ScenarioAnalysis)
        assert plan.scenarios.base.return_pct > 0
