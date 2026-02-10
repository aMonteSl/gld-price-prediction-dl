"""Action-plan controller — orchestrates plan generation and persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from gldpred.decision.action_planner import (
    ActionPlan,
    build_action_plan,
    summarize_action_plan,
)
from gldpred.decision.scenario_analyzer import analyze_scenarios
from gldpred.inference.predictor import TrajectoryForecast

# Persistence directory (next to model_registry)
_PLANS_DIR = Path("data/trade_plans")


def generate_action_plan(
    forecast: TrajectoryForecast,
    *,
    horizon: int,
    take_profit_pct: float = 5.0,
    stop_loss_pct: float = 3.0,
    min_expected_return_pct: float = 1.0,
    risk_aversion_lambda: float = 0.5,
    investment: float = 10_000.0,
    model_id: str = "",
    asset: str = "",
    df: Optional[pd.DataFrame] = None,
) -> ActionPlan:
    """Build an action plan from a forecast and persist it to disk.

    Parameters
    ----------
    forecast : TrajectoryForecast
        Quantile trajectory forecast (price_paths, dates, quantiles, etc.).
    horizon : int
        Planning horizon in days.
    take_profit_pct, stop_loss_pct, min_expected_return_pct : float
        Risk thresholds in percent.
    risk_aversion_lambda : float
        Downside penalty for risk-adjusted scoring.
    investment : float
        Hypothetical investment amount for scenario analysis.
    model_id, asset : str
        Metadata.
    df : DataFrame, optional
        Market data with technical indicators for decision rationale.

    Returns
    -------
    ActionPlan
    """
    # 1. Scenario analysis
    scenarios = analyze_scenarios(
        price_paths=forecast.price_paths,
        quantiles=forecast.quantiles,
        entry_price=forecast.last_price,
        horizon=horizon,
        investment=investment,
    )

    # 2. Build action plan
    plan = build_action_plan(
        price_paths=forecast.price_paths,
        dates=forecast.dates,
        quantiles=forecast.quantiles,
        scenarios=scenarios,
        horizon=horizon,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        min_expected_return_pct=min_expected_return_pct,
        risk_aversion_lambda=risk_aversion_lambda,
        entry_price=forecast.last_price,
        df=df,
        model_id=model_id,
        asset=asset,
    )

    # 3. Persist to JSON
    _save_plan(plan)
    return plan


def _save_plan(plan: ActionPlan) -> Path:
    """Write the plan summary as JSON to the trade_plans directory."""
    _PLANS_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize_action_plan(plan)
    ts = plan.timestamp.replace(":", "-").replace("T", "_")[:19]
    filename = f"plan_{plan.asset}_{ts}.json"
    path = _PLANS_DIR / filename
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


def load_latest_plan(asset: Optional[str] = None) -> Optional[dict]:
    """Load the most recent plan JSON from disk (optionally for an asset)."""
    if not _PLANS_DIR.exists():
        return None
    files = sorted(_PLANS_DIR.glob("plan_*.json"), reverse=True)
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if asset is None or data.get("asset") == asset:
            return data
    return None
