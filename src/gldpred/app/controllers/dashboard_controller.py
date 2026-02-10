"""Dashboard controller — generates quick analysis for all assets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from gldpred.config import SUPPORTED_ASSETS
from gldpred.config.assets import ASSET_CATALOG
from gldpred.data import AssetDataLoader
from gldpred.decision.action_planner import ActionPlan, build_action_plan
from gldpred.decision.engine import DecisionEngine
from gldpred.decision.scenario_analyzer import analyze_scenarios
from gldpred.features import FeatureEngineering
from gldpred.inference import TrajectoryPredictor
from gldpred.registry import ModelAssignments, ModelRegistry


@dataclass
class DashboardAssetResult:
    """Summary for one asset on the dashboard."""

    asset: str
    signal: str  # BUY / HOLD / AVOID
    confidence: float
    expected_return_pct: float
    max_risk_pct: float
    pnl_median: float
    entry_window: str
    best_exit: str
    model_label: str
    plan: Optional[ActionPlan] = None
    error: Optional[str] = None


@dataclass
class DashboardResult:
    """Collection of per-asset results sorted by expected return."""

    items: List[DashboardAssetResult] = field(default_factory=list)
    investment: float = 10_000.0
    horizon: int = 20


def run_dashboard_analysis(
    investment: float = 10_000.0,
    horizon: int = 20,
) -> DashboardResult:
    """Analyze all assets that have an assigned primary model.

    If no assets have primary models, tries the most recent model for
    each asset.  Returns a ``DashboardResult`` with items sorted by
    expected return (descending).
    """
    from gldpred.app.compare_controller import _ARCH_MAP
    from gldpred.models import TCNForecaster
    from gldpred.training import ModelTrainer

    registry = ModelRegistry()
    assignments = ModelAssignments()

    result = DashboardResult(investment=investment, horizon=horizon)

    for ticker in SUPPORTED_ASSETS:
        # Find model
        model_id = assignments.get(ticker)
        if not model_id:
            models = registry.list_models(asset=ticker)
            if models:
                model_id = models[-1]["model_id"]
        if not model_id:
            continue

        try:
            # Load data
            loader = AssetDataLoader(ticker=ticker)
            df = loader.load_data()
            eng = FeatureEngineering()
            df = eng.add_technical_indicators(df)
            feat_df = eng.select_features(df)
            feature_names = feat_df.columns.tolist()

            # Load model
            saved = registry.list_models()
            meta = next((m for m in saved if m["model_id"] == model_id), None)
            if meta is None:
                continue

            arch = meta.get("architecture", "TCN")
            cfg = meta.get("config", {})
            model_cls = _ARCH_MAP.get(arch, TCNForecaster)

            model, scaler, _m = registry.load_model(
                model_id,
                model_cls,
                input_size=len(feature_names),
                hidden_size=cfg.get("hidden_size", 64),
                num_layers=cfg.get("num_layers", 2),
                forecast_steps=cfg.get("forecast_steps", 20),
                quantiles=tuple(cfg.get("quantiles", [0.1, 0.5, 0.9])),
            )

            quantiles = tuple(cfg.get("quantiles", [0.1, 0.5, 0.9]))
            trainer = ModelTrainer(model, quantiles=quantiles, device="cpu")
            trainer.scaler = scaler

            seq_length = cfg.get("seq_length", 20)
            predictor = TrajectoryPredictor(trainer)
            forecast = predictor.predict_trajectory(
                df, feature_names, seq_length, ticker,
            )

            # Scenario analysis
            h = min(horizon, forecast.returns_quantiles.shape[0])
            scenarios = analyze_scenarios(
                price_paths=forecast.price_paths,
                quantiles=forecast.quantiles,
                entry_price=forecast.last_price,
                horizon=h,
                investment=investment,
            )

            # Build action plan
            plan = build_action_plan(
                price_paths=forecast.price_paths,
                dates=forecast.dates,
                quantiles=forecast.quantiles,
                scenarios=scenarios,
                horizon=h,
                take_profit_pct=5.0,
                stop_loss_pct=3.0,
                min_expected_return_pct=1.0,
                risk_aversion_lambda=0.5,
                entry_price=forecast.last_price,
                df=df,
                model_id=model_id,
                asset=ticker,
            )

            # Extract summary info
            entry_str = ""
            if plan.entry_window:
                ew = plan.entry_window
                entry_str = f"Days {ew.start_day}–{ew.end_day}"

            exit_str = ""
            if plan.best_exit:
                bx = plan.best_exit
                exit_str = f"Day {bx.day} ({bx.expected_return_pct:+.1f}%)"

            result.items.append(DashboardAssetResult(
                asset=ticker,
                signal=plan.overall_signal,
                confidence=plan.overall_confidence,
                expected_return_pct=scenarios.base.return_pct,
                max_risk_pct=scenarios.pessimistic.return_pct,
                pnl_median=scenarios.base.value_impact,
                entry_window=entry_str,
                best_exit=exit_str,
                model_label=meta.get("label", model_id),
                plan=plan,
            ))

        except Exception as exc:
            result.items.append(DashboardAssetResult(
                asset=ticker,
                signal="AVOID",
                confidence=0.0,
                expected_return_pct=0.0,
                max_risk_pct=0.0,
                pnl_median=0.0,
                entry_window="",
                best_exit="",
                model_label="",
                error=str(exc),
            ))

    # Sort by expected return descending
    result.items.sort(key=lambda x: x.expected_return_pct, reverse=True)
    # Assign ranks
    for i, item in enumerate(result.items):
        pass  # rank is just position + 1

    return result
