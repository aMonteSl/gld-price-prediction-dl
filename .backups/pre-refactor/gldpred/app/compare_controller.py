"""Compare-tab controller — orchestrates multi-asset portfolio comparison.

Pulls data, loads models from registry, runs forecasts, and feeds results
to ``PortfolioComparator``. The Streamlit GUI only renders; all logic
lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd
import torch.nn as nn

from gldpred.config import SUPPORTED_ASSETS
from gldpred.config.assets import ASSET_CATALOG
from gldpred.data import AssetDataLoader
from gldpred.decision.portfolio import ComparisonResult, PortfolioComparator
from gldpred.features import FeatureEngineering
from gldpred.inference import TrajectoryPredictor
from gldpred.models import GRUForecaster, LSTMForecaster, TCNForecaster
from gldpred.registry import ModelAssignments, ModelRegistry
from gldpred.training import ModelTrainer

_ARCH_MAP: Dict[str, Type[nn.Module]] = {
    "GRU": GRUForecaster,
    "LSTM": LSTMForecaster,
    "TCN": TCNForecaster,
}


@dataclass
class CompareRow:
    """One row in the Compare tab — an (asset, model_id) pair."""

    ticker: str
    model_id: str


def build_compare_rows_from_assignments() -> List[CompareRow]:
    """Return a ``CompareRow`` for every asset that has a primary model."""
    assignments = ModelAssignments()
    return [
        CompareRow(ticker=t, model_id=mid)
        for t, mid in assignments.get_all().items()
    ]


def available_models_for_asset(ticker: str) -> List[Dict[str, Any]]:
    """Return registry metadata for all models trained on *ticker*."""
    return ModelRegistry().list_models(asset=ticker)


def run_comparison(
    rows: List[CompareRow],
    investment: float,
    horizon: int,
) -> ComparisonResult:
    """Execute a full comparison pipeline.

    For each ``CompareRow`` in *rows*:
    1. Download / cache data.
    2. Load model from registry.
    3. Produce forecast.
    4. Collect diagnostics verdict + max-volatility.

    Then feed everything into ``PortfolioComparator.compare()``.

    Args:
        rows: Which (asset, model) pairs to compare.
        investment: Dollar amount to invest.
        horizon: Number of trading days.

    Returns:
        ``ComparisonResult`` with ranked outcomes.

    Raises:
        ValueError: If *rows* is empty.
    """
    if not rows:
        raise ValueError("At least one asset–model pair is required")

    registry = ModelRegistry()

    forecasts: Dict[str, Any] = {}
    dfs: Dict[str, pd.DataFrame] = {}
    diagnostics_verdicts: Dict[str, str] = {}
    max_vols: Dict[str, float] = {}

    for row in rows:
        ticker = row.ticker
        model_id = row.model_id

        # Data
        loader = AssetDataLoader(ticker=ticker)
        df = loader.load_data()
        eng = FeatureEngineering()
        df = eng.add_technical_indicators(df)
        feat_df = eng.select_features(df)
        feature_names = feat_df.columns.tolist()

        # Model metadata
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

        forecasts[ticker] = forecast
        dfs[ticker] = df

        ts = meta.get("training_summary", {})
        diagnostics_verdicts[ticker] = ts.get("diagnostics_verdict", "")

        if ticker in ASSET_CATALOG:
            max_vols[ticker] = ASSET_CATALOG[ticker].max_volatility
        else:
            max_vols[ticker] = 0.02

    comparator = PortfolioComparator(horizon_days=horizon)
    return comparator.compare(
        forecasts=forecasts,
        dfs=dfs,
        investment=investment,
        diagnostics_verdicts=diagnostics_verdicts,
        max_volatilities=max_vols,
    )
