"""Forecasting controller for Streamlit orchestration."""
from __future__ import annotations

from typing import Union

from gldpred.app import state
from gldpred.inference import TrajectoryPredictor
from gldpred.registry import ModelBundle
from gldpred.training import ModelTrainer


def generate_forecast(
    model: Union[ModelTrainer, ModelBundle],
    df,
    feature_names,
    seq_length: int,
    asset: str,
):
    """Generate and store a trajectory forecast.

    *model* can be either a ``ModelTrainer`` (after training) or a
    ``ModelBundle`` (loaded from registry).  Both expose the duck-typed
    interface required by ``TrajectoryPredictor``.
    """
    predictor = TrajectoryPredictor(model)
    forecast = predictor.predict_trajectory(
        df, feature_names, seq_length, asset,
    )
    state.put(state.KEY_FORECAST, forecast)
    return forecast
