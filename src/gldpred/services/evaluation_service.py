"""Evaluation workflows for trained models."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from gldpred.evaluation import ModelEvaluator
from gldpred.services.training_service import split_validation
from gldpred.training import ModelTrainer


def evaluate_on_validation(
    trainer: ModelTrainer,
    X: np.ndarray,
    y: np.ndarray,
    quantiles: List[float],
) -> Tuple[dict, dict]:
    """Compute trajectory and quantile metrics on the validation split."""
    X_val, y_val = split_validation(X, y, test_size=0.2)
    y_pred = trainer.predict(X_val)

    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    y_pred_median = y_pred[:, :, median_idx]

    traj_metrics = ModelEvaluator.evaluate_trajectory(y_val, y_pred_median)
    quant_metrics = ModelEvaluator.evaluate_quantiles(y_val, y_pred, quantiles)
    return traj_metrics, quant_metrics
