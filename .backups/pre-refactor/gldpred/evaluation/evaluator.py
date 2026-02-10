"""Evaluation metrics for multi-step trajectory and quantile forecasts."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class ModelEvaluator:
    """Compute trajectory accuracy and quantile calibration metrics."""

    # ------------------------------------------------------------------
    # Trajectory (median) metrics
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate_trajectory(
        y_true: np.ndarray,
        y_pred_median: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate median trajectory predictions.

        Args:
            y_true: (N, K) actual future returns.
            y_pred_median: (N, K) P50 predicted returns.

        Returns:
            Dict with MSE, RMSE, MAE, directional accuracy.
        """
        mask = ~np.isnan(y_true).any(axis=1)
        yt, yp = y_true[mask], y_pred_median[mask]

        mse_per_step = np.mean((yt - yp) ** 2, axis=0)
        mae_per_step = np.mean(np.abs(yt - yp), axis=0)
        dir_per_step = np.mean(np.sign(yt) == np.sign(yp), axis=0)

        overall_mse = float(np.mean(mse_per_step))
        return {
            "mse": overall_mse,
            "rmse": float(np.sqrt(overall_mse)),
            "mae": float(np.mean(mae_per_step)),
            "directional_accuracy": float(np.mean(dir_per_step)),
            "mse_per_step": mse_per_step.tolist(),
            "mae_per_step": mae_per_step.tolist(),
            "dir_acc_per_step": dir_per_step.tolist(),
        }

    # ------------------------------------------------------------------
    # Quantile calibration
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate_quantiles(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantiles: List[float],
    ) -> Dict[str, float]:
        """Evaluate quantile calibration.

        Args:
            y_true: (N, K) actual returns.
            y_pred: (N, K, Q) predicted quantiles.
            quantiles: Q quantile levels (e.g. [0.1, 0.5, 0.9]).

        Returns:
            Dict with per-quantile coverage and calibration error.
        """
        mask = ~np.isnan(y_true).any(axis=1)
        yt, yp = y_true[mask], y_pred[mask]

        result: Dict[str, float] = {}
        for qi, q in enumerate(quantiles):
            frac = float(np.mean(yt < yp[:, :, qi]))
            result[f"q{int(q * 100)}_coverage"] = frac
            result[f"q{int(q * 100)}_cal_error"] = abs(frac - q)

        if len(quantiles) >= 2:
            width = yp[:, :, -1] - yp[:, :, 0]
            result["mean_interval_width"] = float(np.mean(width))

        return result

    # ------------------------------------------------------------------
    # Print helper
    # ------------------------------------------------------------------
    @staticmethod
    def print_metrics(metrics: Dict[str, Any]) -> None:
        """Pretty-print a metrics dictionary."""
        print("\nMETRICS:")
        print("=" * 50)
        for k, v in metrics.items():
            if isinstance(v, list):
                print(f"  {k}: [{', '.join(f'{x:.6f}' for x in v)}]")
            elif isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print("=" * 50)
