"""Evaluation metrics for regression, classification, and multi-task models."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class ModelEvaluator:
    """Compute and display model performance metrics."""

    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Return MSE, RMSE, MAE, R² for a regression task."""
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        mse = float(mean_squared_error(yt, yp))
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
        }

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Return accuracy, precision, recall, F1, confusion-matrix."""
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        yt, yp = y_true[mask], y_pred[mask]
        yb = (yp > threshold).astype(int)
        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(yt, yb)),
            "precision": float(precision_score(yt, yb, zero_division=0)),
            "recall": float(recall_score(yt, yb, zero_division=0)),
            "f1": float(f1_score(yt, yb, zero_division=0)),
        }
        cm = confusion_matrix(yt, yb)
        if cm.shape == (2, 2):
            metrics["confusion_matrix"] = cm.tolist()
        return metrics

    @staticmethod
    def evaluate_multitask(
        y_true_reg: np.ndarray,
        y_pred_reg: np.ndarray,
        y_true_cls: np.ndarray,
        y_pred_cls: np.ndarray,
        cls_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Evaluate a multi-task model (regression + classification).

        Returns a single dict with ``reg_*`` and ``cls_*`` prefixed keys
        plus ``threshold``.
        """
        reg = ModelEvaluator.evaluate_regression(y_true_reg, y_pred_reg)
        cls = ModelEvaluator.evaluate_classification(
            y_true_cls, y_pred_cls, threshold=cls_threshold
        )
        combined: Dict[str, Any] = {"threshold": cls_threshold}
        for k, v in reg.items():
            combined[f"reg_{k}"] = v
        for k, v in cls.items():
            combined[f"cls_{k}"] = v
        return combined

    @staticmethod
    def print_metrics(metrics: Dict[str, Any], task: str = "regression") -> None:
        """Pretty-print metrics to stdout."""
        print(f"\n{task.upper()} METRICS:")
        print("=" * 50)
        if task == "regression":
            for k in ("mse", "rmse", "mae", "r2"):
                print(f"  {k.upper():6s}: {metrics[k]:.6f}")
        elif task == "classification":
            for k in ("accuracy", "precision", "recall", "f1"):
                print(f"  {k.capitalize():10s}: {metrics[k]:.4f}")
            if "confusion_matrix" in metrics:
                print(f"  CM: {metrics['confusion_matrix']}")
        elif task == "multitask":
            print("  --- Regression ---")
            for k in ("reg_mse", "reg_rmse", "reg_mae", "reg_r2"):
                print(f"    {k:12s}: {metrics[k]:.6f}")
            print("  --- Classification ---")
            for k in ("cls_accuracy", "cls_precision", "cls_recall", "cls_f1"):
                print(f"    {k:16s}: {metrics[k]:.4f}")
            if "cls_confusion_matrix" in metrics:
                print(f"    CM: {metrics['cls_confusion_matrix']}")
        print("=" * 50)
