"""Tests for trajectory and quantile evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from gldpred.evaluation import ModelEvaluator
from conftest import FORECAST_STEPS, QUANTILES


class TestTrajectoryMetrics:
    def test_perfect_prediction(self):
        """Zero error when predictions match targets."""
        y = np.random.randn(20, FORECAST_STEPS).astype(np.float32) * 0.01
        metrics = ModelEvaluator.evaluate_trajectory(y, y)
        assert metrics["mse"] == pytest.approx(0.0, abs=1e-7)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-7)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-7)
        assert metrics["directional_accuracy"] == pytest.approx(1.0, abs=1e-7)

    def test_nonzero_error(self):
        """Metrics are positive for imperfect predictions."""
        y_true = np.random.randn(30, FORECAST_STEPS).astype(np.float32) * 0.01
        y_pred = y_true + np.random.randn(30, FORECAST_STEPS).astype(np.float32) * 0.005
        metrics = ModelEvaluator.evaluate_trajectory(y_true, y_pred)
        assert metrics["mse"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert 0 <= metrics["directional_accuracy"] <= 1

    def test_per_step_keys(self):
        """Result contains per-step breakdowns."""
        y = np.random.randn(20, FORECAST_STEPS).astype(np.float32) * 0.01
        p = y + 0.001
        metrics = ModelEvaluator.evaluate_trajectory(y, p)
        assert len(metrics["mse_per_step"]) == FORECAST_STEPS
        assert len(metrics["mae_per_step"]) == FORECAST_STEPS
        assert len(metrics["dir_acc_per_step"]) == FORECAST_STEPS

    def test_nan_filtering(self):
        """Rows with NaN targets are excluded."""
        y_true = np.random.randn(10, FORECAST_STEPS).astype(np.float32)
        y_true[0, :] = np.nan
        y_pred = y_true.copy()
        y_pred[0, :] = 0
        metrics = ModelEvaluator.evaluate_trajectory(y_true, y_pred)
        # Should compute on 9 rows, and perfect match for those
        assert metrics["mse"] == pytest.approx(0.0, abs=1e-7)


class TestQuantileMetrics:
    def test_coverage_keys(self):
        """Coverage keys present for each quantile."""
        y_true = np.random.randn(30, FORECAST_STEPS) * 0.01
        y_pred = np.random.randn(30, FORECAST_STEPS, len(QUANTILES)) * 0.01
        metrics = ModelEvaluator.evaluate_quantiles(y_true, y_pred, list(QUANTILES))
        assert "q10_coverage" in metrics
        assert "q50_coverage" in metrics
        assert "q90_coverage" in metrics

    def test_interval_width(self):
        """Interval width is non-negative."""
        y_true = np.random.randn(30, FORECAST_STEPS) * 0.01
        # Ensure P90 > P10
        y_pred = np.zeros((30, FORECAST_STEPS, 3))
        y_pred[:, :, 0] = -0.01  # P10
        y_pred[:, :, 1] = 0.0    # P50
        y_pred[:, :, 2] = 0.01   # P90
        metrics = ModelEvaluator.evaluate_quantiles(y_true, y_pred, list(QUANTILES))
        assert metrics["mean_interval_width"] >= 0

    def test_calibration_error(self):
        """Calibration error is between 0 and 1."""
        y_true = np.random.randn(50, FORECAST_STEPS) * 0.01
        y_pred = np.random.randn(50, FORECAST_STEPS, 3) * 0.01
        metrics = ModelEvaluator.evaluate_quantiles(y_true, y_pred, list(QUANTILES))
        for q in [10, 50, 90]:
            assert 0 <= metrics[f"q{q}_cal_error"] <= 1.0
