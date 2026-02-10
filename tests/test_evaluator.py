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

class TestQuantileCalibration:
    """Test that pinball loss produces calibrated quantile forecasts."""

    def test_pinball_produces_calibrated_quantiles(self):
        """Train a small model and verify quantile coverage is within tolerance."""
        from gldpred.models import TCNForecaster
        from gldpred.training import ModelTrainer
        
        # Generate synthetic data with known distribution
        np.random.seed(42)
        N = 500
        seq_len = 10
        input_size = 5
        forecast_steps = 5
        quantiles = (0.1, 0.5, 0.9)
        
        # Simple AR-like synthetic sequences
        X = np.random.randn(N, seq_len, input_size).astype(np.float32) * 0.01
        y = np.random.randn(N, forecast_steps).astype(np.float32) * 0.01
        
        # Train a tiny model
        model = TCNForecaster(
            input_size=input_size,
            hidden_size=16,
            num_layers=1,
            forecast_steps=forecast_steps,
            quantiles=quantiles,
        )
        trainer = ModelTrainer(model, quantiles=quantiles)
        train_loader, val_loader = trainer.prepare_data(X, y, test_size=0.2, batch_size=32)
        trainer.train(train_loader, val_loader, epochs=20, learning_rate=0.01)
        
        # Predict on validation set
        X_val = X[int(N * 0.8):]
        y_val = y[int(N * 0.8):]
        preds = trainer.predict(X_val)  # (N_val, K, 3)
        
        # Evaluate quantile calibration
        metrics = ModelEvaluator.evaluate_quantiles(y_val, preds, list(quantiles))
        
        # P10 should have ~10% of actuals below it (with tolerance)
        # P90 should have ~90% of actuals below it (with tolerance)
        # Tolerance is generous because small synthetic dataset
        assert 0.05 < metrics["q10_coverage"] < 0.20, f"P10 coverage {metrics['q10_coverage']} out of range"
        assert 0.40 < metrics["q50_coverage"] < 0.60, f"P50 coverage {metrics['q50_coverage']} out of range"
        assert 0.80 < metrics["q90_coverage"] < 0.95, f"P90 coverage {metrics['q90_coverage']} out of range"
        
        # Calibration error should be reasonable
        assert metrics["q10_cal_error"] < 0.15
        assert metrics["q90_cal_error"] < 0.15