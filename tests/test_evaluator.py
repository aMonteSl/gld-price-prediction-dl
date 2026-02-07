"""Tests for evaluator — regression, classification, and multi-task metrics."""
from __future__ import annotations

import numpy as np
import pytest

from gldpred.evaluation import ModelEvaluator


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = ModelEvaluator.evaluate_regression(y, y)
        assert metrics["mse"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_keys_present(self):
        y_true = np.random.randn(50)
        y_pred = y_true + np.random.randn(50) * 0.1
        metrics = ModelEvaluator.evaluate_regression(y_true, y_pred)
        for key in ("mse", "rmse", "mae", "r2"):
            assert key in metrics

    def test_rmse_equals_sqrt_mse(self):
        y_true = np.random.randn(50)
        y_pred = np.random.randn(50)
        metrics = ModelEvaluator.evaluate_regression(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]), abs=1e-9)


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y = np.array([1, 0, 1, 0, 1], dtype=float)
        preds = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        metrics = ModelEvaluator.evaluate_classification(y, preds)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_keys_present(self):
        y = np.array([1, 0, 1, 0])
        preds = np.array([0.6, 0.4, 0.7, 0.3])
        metrics = ModelEvaluator.evaluate_classification(y, preds)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in metrics

    def test_confusion_matrix(self):
        y = np.array([1, 0, 1, 0], dtype=float)
        preds = np.array([0.9, 0.1, 0.8, 0.2])
        metrics = ModelEvaluator.evaluate_classification(y, preds)
        assert "confusion_matrix" in metrics
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2


class TestMultitaskMetrics:
    def test_keys_have_prefixes(self):
        y_reg = np.random.randn(50)
        y_cls = np.random.randint(0, 2, 50).astype(float)
        p_reg = y_reg + 0.01
        p_cls = np.random.rand(50)
        metrics = ModelEvaluator.evaluate_multitask(y_reg, p_reg, y_cls, p_cls)
        assert "reg_mse" in metrics
        assert "cls_accuracy" in metrics
        assert "threshold" in metrics

    def test_threshold_passed_through(self):
        y_reg = np.random.randn(20)
        y_cls = np.random.randint(0, 2, 20).astype(float)
        metrics = ModelEvaluator.evaluate_multitask(
            y_reg, y_reg, y_cls, np.random.rand(20), cls_threshold=0.7
        )
        assert metrics["threshold"] == 0.7
