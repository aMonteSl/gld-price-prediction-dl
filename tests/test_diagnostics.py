"""Tests for DiagnosticsAnalyzer — synthetic loss curves."""
from __future__ import annotations

import numpy as np
import pytest

from gldpred.diagnostics import DiagnosticsAnalyzer


def _make_history(train: list[float], val: list[float]) -> dict:
    return {"train_loss": train, "val_loss": val}


class TestDiagnosticsVerdict:
    """Feed synthetic loss curves and check verdicts."""

    def test_healthy(self):
        """Both curves decrease smoothly."""
        n = 50
        train = [1.0 - 0.015 * i for i in range(n)]
        val = [1.1 - 0.014 * i for i in range(n)]
        result = DiagnosticsAnalyzer.analyze(_make_history(train, val))
        assert result.verdict == "healthy"

    def test_overfitting(self):
        """Train loss keeps decreasing while val loss rises."""
        n = 50
        train = [1.0 - 0.02 * i for i in range(n)]
        val = [1.0 - 0.01 * i for i in range(25)] + [0.75 + 0.02 * i for i in range(25)]
        result = DiagnosticsAnalyzer.analyze(_make_history(train, val))
        assert result.verdict == "overfitting"
        assert len(result.suggestions) > 0

    def test_underfitting(self):
        """Both curves high and flat (barely improving)."""
        n = 50
        # Perfectly flat at a high value — clearly underfitting
        train = [0.98] * n
        val = [1.00] * n
        result = DiagnosticsAnalyzer.analyze(_make_history(train, val))
        assert result.verdict == "underfitting"

    def test_noisy(self):
        """Validation curve oscillates wildly."""
        n = 50
        np.random.seed(0)
        train = [1.0 - 0.01 * i for i in range(n)]
        val = [1.0 + 0.5 * np.sin(i * 0.8) for i in range(n)]
        result = DiagnosticsAnalyzer.analyze(_make_history(train, val))
        assert result.verdict in ("noisy", "overfitting")  # oscillation can look like overfit

    def test_too_few_epochs(self):
        """With < 2 epochs, should return noisy / fallback."""
        result = DiagnosticsAnalyzer.analyze(_make_history([0.5], [0.6]))
        assert result.verdict == "noisy"
        assert "Too few" in result.explanation


class TestDiagnosticsFields:
    """Verify result fields are populated correctly."""

    def test_best_epoch(self):
        val = [1.0, 0.9, 0.8, 0.85, 0.9]
        result = DiagnosticsAnalyzer.analyze(
            _make_history([1.0, 0.9, 0.8, 0.7, 0.6], val)
        )
        assert result.best_epoch == 2  # 0-indexed

    def test_generalization_gap(self):
        train = [1.0, 0.5, 0.3]
        val = [1.1, 0.6, 0.5]
        result = DiagnosticsAnalyzer.analyze(_make_history(train, val))
        expected_gap = val[-1] - train[-1]
        assert abs(result.generalization_gap - expected_gap) < 1e-6

    def test_slopes_are_floats(self):
        result = DiagnosticsAnalyzer.analyze(
            _make_history([1.0, 0.9, 0.8], [1.1, 1.0, 0.95])
        )
        assert isinstance(result.train_slope, float)
        assert isinstance(result.val_slope, float)
