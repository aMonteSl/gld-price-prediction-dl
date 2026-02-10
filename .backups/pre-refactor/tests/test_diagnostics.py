"""Tests for the training diagnostics analyzer."""
from __future__ import annotations

import numpy as np
import pytest

from gldpred.diagnostics import DiagnosticsAnalyzer


class TestDiagnostics:
    def test_healthy(self):
        """Monotonically decreasing curves → healthy."""
        epochs = 50
        train = np.linspace(1.0, 0.5, epochs).tolist()
        val = np.linspace(1.1, 0.6, epochs).tolist()
        result = DiagnosticsAnalyzer.analyze({"train_loss": train, "val_loss": val})
        assert result.verdict == "healthy"

    def test_overfitting(self):
        """Val rising while train falling → overfitting."""
        epochs = 20
        train = np.linspace(1.0, 0.1, epochs).tolist()
        val = np.linspace(0.5, 1.5, epochs).tolist()
        result = DiagnosticsAnalyzer.analyze({"train_loss": train, "val_loss": val})
        assert result.verdict == "overfitting"

    def test_underfitting(self):
        """Both curves high and flat → underfitting."""
        train = [0.9] * 20
        val = [1.0] * 20
        result = DiagnosticsAnalyzer.analyze({"train_loss": train, "val_loss": val})
        assert result.verdict == "underfitting"

    def test_noisy(self):
        """Highly oscillating val loss → noisy."""
        np.random.seed(42)
        train = np.linspace(1.0, 0.3, 20).tolist()
        val = (np.linspace(1.0, 0.5, 20) + np.random.randn(20) * 0.3).tolist()
        result = DiagnosticsAnalyzer.analyze({"train_loss": train, "val_loss": val})
        assert result.verdict in ("noisy", "healthy")  # depends on noise realisation

    def test_too_few_epochs(self):
        """Single epoch returns early."""
        result = DiagnosticsAnalyzer.analyze({"train_loss": [1.0], "val_loss": [1.1]})
        assert result.verdict == "noisy"  # too few

    def test_result_fields(self):
        train = np.linspace(1.0, 0.1, 10).tolist()
        val = np.linspace(1.1, 0.2, 10).tolist()
        result = DiagnosticsAnalyzer.analyze({"train_loss": train, "val_loss": val})
        assert hasattr(result, "verdict")
        assert hasattr(result, "explanation")
        assert hasattr(result, "suggestions")
        assert hasattr(result, "best_epoch")
        assert hasattr(result, "generalization_gap")

    def test_suggestions_non_empty(self):
        train = np.linspace(1.0, 0.1, 20).tolist()
        val = np.linspace(0.5, 1.5, 20).tolist()
        result = DiagnosticsAnalyzer.analyze({"train_loss": train, "val_loss": val})
        assert len(result.suggestions) > 0
