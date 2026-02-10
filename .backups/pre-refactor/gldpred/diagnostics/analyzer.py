"""Analyse training/validation loss curves and produce diagnostics.

Main entry point: :func:`DiagnosticsAnalyzer.analyze`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class DiagnosticsResult:
    """Structured diagnostics output."""

    verdict: str = "healthy"
    explanation: str = ""
    suggestions: List[str] = field(default_factory=list)
    best_epoch: int = 0
    generalization_gap: float = 0.0
    train_slope: float = 0.0
    val_slope: float = 0.0


class DiagnosticsAnalyzer:
    """Stateless analyser — call :meth:`analyze` with the loss history."""

    @staticmethod
    def analyze(
        history: Dict[str, List[float]],
        *,
        tail_fraction: float = 0.25,
    ) -> DiagnosticsResult:
        """Analyse train & val loss curves and return a verdict."""
        train = np.asarray(history["train_loss"], dtype=float)
        val = np.asarray(history["val_loss"], dtype=float)
        n = len(train)

        if n < 2:
            return DiagnosticsResult(
                verdict="noisy",
                explanation="Too few epochs to diagnose.",
                suggestions=["Train for more epochs."],
                best_epoch=0,
            )

        best_epoch = int(np.argmin(val))
        gap = float(val[-1] - train[-1])

        tail = max(int(n * tail_fraction), 2)
        t_slope = float(_slope(train[-tail:]))
        v_slope = float(_slope(val[-tail:]))

        result = DiagnosticsResult(
            best_epoch=best_epoch,
            generalization_gap=gap,
            train_slope=t_slope,
            val_slope=v_slope,
        )

        if v_slope > 0 and t_slope < -1e-7:
            result.verdict = "overfitting"
            result.explanation = (
                "Validation loss is rising while training loss keeps "
                "decreasing — the model is memorising the training data."
            )
            result.suggestions = [
                f"Stop training earlier (best epoch ≈ {best_epoch + 1}).",
                "Reduce hidden size or number of layers.",
                "Increase dropout.",
                "Use more training data (longer date range).",
            ]
        elif (
            t_slope > -1e-7
            and v_slope > -1e-7
            and train[-1] > 0.5 * train[0]
        ):
            result.verdict = "underfitting"
            result.explanation = (
                "Both losses remain high and flat — the model lacks "
                "capacity or the learning rate is too low."
            )
            result.suggestions = [
                "Increase hidden size or number of layers.",
                "Train for more epochs.",
                "Raise the learning rate slightly.",
                "Verify data quality and feature engineering.",
            ]
        elif _is_noisy(val, tail):
            result.verdict = "noisy"
            result.explanation = (
                "The validation loss oscillates significantly — "
                "training is unstable."
            )
            result.suggestions = [
                "Lower the learning rate.",
                "Increase batch size for smoother gradients.",
                "Increase sequence length for more context.",
            ]
        else:
            result.verdict = "healthy"
            result.explanation = (
                "Both loss curves are decreasing with a stable gap — "
                "training converged normally."
            )
            result.suggestions = [
                "Current settings look good.",
                "You may try a few more epochs to squeeze out marginal gains.",
            ]

        return result


def _slope(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def _is_noisy(arr: np.ndarray, tail: int, *, threshold: float = 0.15) -> bool:
    segment = arr[-tail:]
    mean = segment.mean()
    if abs(mean) < 1e-12:
        return False
    cv = segment.std() / abs(mean)
    return cv > threshold
