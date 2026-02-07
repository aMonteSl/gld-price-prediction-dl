"""
Inference module for generating predictions and buy/no-buy signals.

Wraps the ModelTrainer.predict() functionality with additional
signal-generation logic for use in application code and the Streamlit GUI.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np


class Predictor:
    """Generate predictions and trading signals from a trained model."""

    def __init__(self, trainer) -> None:
        """
        Initialize the predictor.

        Args:
            trainer: A trained ModelTrainer instance (with fitted scaler).
        """
        self.trainer = trainer

    @property
    def task(self) -> str:
        return self.trainer.task

    def predict(
        self, X: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run model inference on input sequences.

        Args:
            X: Input sequences as numpy array (n_samples, seq_length, n_features).

        Returns:
            For regression/classification: numpy array of raw predictions.
            For multitask: tuple (reg_preds, cls_probs).
        """
        return self.trainer.predict(X)

    def predict_signals(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary buy/no-buy signals.

        For classification: thresholds the raw probabilities.
        For multitask: thresholds the classification head output.

        Args:
            X: Input sequences.
            threshold: Probability threshold for the buy signal.

        Returns:
            Numpy array of binary signals (1 = buy, 0 = no-buy).
        """
        predictions = self.predict(X)
        if self.task == "multitask":
            _, cls_probs = predictions
            return (cls_probs > threshold).astype(int)
        return (predictions > threshold).astype(int)

    def predict_returns(self, X: np.ndarray) -> np.ndarray:
        """
        Predict future returns.

        For regression: alias for predict().
        For multitask: returns the regression head output.

        Args:
            X: Input sequences.

        Returns:
            Numpy array of predicted returns.
        """
        predictions = self.predict(X)
        if self.task == "multitask":
            reg_preds, _ = predictions
            return reg_preds
        return predictions

    def predict_implied_prices(
        self, X: np.ndarray, current_prices: np.ndarray
    ) -> np.ndarray:
        """
        Compute implied future prices from predicted returns.

        Args:
            X: Input sequences.
            current_prices: Array of current prices aligned with predictions.

        Returns:
            Numpy array of implied prices.
        """
        predicted_returns = self.predict_returns(X)
        return np.array(current_prices) * (1 + predicted_returns)
