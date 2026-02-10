"""Training workflows for model fitting."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pandas as pd

from gldpred.features import FeatureEngineering
from gldpred.training import ModelTrainer


def build_sequences(
    df: pd.DataFrame,
    daily_returns: pd.Series,
    seq_length: int,
    forecast_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create feature/target sequences for training."""
    eng = FeatureEngineering()
    features_df = eng.select_features(df)
    return eng.create_sequences(
        features_df, daily_returns,
        seq_length=seq_length,
        forecast_steps=forecast_steps,
    )


def train_model(
    trainer: ModelTrainer,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    on_epoch: Callable,
    refit_scaler: bool = True,
    early_stopping: bool = False,
    patience: int = 5,
) -> dict:
    """Train a model and return loss history."""
    train_loader, val_loader = trainer.prepare_data(
        X,
        y,
        test_size=0.2,
        batch_size=batch_size,
        refit_scaler=refit_scaler,
    )
    return trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        on_epoch=on_epoch,
        early_stopping=early_stopping,
        patience=patience,
    )


def split_validation(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split out the validation set using the same temporal split."""
    split = int(len(X) * (1 - test_size))
    return X[split:], y[split:]
