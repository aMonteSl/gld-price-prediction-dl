"""Training pipeline with pinball loss for multi-step quantile forecasting.

Supports training from scratch and fine-tuning loaded models.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------

def pinball_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """Quantile (pinball) loss.

    Args:
        y_pred: (B, K, Q) predicted quantile values.
        y_true: (B, K) observed values.
        quantiles: (Q,) quantile levels, e.g. [0.1, 0.5, 0.9].

    Returns:
        Scalar loss.
    """
    errors = y_true.unsqueeze(-1) - y_pred  # (B, K, Q)
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
    return loss.mean()


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class ModelTrainer:
    """Train and evaluate multi-step quantile forecasting models."""

    def __init__(
        self,
        model: nn.Module,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.quantiles_tuple = quantiles
        self._q = torch.tensor(quantiles, dtype=torch.float32)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.scaler = StandardScaler()
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def quantiles(self) -> Tuple[float, ...]:
        """Quantile levels used by this trainer."""
        return self.quantiles_tuple

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        batch_size: int = 32,
        *,
        refit_scaler: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare train/val DataLoaders with temporal split.

        Args:
            X: (N, seq_len, F) input sequences.
            y: (N, K) multi-step return targets.
            refit_scaler: if False, the existing scaler is used (fine-tune).
        """
        # Drop NaN rows
        valid = (
            ~np.isnan(y).any(axis=1)
            & ~np.isnan(X.reshape(X.shape[0], -1)).any(axis=1)
        )
        X, y = X[valid], y[valid]

        # Temporal split — no shuffling
        split = int(len(X) * (1 - test_size))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Scale features
        n_tr, s, f = X_train.shape
        if refit_scaler:
            X_train_s = self.scaler.fit_transform(
                X_train.reshape(-1, f)
            ).reshape(n_tr, s, f)
        else:
            X_train_s = self.scaler.transform(
                X_train.reshape(-1, f)
            ).reshape(n_tr, s, f)

        n_va = X_val.shape[0]
        X_val_s = self.scaler.transform(
            X_val.reshape(-1, f)
        ).reshape(n_va, s, f)

        def _loader(xa, ya, shuffle):
            return DataLoader(
                TensorDataset(
                    torch.FloatTensor(xa),
                    torch.FloatTensor(ya),
                ),
                batch_size=batch_size,
                shuffle=shuffle,
            )

        return _loader(X_train_s, y_train, True), _loader(X_val_s, y_val, False)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 0.001,
        on_epoch: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """Run training loop and return loss history.

        Args:
            on_epoch: optional callback ``(epoch_0based, total, history)``.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        q = self._q.to(self.device)

        for epoch in range(epochs):
            # --- train ---
            self.model.train()
            running = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = pinball_loss(pred, by, q)
                loss.backward()
                optimizer.step()
                running += loss.item()
            self.history["train_loss"].append(running / len(train_loader))

            # --- val ---
            self.model.eval()
            running = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    pred = self.model(bx)
                    loss = pinball_loss(pred, by, q)
                    running += loss.item()
            self.history["val_loss"].append(running / len(val_loader))

            if on_epoch:
                on_epoch(epoch, epochs, self.history)
            elif (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train: {self.history['train_loss'][-1]:.6f}  "
                    f"Val: {self.history['val_loss'][-1]:.6f}"
                )

        return self.history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference. Returns (N, K, Q) numpy array."""
        self.model.eval()
        n, s, f = X.shape
        X_s = self.scaler.transform(X.reshape(-1, f)).reshape(n, s, f)
        X_t = torch.FloatTensor(X_s).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
        return out.cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, filepath: str) -> None:
        """Save model weights, scaler, and training state."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler": self.scaler,
                "quantiles": self.quantiles_tuple,
                "history": self.history,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Load model weights, scaler, and training history."""
        ckpt = torch.load(
            filepath, map_location=self.device, weights_only=False,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.scaler = ckpt["scaler"]
        self.quantiles_tuple = ckpt.get("quantiles", self.quantiles_tuple)
        self._q = torch.tensor(self.quantiles_tuple, dtype=torch.float32)
        prev = ckpt.get("history")
        if prev:
            self.history = prev
