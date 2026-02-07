"""Training pipeline for GLD price prediction models.

Supports three task modes:

* **regression** — predict continuous future returns (MSE loss).
* **classification** — predict buy/no-buy signal (BCE loss).
* **multitask** — shared backbone with both heads
  (weighted MSE + BCEWithLogits).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Valid task strings
TASKS = {"regression", "classification", "multitask"}


class ModelTrainer:
    """Train, evaluate, and persist PyTorch prediction models."""

    def __init__(
        self,
        model: nn.Module,
        task: str = "regression",
        device: Optional[torch.device] = None,
        *,
        w_reg: float = 1.0,
        w_cls: float = 1.0,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got '{task}'")

        self.model = model
        self.task = task
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Loss setup
        self.w_reg = w_reg
        self.w_cls = w_cls
        if task == "regression":
            self.criterion = nn.MSELoss()
        elif task == "classification":
            self.criterion = nn.BCELoss()
        else:  # multitask
            self._mse = nn.MSELoss()
            self._bce_logits = nn.BCEWithLogitsLoss()

        self.scaler = StandardScaler()
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        batch_size: int = 32,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare train/val DataLoaders from numpy arrays.

        For *multitask* mode ``y`` must have shape ``(N, 2)`` — column 0 is
        the regression target and column 1 is the classification label.
        """
        # Remove NaN rows
        if y.ndim == 1:
            valid = ~np.isnan(y)
        else:
            valid = ~np.isnan(y).any(axis=1)
        X, y = X[valid], y[valid]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Normalise features
        n_samples, n_steps, n_feat = X_train.shape
        X_train_s = self.scaler.fit_transform(
            X_train.reshape(-1, n_feat)
        ).reshape(n_samples, n_steps, n_feat)
        X_val_s = self.scaler.transform(
            X_val.reshape(-1, n_feat)
        ).reshape(-1, n_steps, n_feat)

        def _to_loader(X_arr: np.ndarray, y_arr: np.ndarray, shuffle: bool) -> DataLoader:
            tensors = [torch.FloatTensor(X_arr), torch.FloatTensor(y_arr)]
            return DataLoader(
                TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle
            )

        return _to_loader(X_train_s, y_train, True), _to_loader(X_val_s, y_val, False)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_loss(
        self,
        outputs: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.task == "multitask":
            reg_out, cls_logits = outputs
            reg_target = targets[:, 0]
            cls_target = targets[:, 1]
            loss_reg = self._mse(reg_out, reg_target)
            loss_cls = self._bce_logits(cls_logits, cls_target)
            return self.w_reg * loss_reg + self.w_cls * loss_cls
        return self.criterion(outputs, targets)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 0.001,
    ) -> Dict[str, List[float]]:
        """Run the training loop and return the loss history."""
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            # --- train ---
            self.model.train()
            running = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self._compute_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running += loss.item()
            train_loss = running / len(train_loader)

            # --- val ---
            self.model.eval()
            running = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self._compute_loss(outputs, batch_y)
                    running += loss.item()
            val_loss = running / len(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        return self.history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Run inference. Returns raw numpy predictions.

        For *multitask* returns ``(reg_preds, cls_probs)`` where
        ``cls_probs`` has sigmoid already applied.
        """
        self.model.eval()
        n_samples, n_steps, n_feat = X.shape
        X_s = self.scaler.transform(X.reshape(-1, n_feat)).reshape(n_samples, n_steps, n_feat)
        X_t = torch.FloatTensor(X_s).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_t)

        if self.task == "multitask":
            reg_out, cls_logits = outputs
            return reg_out.cpu().numpy(), torch.sigmoid(cls_logits).cpu().numpy()

        return outputs.cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler": self.scaler,
                "task": self.task,
                "w_reg": self.w_reg,
                "w_cls": self.w_cls,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.scaler = ckpt["scaler"]
        self.task = ckpt["task"]
        self.w_reg = ckpt.get("w_reg", 1.0)
        self.w_cls = ckpt.get("w_cls", 1.0)
        print(f"Model loaded from {filepath}")
