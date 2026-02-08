"""Model registry: save, load, list, and manage trained model artifacts.

Each saved model gets a directory under ``data/model_registry/<model_id>/``
containing weights, scaler, and metadata JSON.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import joblib
import numpy as np
import torch
import torch.nn as nn

DEFAULT_REGISTRY_DIR = Path("data") / "model_registry"
MAX_LABEL_LENGTH = 60


class ModelRegistry:
    """Persist and retrieve trained models with full metadata."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_REGISTRY_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_model(
        self,
        model: nn.Module,
        scaler: Any,
        config: Dict[str, Any],
        feature_names: List[str],
        training_summary: Dict[str, Any],
        evaluation_summary: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
    ) -> str:
        """Save a trained model with all artifacts.

        Args:
            model: Trained PyTorch model.
            scaler: Fitted StandardScaler.
            config: Model configuration dict.
            feature_names: List of feature column names.
            training_summary: Training history and metrics.
            evaluation_summary: Optional evaluation metrics.
            label: Optional custom name for the model (max 60 chars).
                   If omitted, auto-generated from timestamp.

        Returns:
            model_id string.
        """
        # Validate and sanitize label
        if label:
            label = _validate_label(label)
        else:
            # Auto-generate label from asset and architecture
            asset = config.get("asset", "GLD")
            arch = config.get("architecture", "TCN")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            label = f"{asset}_{arch}_{timestamp}"

        # Generate unique model_id (filesystem-safe)
        model_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        model_dir = self.base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), model_dir / "weights.pth")
        joblib.dump(scaler, model_dir / "scaler.joblib")

        metadata = {
            "model_id": model_id,
            "label": label,
            "asset": config.get("asset", "GLD"),
            "architecture": config.get("architecture", "TCN"),
            "created_at": datetime.now().isoformat(),
            "config": _serializable(config),
            "feature_names": feature_names,
            "training_summary": _serializable(training_summary),
            "evaluation_summary": _serializable(evaluation_summary or {}),
        }
        (model_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )
        return model_id

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load_model(
        self,
        model_id: str,
        model_class: Type[nn.Module],
        **model_kwargs: Any,
    ) -> tuple[nn.Module, Any, Dict[str, Any]]:
        """Load model weights, scaler, and metadata.

        Returns:
            (model, scaler, metadata_dict).
        """
        model_dir = self.base_dir / model_id
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model '{model_id}' not found in registry"
            )

        metadata = json.loads((model_dir / "metadata.json").read_text())

        model = model_class(**model_kwargs)
        model.load_state_dict(
            torch.load(
                model_dir / "weights.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        scaler = joblib.load(model_dir / "scaler.joblib")

        return model, scaler, metadata

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------
    def list_models(
        self,
        asset: Optional[str] = None,
        architecture: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return metadata for all models matching optional filters."""
        models: List[Dict[str, Any]] = []
        if not self.base_dir.exists():
            return models
        for entry in sorted(self.base_dir.iterdir()):
            meta_file = entry / "metadata.json"
            if not entry.is_dir() or not meta_file.exists():
                continue
            meta = json.loads(meta_file.read_text())
            if asset and meta.get("asset") != asset:
                continue
            if architecture and meta.get("architecture") != architecture:
                continue
            models.append(meta)
        return models

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete_model(self, model_id: str) -> None:
        """Delete a single model by ID.

        Args:
            model_id: The unique model identifier.

        Raises:
            FileNotFoundError: If the model does not exist.
        """
        model_dir = self.base_dir / model_id
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model '{model_id}' not found in registry"
            )
        shutil.rmtree(model_dir)

    def delete_all_models(
        self, asset: Optional[str] = None, confirmed: bool = False
    ) -> int:
        """Delete all models matching optional asset filter.

        Args:
            asset: If provided, only delete models for this asset.
            confirmed: Must be True to actually delete (safety check).

        Returns:
            Number of models deleted.

        Raises:
            ValueError: If confirmed is not True.
        """
        if not confirmed:
            raise ValueError(
                "Must set confirmed=True to delete all models"
            )

        models = self.list_models(asset=asset)
        count = 0
        for meta in models:
            model_id = meta["model_id"]
            model_dir = self.base_dir / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)
                count += 1
        return count


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _validate_label(label: str) -> str:
    """Validate and sanitize a custom model label.

    Args:
        label: User-provided label.

    Returns:
        Sanitized label string.

    Raises:
        ValueError: If label is empty or too long.
    """
    label = label.strip()
    if not label:
        raise ValueError("Model label cannot be empty")
    if len(label) > MAX_LABEL_LENGTH:
        raise ValueError(
            f"Model label too long (max {MAX_LABEL_LENGTH} chars)"
        )
    # Preserve original label exactly (don't sanitize for filesystem)
    # Only model_id needs to be filesystem-safe; label is for display only
    return label


def _serializable(obj: Any) -> Any:
    """Recursively convert numpy/torch types for JSON."""
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
