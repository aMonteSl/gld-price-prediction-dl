"""Model loader controller — load saved models for inference.

Provides a single entry-point for all tabs (Forecast, Recommendation,
Evaluation) to obtain a ready-to-use model bundle from the registry.
"""
from __future__ import annotations

from typing import Optional

from gldpred.app import state
from gldpred.registry import ModelAssignments, ModelBundle, ModelRegistry


def get_active_model() -> Optional[ModelBundle]:
    """Return the currently active model bundle, or None."""
    return state.get(state.KEY_ACTIVE_MODEL)


def load_model_from_registry(model_id: str) -> ModelBundle:
    """Load a model bundle from registry and set it as active.

    Returns:
        The loaded ModelBundle.

    Raises:
        FileNotFoundError: If model_id is not in the registry.
        ValueError: If the architecture is unknown.
    """
    registry = ModelRegistry()
    bundle = registry.load_bundle(model_id)
    state.put(state.KEY_ACTIVE_MODEL, bundle)
    state.put(state.KEY_ACTIVE_MODEL_ID, model_id)
    return bundle


def activate_after_training() -> None:
    """After training, set the just-trained model as active.

    Reads KEY_TRAINER and KEY_LAST_MODEL_ID from state and creates
    a ModelBundle wrapping the in-memory trainer so downstream tabs
    work immediately after training without a disk round-trip.
    """
    trainer = state.get(state.KEY_TRAINER)
    model_id = state.get(state.KEY_LAST_MODEL_ID)
    if trainer is None or model_id is None:
        return

    # Load the metadata from registry to get all the saved info
    registry = ModelRegistry()
    models = registry.list_models()
    meta = next((m for m in models if m["model_id"] == model_id), {})

    cfg = meta.get("config", {})
    bundle = ModelBundle(
        model=trainer.model,
        scaler=trainer.scaler,
        metadata=meta,
        model_id=model_id,
        label=meta.get("label", model_id),
        asset=meta.get("asset", state.get(state.KEY_ASSET, "GLD")),
        architecture=cfg.get("architecture", "TCN"),
        feature_names=meta.get("feature_names", state.get(state.KEY_FEATURE_NAMES, [])),
        quantiles_tuple=trainer.quantiles_tuple,
        config=cfg,
    )
    state.put(state.KEY_ACTIVE_MODEL, bundle)
    state.put(state.KEY_ACTIVE_MODEL_ID, model_id)


def auto_select_model(asset: str) -> Optional[ModelBundle]:
    """Auto-select a model for the given asset.

    Priority:
      1. Current active model (if it matches the asset)
      2. Primary model assignment for the asset
      3. Most recently created model for the asset

    Returns:
        ModelBundle if a model was found and loaded, None otherwise.
    """
    # 1. Check if current active model matches
    current = state.get(state.KEY_ACTIVE_MODEL)
    if current is not None and current.asset == asset:
        return current

    # 2. Try primary assignment
    assignments = ModelAssignments()
    primary_id = assignments.get(asset)
    if primary_id:
        try:
            return load_model_from_registry(primary_id)
        except (FileNotFoundError, ValueError):
            pass  # Primary model was deleted; fall through

    # 3. Most recent model for this asset
    registry = ModelRegistry()
    models = registry.list_models(asset=asset)
    if models:
        # list_models returns sorted by directory name (timestamp-based)
        latest = models[-1]
        try:
            return load_model_from_registry(latest["model_id"])
        except (FileNotFoundError, ValueError):
            pass

    return None


def list_asset_models(asset: str) -> list[dict]:
    """Return metadata for all models belonging to the given asset."""
    return ModelRegistry().list_models(asset=asset)
