"""Model registry controller for Streamlit orchestration."""
from __future__ import annotations

from typing import Dict, List, Optional

from gldpred.registry import ModelAssignments, ModelRegistry


def list_models(asset: Optional[str] = None) -> List[dict]:
    """Return model metadata list for an asset or all assets."""
    registry = ModelRegistry()
    return registry.list_models(asset=asset)


def get_assignments() -> Dict[str, str]:
    """Return asset -> model_id assignments."""
    return ModelAssignments().get_all()


def update_label(model_id: str, new_label: str) -> None:
    """Update a model label."""
    ModelRegistry().update_model_label(model_id, new_label)


def set_primary(asset: str, model_id: str) -> None:
    """Assign a primary model for an asset."""
    ModelAssignments().assign(asset, model_id)


def unset_primary(asset: str) -> None:
    """Remove a primary model assignment for an asset."""
    ModelAssignments().unassign(asset)


def delete_model(model_id: str, asset: str, is_primary: bool) -> None:
    """Delete a model and unassign if primary."""
    assignments = ModelAssignments()
    if is_primary:
        assignments.unassign(asset)
    ModelRegistry().delete_model(model_id)


def bulk_delete(models: List[dict], asset: Optional[str]) -> int:
    """Delete models in bulk and unassign affected primaries."""
    assignments = ModelAssignments()
    all_assignments = assignments.get_all()
    deleted = ModelRegistry().delete_all_models(asset=asset, confirmed=True)
    for meta in models:
        asset_name = meta.get("asset", "")
        if all_assignments.get(asset_name) == meta.get("model_id"):
            assignments.unassign(asset_name)
    return deleted
