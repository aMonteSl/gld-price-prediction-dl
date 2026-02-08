"""Model registry for persisting and managing trained models."""

from gldpred.registry.assignments import ModelAssignments
from gldpred.registry.store import ModelRegistry

__all__ = ["ModelAssignments", "ModelRegistry"]
