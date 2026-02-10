"""Decision policy module — transparent scoring pipeline.

Provides ``DecisionPolicy``, a composable scoring system that wraps
the existing ``DecisionEngine`` and adds:

- Explicit ``ScoreFactor`` breakdown (name, weight, value, rationale)
- ``PolicyResult`` with full transparency for the UI
- Configurable factor weights
"""
from gldpred.core.policy.scoring import (
    DecisionPolicy,
    PolicyResult,
    ScoreFactor,
)

__all__ = [
    "DecisionPolicy",
    "PolicyResult",
    "ScoreFactor",
]
