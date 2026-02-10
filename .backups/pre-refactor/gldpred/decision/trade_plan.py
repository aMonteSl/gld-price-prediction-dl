"""Trade-plan engine — **v2** re-exports from ``action_planner`` and
``scenario_analyzer``.

This module exists only as a backward-compatible bridge.  All logic now
lives in :pymod:`gldpred.decision.action_planner` and
:pymod:`gldpred.decision.scenario_analyzer`.
"""
from gldpred.decision.action_planner import (       # noqa: F401
    ActionPlan,
    DayRecommendation,
    DecisionRationale,
    EntryWindow,
    ExitPoint,
    build_action_plan,
    summarize_action_plan,
)
from gldpred.decision.scenario_analyzer import (    # noqa: F401
    ScenarioAnalysis,
    ScenarioOutcome,
    analyze_scenarios,
)

# Re-export the old names so existing imports keep working.  The canonical
# location for these symbols is the modules above.
__all__ = [
    "ActionPlan",
    "DayRecommendation",
    "DecisionRationale",
    "EntryWindow",
    "ExitPoint",
    "ScenarioAnalysis",
    "ScenarioOutcome",
    "analyze_scenarios",
    "build_action_plan",
    "summarize_action_plan",
]

