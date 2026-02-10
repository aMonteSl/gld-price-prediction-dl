"""Decision engine — recommendation logic, action planning, and portfolio comparison."""
from gldpred.decision.action_planner import (
    ActionPlan,
    DayRecommendation,
    DecisionRationale,
    EntryWindow,
    ExitPoint,
    build_action_plan,
    summarize_action_plan,
)
from gldpred.decision.engine import (
    DecisionEngine,
    Recommendation,
    RecommendationHistory,
    RiskMetrics,
)
from gldpred.decision.portfolio import (
    AssetOutcome,
    ComparisonResult,
    PortfolioComparator,
)
from gldpred.decision.scenario_analyzer import (
    ScenarioAnalysis,
    ScenarioOutcome,
    analyze_scenarios,
)

__all__ = [
    "ActionPlan",
    "AssetOutcome",
    "ComparisonResult",
    "DayRecommendation",
    "DecisionEngine",
    "DecisionRationale",
    "EntryWindow",
    "ExitPoint",
    "PortfolioComparator",
    "Recommendation",
    "RecommendationHistory",
    "RiskMetrics",
    "ScenarioAnalysis",
    "ScenarioOutcome",
    "analyze_scenarios",
    "build_action_plan",
    "summarize_action_plan",
]
