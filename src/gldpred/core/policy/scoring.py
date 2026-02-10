"""Transparent scoring pipeline backed by DecisionEngine.

``DecisionPolicy`` runs the existing ``DecisionEngine.recommend()`` logic
then unpacks the score into individually weighted ``ScoreFactor`` objects.
The UI can render each factor with a progress bar, colour, and rationale.

.. note:: This does **not** change the ML logic or thresholds.
   It wraps the existing engine and enriches the output format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from gldpred.decision.engine import DecisionEngine, Recommendation, RiskMetrics


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class ScoreFactor:
    """One factor in the decision score breakdown.

    Attributes:
        name: Machine-readable factor identifier.
        label_en: English label for display.
        label_es: Spanish label for display.
        contribution: Signed score contribution (can be negative).
        max_possible: Maximum absolute contribution this factor can have.
        rationale: One-sentence explanation of this factor's outcome.
        sentiment: ``"positive"`` | ``"negative"`` | ``"neutral"``
    """

    name: str
    label_en: str
    label_es: str
    contribution: float
    max_possible: float
    rationale: str = ""
    sentiment: str = "neutral"


@dataclass
class PolicyResult:
    """Full transparent result from the decision policy.

    Contains everything needed for the UI to render a rich decision card.
    """

    action: str                        # BUY / HOLD / AVOID
    confidence: float                  # 0-100
    total_score: float                 # raw score (0-100)
    factors: List[ScoreFactor] = field(default_factory=list)
    risk: RiskMetrics = field(default_factory=RiskMetrics)
    rationale: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    market_regime: str = "unknown"
    cumulative_return: float = 0.0
    uncertainty_spread: float = 0.0

    # Original Recommendation (for backward compat)
    _recommendation: Optional[Recommendation] = field(
        default=None, repr=False,
    )

    @property
    def recommendation(self) -> Recommendation:
        """Return the underlying ``Recommendation``."""
        if self._recommendation is not None:
            return self._recommendation
        return Recommendation(
            action=self.action,
            confidence=self.confidence,
            rationale=self.rationale,
            warnings=self.warnings,
            details=self.details,
            risk=self.risk,
        )


# ======================================================================
# Factor label catalog
# ======================================================================

_FACTOR_META = {
    "base": {
        "label_en": "Base Score",
        "label_es": "Puntuación Base",
        "max": 50.0,
    },
    "expected_return": {
        "label_en": "Expected Return",
        "label_es": "Retorno Esperado",
        "max": 20.0,
    },
    "uncertainty": {
        "label_en": "Forecast Uncertainty",
        "label_es": "Incertidumbre del Pronóstico",
        "max": 10.0,
    },
    "trend": {
        "label_en": "Trend (SMA)",
        "label_es": "Tendencia (SMA)",
        "max": 15.0,
    },
    "volatility": {
        "label_en": "Volatility (ATR)",
        "label_es": "Volatilidad (ATR)",
        "max": 10.0,
    },
    "regime": {
        "label_en": "Market Regime",
        "label_es": "Régimen de Mercado",
        "max": 5.0,
    },
    "conflicts": {
        "label_en": "Conflicting Signals",
        "label_es": "Señales Conflictivas",
        "max": 8.0,
    },
    "diagnostics": {
        "label_en": "Model Health",
        "label_es": "Salud del Modelo",
        "max": 10.0,
    },
}


# ======================================================================
# DecisionPolicy
# ======================================================================

class DecisionPolicy:
    """Transparent wrapper around ``DecisionEngine``.

    Usage::

        policy = DecisionPolicy()
        result = policy.evaluate(returns_q, df, quantiles)
        for f in result.factors:
            print(f"{f.label_en}: {f.contribution:+.1f}")
    """

    def __init__(
        self,
        horizon_days: int = 5,
        min_expected_return: float = 0.008,
        max_volatility: float = 0.02,
        buy_threshold: float = 65.0,
        avoid_threshold: float = 35.0,
    ) -> None:
        self._engine = DecisionEngine(
            horizon_days=horizon_days,
            min_expected_return=min_expected_return,
            max_volatility=max_volatility,
            buy_threshold=buy_threshold,
            avoid_threshold=avoid_threshold,
        )
        self.buy_threshold = buy_threshold
        self.avoid_threshold = avoid_threshold

    def evaluate(
        self,
        returns_quantiles: np.ndarray,
        df: pd.DataFrame,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        diagnostics_verdict: Optional[str] = None,
    ) -> PolicyResult:
        """Run the engine and return a transparent ``PolicyResult``."""
        rec = self._engine.recommend(
            returns_quantiles, df, quantiles, diagnostics_verdict,
        )

        # Unpack score_components from the Recommendation
        components = rec.score_components
        score_rationale = rec.score_rationale

        factors: List[ScoreFactor] = []
        for name, contribution in components.items():
            meta = _FACTOR_META.get(name, {})
            sentiment = "neutral"
            if contribution > 0:
                sentiment = "positive"
            elif contribution < 0:
                sentiment = "negative"

            factors.append(ScoreFactor(
                name=name,
                label_en=meta.get("label_en", name.replace("_", " ").title()),
                label_es=meta.get("label_es", name.replace("_", " ").title()),
                contribution=contribution,
                max_possible=meta.get("max", abs(contribution) + 5),
                rationale=score_rationale.get(name, ""),
                sentiment=sentiment,
            ))

        # Sort: base first, then by absolute contribution descending
        factors.sort(
            key=lambda f: (f.name != "base", -abs(f.contribution)),
        )

        return PolicyResult(
            action=rec.action,
            confidence=rec.confidence,
            total_score=rec.details.get("score", rec.confidence),
            factors=factors,
            risk=rec.risk,
            rationale=rec.rationale,
            warnings=rec.warnings,
            details=rec.details,
            market_regime=rec.details.get("market_regime", "unknown"),
            cumulative_return=rec.details.get("cumulative_return_median", 0.0),
            uncertainty_spread=rec.details.get("mean_uncertainty_spread", 0.0),
            _recommendation=rec,
        )
