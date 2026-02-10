"""Model health and accountability service.

Analyses model staleness, prediction accuracy from trade-log outcomes,
training diagnostics quality, and produces actionable recalibration
recommendations.

Usage::

    from gldpred.services.health_service import HealthService, ModelHealthReport

    svc = HealthService()
    report = svc.report_for_model(model_id)
    reports = svc.report_all_assigned()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from gldpred.registry.store import ModelRegistry
from gldpred.registry.assignments import ModelAssignments
from gldpred.storage.trade_log import TradeLogStore


# ── Staleness thresholds (in days) ──────────────────────────────────────
FRESH_MAX_DAYS = 7
AGING_MAX_DAYS = 14
STALE_MAX_DAYS = 30
# > STALE_MAX_DAYS → expired


@dataclass
class ModelHealthReport:
    """Health analysis for a single model."""

    model_id: str
    label: str
    asset: str
    architecture: str
    created_at: str                           # ISO datetime
    age_days: int
    staleness: str                            # fresh / aging / stale / expired
    staleness_color: str                      # green / orange / red / gray
    is_primary: bool

    # Prediction accuracy (from closed trades for this model)
    total_trades: int = 0
    closed_trades: int = 0
    open_trades: int = 0
    win_rate: float = 0.0
    avg_predicted_return: float = 0.0
    avg_actual_return: float = 0.0
    prediction_bias: float = 0.0              # positive = overoptimistic

    # Training quality
    training_verdict: str = "unknown"
    best_val_loss: float = 0.0
    total_epochs: int = 0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class HealthService:
    """Compute health reports for models in the registry.

    Args:
        registry: ``ModelRegistry`` instance (default: fresh instance).
        assignments: ``ModelAssignments`` instance (default: fresh instance).
        trade_log: ``TradeLogStore`` instance (default: fresh instance).
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        assignments: Optional[ModelAssignments] = None,
        trade_log: Optional[TradeLogStore] = None,
    ) -> None:
        self._registry = registry or ModelRegistry()
        self._assignments = assignments or ModelAssignments()
        self._trade_log = trade_log or TradeLogStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report_for_model(
        self,
        model_id: str,
        *,
        now: Optional[datetime] = None,
    ) -> ModelHealthReport:
        """Generate a health report for a specific model.

        Args:
            model_id: Registry model identifier.
            now: Override current time (for testing).

        Returns:
            ``ModelHealthReport`` with staleness, accuracy, and advice.

        Raises:
            FileNotFoundError: If model does not exist in registry.
        """
        now = now or datetime.now()

        # 1. Load metadata from registry
        models = self._registry.list_models()
        meta = None
        for m in models:
            if m.get("model_id") == model_id:
                meta = m
                break
        if meta is None:
            raise FileNotFoundError(
                f"Model '{model_id}' not found in registry"
            )

        asset = meta.get("asset", "")
        label = meta.get("label", model_id)
        architecture = meta.get("architecture", "")
        created_at = meta.get("created_at", "")

        # 2. Compute age & staleness
        age_days = _compute_age_days(created_at, now)
        staleness, staleness_color = staleness_verdict(age_days)

        # 3. Is this the primary model for its asset?
        is_primary = self._assignments.get(asset) == model_id

        # 4. Trade-log accuracy for this model
        all_trades = self._trade_log.load_all()
        model_trades = [t for t in all_trades if t.model_id == model_id]
        closed = [t for t in model_trades if t.status == "closed"]
        open_trades = [t for t in model_trades if t.status == "open"]

        total_trades = len(model_trades)
        closed_trades = len(closed)
        open_count = len(open_trades)

        win_rate = 0.0
        avg_predicted = 0.0
        avg_actual = 0.0
        prediction_bias = 0.0

        if closed:
            wins = [t for t in closed if (t.actual_return_pct or 0) > 0]
            win_rate = len(wins) / len(closed) * 100
            avg_predicted = sum(t.expected_return_pct for t in closed) / len(closed)
            avg_actual = sum(t.actual_return_pct or 0 for t in closed) / len(closed)
            prediction_bias = avg_predicted - avg_actual

        # 5. Training quality from metadata
        training_summary = meta.get("training_summary", {})
        training_verdict = _extract_training_verdict(training_summary)
        val_losses = training_summary.get("val_losses", [])
        best_val_loss = min(val_losses) if val_losses else 0.0
        total_epochs = len(val_losses)

        # 6. Build recommendations
        recommendations = _build_recommendations(
            staleness=staleness,
            age_days=age_days,
            win_rate=win_rate,
            closed_trades=closed_trades,
            prediction_bias=prediction_bias,
            training_verdict=training_verdict,
        )

        return ModelHealthReport(
            model_id=model_id,
            label=label,
            asset=asset,
            architecture=architecture,
            created_at=created_at,
            age_days=age_days,
            staleness=staleness,
            staleness_color=staleness_color,
            is_primary=is_primary,
            total_trades=total_trades,
            closed_trades=closed_trades,
            open_trades=open_count,
            win_rate=win_rate,
            avg_predicted_return=avg_predicted,
            avg_actual_return=avg_actual,
            prediction_bias=prediction_bias,
            training_verdict=training_verdict,
            best_val_loss=best_val_loss,
            total_epochs=total_epochs,
            recommendations=recommendations,
        )

    def report_all_assigned(
        self,
        *,
        now: Optional[datetime] = None,
    ) -> List[ModelHealthReport]:
        """Health reports for all models currently assigned as primary.

        Returns:
            List of ``ModelHealthReport`` sorted by asset ticker.
        """
        mapping = self._assignments.get_all()
        reports: List[ModelHealthReport] = []
        for _asset, model_id in sorted(mapping.items()):
            try:
                reports.append(self.report_for_model(model_id, now=now))
            except FileNotFoundError:
                continue
        return reports

    def report_all_models(
        self,
        *,
        asset: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> List[ModelHealthReport]:
        """Health reports for all models in the registry.

        Args:
            asset: Optional filter by asset ticker.
            now: Override current time (for testing).

        Returns:
            List of ``ModelHealthReport`` sorted by creation date (newest first).
        """
        models = self._registry.list_models(asset=asset)
        reports: List[ModelHealthReport] = []
        for meta in models:
            mid = meta.get("model_id", "")
            if not mid:
                continue
            try:
                reports.append(self.report_for_model(mid, now=now))
            except FileNotFoundError:
                continue
        # Newest first
        reports.sort(key=lambda r: r.created_at, reverse=True)
        return reports


# ======================================================================
# Standalone helpers (importable for testing)
# ======================================================================

def staleness_verdict(age_days: int) -> Tuple[str, str]:
    """Return (verdict_str, color_str) for a model age.

    >>> staleness_verdict(3)
    ('fresh', 'green')
    >>> staleness_verdict(10)
    ('aging', 'orange')
    >>> staleness_verdict(20)
    ('stale', 'red')
    >>> staleness_verdict(45)
    ('expired', 'gray')
    """
    if age_days <= FRESH_MAX_DAYS:
        return ("fresh", "green")
    if age_days <= AGING_MAX_DAYS:
        return ("aging", "orange")
    if age_days <= STALE_MAX_DAYS:
        return ("stale", "red")
    return ("expired", "gray")


# ======================================================================
# Private helpers
# ======================================================================

def _compute_age_days(created_at: str, now: datetime) -> int:
    """Parse ISO date and return days elapsed."""
    if not created_at:
        return 999  # unknown → treat as expired
    try:
        created = datetime.fromisoformat(created_at)
        delta = now - created
        return max(int(delta.total_seconds() / 86400), 0)
    except (ValueError, TypeError):
        return 999


def _extract_training_verdict(training_summary: Dict[str, Any]) -> str:
    """Pull diagnostics verdict from training summary if available."""
    # The trainer may store diagnostics directly
    diag = training_summary.get("diagnostics", {})
    if isinstance(diag, dict) and "verdict" in diag:
        return diag["verdict"]
    # Fallback: check if there's a nested key
    verdict = training_summary.get("verdict", "")
    if verdict:
        return verdict
    return "unknown"


def _build_recommendations(
    *,
    staleness: str,
    age_days: int,
    win_rate: float,
    closed_trades: int,
    prediction_bias: float,
    training_verdict: str,
) -> List[str]:
    """Generate actionable recommendations based on health indicators."""
    recs: List[str] = []

    # Staleness-based
    if staleness == "expired":
        recs.append(
            f"Model is {age_days} days old — significantly outdated. "
            "Retrain immediately with recent data."
        )
    elif staleness == "stale":
        recs.append(
            f"Model is {age_days} days old — getting stale. "
            "Consider retraining within the next few days."
        )
    elif staleness == "aging":
        recs.append(
            f"Model is {age_days} days old — still usable but aging. "
            "Plan a recalibration soon."
        )
    else:
        recs.append("Model freshness is good — no recalibration needed yet.")

    # Accuracy-based (only if we have enough closed trades)
    if closed_trades >= 3:
        if win_rate < 40:
            recs.append(
                f"Win rate is low ({win_rate:.0f}%). "
                "Consider retraining or switching architecture."
            )
        elif win_rate >= 60:
            recs.append(
                f"Win rate is solid ({win_rate:.0f}%). "
                "Model is performing well."
            )

        if abs(prediction_bias) > 1.5:
            direction = "over-optimistic" if prediction_bias > 0 else "over-pessimistic"
            recs.append(
                f"Prediction bias detected: model is {direction} "
                f"by {abs(prediction_bias):.1f}pp on average."
            )
    elif closed_trades == 0:
        recs.append(
            "No closed trades yet — archive and close some recommendations "
            "to track prediction accuracy."
        )

    # Training-quality-based
    if training_verdict == "overfitting":
        recs.append(
            "Training showed overfitting signs — consider reducing "
            "model complexity or adding dropout."
        )
    elif training_verdict == "underfitting":
        recs.append(
            "Training showed underfitting — consider increasing "
            "capacity or training for more epochs."
        )
    elif training_verdict == "noisy":
        recs.append(
            "Training was noisy — try lowering learning rate "
            "or increasing batch size."
        )

    return recs
