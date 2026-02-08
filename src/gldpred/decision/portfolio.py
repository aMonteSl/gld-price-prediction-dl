"""Multi-asset portfolio comparison engine.

Given an investment amount and a set of assets (each with a trained model),
this module runs forecasts for every asset, computes EUR/USD outcomes,
and produces a ranked leaderboard with risk-adjusted metrics.

Usage::

    from gldpred.decision.portfolio import PortfolioComparator, AssetOutcome

    comparator = PortfolioComparator()
    outcomes = comparator.compare(
        forecasts={"GLD": forecast_gld, "SLV": forecast_slv},
        dfs={"GLD": df_gld, "SLV": df_slv},
        investment=1000.0,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from gldpred.decision.engine import DecisionEngine, Recommendation


@dataclass
class AssetOutcome:
    """Projected outcome for a single asset given a fixed investment."""

    ticker: str
    investment: float                # amount invested (currency units)
    current_price: float
    shares: float                    # investment / current_price
    # Projected portfolio values at end of horizon
    projected_value_p10: float       # pessimistic
    projected_value_p50: float       # median
    projected_value_p90: float       # optimistic
    pnl_p10: float                   # profit/loss pessimistic
    pnl_p50: float                   # profit/loss median
    pnl_p90: float                   # profit/loss optimistic
    pnl_pct_p10: float               # % return pessimistic
    pnl_pct_p50: float               # % return median
    pnl_pct_p90: float               # % return optimistic
    recommendation: Recommendation = field(default_factory=lambda: Recommendation(action="HOLD", confidence=50.0))
    rank: int = 0


@dataclass
class ComparisonResult:
    """Full comparison output across multiple assets."""

    outcomes: List[AssetOutcome]
    investment: float
    horizon_days: int
    best_asset: str = ""
    summary: str = ""


class PortfolioComparator:
    """Compare projected outcomes across multiple assets.

    The comparator is stateless — pass forecast data and DataFrames for
    each asset to :meth:`compare`.
    """

    def __init__(
        self,
        horizon_days: int = 5,
        min_expected_return: float = 0.008,
    ) -> None:
        self.horizon_days = horizon_days
        self.min_expected_return = min_expected_return

    def compare(
        self,
        forecasts: Dict[str, Any],
        dfs: Dict[str, Any],
        investment: float = 1000.0,
        diagnostics_verdicts: Optional[Dict[str, str]] = None,
        max_volatilities: Optional[Dict[str, float]] = None,
    ) -> ComparisonResult:
        """Run comparison across assets.

        Args:
            forecasts: ``{ticker: TrajectoryForecast}`` for each asset.
            dfs: ``{ticker: DataFrame}`` with technical indicators.
            investment: Amount to hypothetically invest in each asset.
            diagnostics_verdicts: Optional ``{ticker: verdict}`` strings.
            max_volatilities: Optional ``{ticker: max_vol}`` overrides.

        Returns:
            ComparisonResult with ranked outcomes.
        """
        diagnostics_verdicts = diagnostics_verdicts or {}
        max_volatilities = max_volatilities or {}

        outcomes: List[AssetOutcome] = []

        for ticker, forecast in forecasts.items():
            df = dfs.get(ticker)
            if df is None or df.empty:
                continue

            max_vol = max_volatilities.get(ticker, 0.02)
            engine = DecisionEngine(
                horizon_days=self.horizon_days,
                min_expected_return=self.min_expected_return,
                max_volatility=max_vol,
            )

            reco = engine.recommend(
                forecast.returns_quantiles,
                df,
                quantiles=tuple(forecast.quantiles),
                diagnostics_verdict=diagnostics_verdicts.get(ticker),
            )

            # Compute monetary outcomes
            outcome = self._compute_outcome(
                ticker, forecast, investment, reco,
            )
            outcomes.append(outcome)

        # Rank by median PnL (descending)
        outcomes.sort(key=lambda o: o.pnl_pct_p50, reverse=True)
        for i, o in enumerate(outcomes):
            o.rank = i + 1

        best = outcomes[0].ticker if outcomes else ""

        result = ComparisonResult(
            outcomes=outcomes,
            investment=investment,
            horizon_days=self.horizon_days,
            best_asset=best,
        )
        result.summary = self._build_summary(result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_outcome(
        ticker: str,
        forecast: Any,
        investment: float,
        reco: Recommendation,
    ) -> AssetOutcome:
        """Compute projected portfolio value for one asset."""
        last_price = forecast.last_price
        shares = investment / last_price if last_price > 0 else 0.0

        q_idx = {round(q, 2): i for i, q in enumerate(forecast.quantiles)}
        lo_i = q_idx.get(0.1, 0)
        med_i = q_idx.get(0.5, 1)
        hi_i = q_idx.get(0.9, len(forecast.quantiles) - 1)

        # End-of-horizon prices
        end_idx = min(forecast.price_paths.shape[0] - 1, forecast.price_paths.shape[0] - 1)
        price_p10 = float(forecast.price_paths[end_idx, lo_i])
        price_p50 = float(forecast.price_paths[end_idx, med_i])
        price_p90 = float(forecast.price_paths[end_idx, hi_i])

        val_p10 = shares * price_p10
        val_p50 = shares * price_p50
        val_p90 = shares * price_p90

        pnl_p10 = val_p10 - investment
        pnl_p50 = val_p50 - investment
        pnl_p90 = val_p90 - investment

        pnl_pct_p10 = (pnl_p10 / investment * 100) if investment > 0 else 0.0
        pnl_pct_p50 = (pnl_p50 / investment * 100) if investment > 0 else 0.0
        pnl_pct_p90 = (pnl_p90 / investment * 100) if investment > 0 else 0.0

        return AssetOutcome(
            ticker=ticker,
            investment=investment,
            current_price=last_price,
            shares=shares,
            projected_value_p10=round(val_p10, 2),
            projected_value_p50=round(val_p50, 2),
            projected_value_p90=round(val_p90, 2),
            pnl_p10=round(pnl_p10, 2),
            pnl_p50=round(pnl_p50, 2),
            pnl_p90=round(pnl_p90, 2),
            pnl_pct_p10=round(pnl_pct_p10, 2),
            pnl_pct_p50=round(pnl_pct_p50, 2),
            pnl_pct_p90=round(pnl_pct_p90, 2),
            recommendation=reco,
        )

    @staticmethod
    def _build_summary(result: ComparisonResult) -> str:
        """Build a plain-text leaderboard summary."""
        if not result.outcomes:
            return "No assets to compare."
        lines = [
            f"Investment: ${result.investment:,.2f} | "
            f"Horizon: {result.horizon_days} days",
            "",
        ]
        for o in result.outcomes:
            lines.append(
                f"#{o.rank} {o.ticker}: "
                f"P50 PnL {o.pnl_pct_p50:+.2f}% "
                f"(${o.pnl_p50:+,.2f}) — "
                f"{o.recommendation.action} "
                f"({o.recommendation.confidence:.0f}/100)"
            )
        return "\n".join(lines)
