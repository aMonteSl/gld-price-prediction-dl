"""Scenario analysis for quantile trajectory forecasts.

Computes **optimistic** (P90), **base** (P50), and **pessimistic** (P10)
projections over a planning horizon, with percentage returns and
dollar-value impact for a given hypothetical investment.

All functions are **pure** — no Streamlit dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# ======================================================================
# Data classes
# ======================================================================

@dataclass(frozen=True)
class ScenarioOutcome:
    """Projection for a single quantile scenario."""

    label: str            # "optimistic" | "base" | "pessimistic"
    quantile: float       # e.g. 0.9, 0.5, 0.1
    return_pct: float     # cumulative return in percent
    final_price: float    # price at horizon end
    value_impact: float   # dollar P&L for the given investment
    price_path: List[float]  # day-by-day prices (length = horizon + 1)


@dataclass(frozen=True)
class ScenarioAnalysis:
    """Three-scenario comparison for a forecast horizon."""

    optimistic: ScenarioOutcome   # P90
    base: ScenarioOutcome         # P50
    pessimistic: ScenarioOutcome  # P10
    entry_price: float
    horizon: int
    investment: float


# ======================================================================
# Public API
# ======================================================================

def analyze_scenarios(
    price_paths: np.ndarray,
    quantiles: List[float],
    *,
    entry_price: float,
    horizon: int,
    investment: float = 10_000.0,
) -> ScenarioAnalysis:
    """Build a three-scenario analysis from forecast price paths.

    Parameters
    ----------
    price_paths : ndarray ``(K+1, Q)``
        Row 0 = entry price; rows 1..K = forecast prices.
    quantiles : list[float]
        Quantile levels used (e.g. ``[0.1, 0.5, 0.9]``).
    entry_price : float
        Current price (day-0 close).
    horizon : int
        Number of forecast days to analyse.
    investment : float
        Hypothetical investment amount for value-impact calculations.

    Returns
    -------
    ScenarioAnalysis
    """
    K = price_paths.shape[0] - 1
    h = min(max(horizon, 1), K)

    q_map = {round(q, 2): i for i, q in enumerate(quantiles)}
    idx_p10 = q_map.get(0.1, 0)
    idx_p50 = q_map.get(0.5, 1)
    idx_p90 = q_map.get(0.9, 2)

    def _outcome(label: str, quantile: float, col: int) -> ScenarioOutcome:
        final = float(price_paths[h, col])
        ret = (final - entry_price) / entry_price
        pnl = investment * ret
        path = [float(price_paths[t, col]) for t in range(h + 1)]
        return ScenarioOutcome(
            label=label,
            quantile=quantile,
            return_pct=round(ret * 100, 2),
            final_price=round(final, 2),
            value_impact=round(pnl, 2),
            price_path=path,
        )

    return ScenarioAnalysis(
        optimistic=_outcome("optimistic", 0.9, idx_p90),
        base=_outcome("base", 0.5, idx_p50),
        pessimistic=_outcome("pessimistic", 0.1, idx_p10),
        entry_price=entry_price,
        horizon=h,
        investment=investment,
    )
