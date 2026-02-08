"""Rule-based decision engine for trade recommendations.

Combines forecast trajectories, trend filters, and volatility analysis to
produce a **BUY / HOLD / AVOID** recommendation with confidence and rationale.

.. warning::
    This is a **decision-support tool**, not financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Recommendation:
    """Output of the decision engine."""

    action: str  # "BUY", "HOLD", "AVOID"
    confidence: float  # 0-100
    rationale: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class DecisionEngine:
    """Score-based recommendation from trajectory forecasts and technicals.

    Parameters
    ----------
    horizon_days : int
        Number of forecast days to consider for the decision window.
    min_expected_return : float
        Minimum cumulative return (decimal) to qualify as BUY.
    max_volatility : float
        Maximum ATR-% level above which risk is elevated.
    buy_threshold : float
        Score ≥ this → BUY.
    avoid_threshold : float
        Score ≤ this → AVOID.
    """

    def __init__(
        self,
        horizon_days: int = 5,
        min_expected_return: float = 0.008,
        max_volatility: float = 0.02,
        buy_threshold: float = 65.0,
        avoid_threshold: float = 35.0,
    ) -> None:
        self.horizon_days = horizon_days
        self.min_expected_return = min_expected_return
        self.max_volatility = max_volatility
        self.buy_threshold = buy_threshold
        self.avoid_threshold = avoid_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def recommend(
        self,
        returns_quantiles: np.ndarray,
        df: pd.DataFrame,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        diagnostics_verdict: Optional[str] = None,
    ) -> Recommendation:
        """Generate a recommendation.

        Parameters
        ----------
        returns_quantiles : ndarray of shape ``(K, Q)``
            Forecasted daily returns for each quantile.
        df : DataFrame
            Latest market data with technical indicators (SMA_50, SMA_200,
            atr_pct, Close, etc.).
        quantiles : tuple
            Which quantile columns correspond to the Q dimension.
        diagnostics_verdict : str or None
            Training diagnostics verdict (e.g. ``"healthy"``).
        """
        score = 50.0
        rationale: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        q_idx = {round(q, 2): i for i, q in enumerate(quantiles)}
        median_idx = q_idx.get(0.5, len(quantiles) // 2)
        lo_idx = q_idx.get(0.1, 0)
        hi_idx = q_idx.get(0.9, len(quantiles) - 1)

        # Window over decision horizon (clip to available steps)
        h = min(self.horizon_days, returns_quantiles.shape[0])

        # --- 1. Expected return (median) ---------------------------------
        cum_return = float(np.sum(returns_quantiles[:h, median_idx]))
        details["cumulative_return_median"] = cum_return

        if cum_return >= self.min_expected_return:
            score += 20
            rationale.append(
                f"Expected {h}-day return {cum_return:+.2%} exceeds "
                f"threshold {self.min_expected_return:.2%}"
            )
        elif cum_return <= -self.min_expected_return:
            score -= 20
            rationale.append(
                f"Expected {h}-day return {cum_return:+.2%} is negative"
            )
        else:
            rationale.append(
                f"Expected {h}-day return {cum_return:+.2%} is near zero"
            )

        # --- 2. Uncertainty width ----------------------------------------
        spread = float(
            np.mean(
                returns_quantiles[:h, hi_idx] - returns_quantiles[:h, lo_idx]
            )
        )
        details["mean_uncertainty_spread"] = spread
        if spread > 0.03:
            score -= 10
            warnings.append(
                f"High forecast uncertainty (avg daily spread {spread:.4f})"
            )
        elif spread < 0.01:
            score += 5
            rationale.append("Low forecast uncertainty — tight prediction band")

        # --- 3. Trend filter (SMA) ----------------------------------------
        trend_score, trend_msgs = self._trend_filter(df)
        score += trend_score
        rationale.extend(trend_msgs)

        # --- 4. Volatility filter (ATR %) ---------------------------------
        vol_score, vol_msgs, vol_warns = self._volatility_filter(df)
        score += vol_score
        rationale.extend(vol_msgs)
        warnings.extend(vol_warns)

        # --- 5. Diagnostics gate ------------------------------------------
        if diagnostics_verdict:
            details["diagnostics_verdict"] = diagnostics_verdict
            if diagnostics_verdict.lower() == "healthy":
                score += 5
                rationale.append("Model diagnostics: healthy")
            elif diagnostics_verdict.lower() in ("overfitting", "overfit"):
                score -= 10
                warnings.append(
                    "Model diagnostics: overfitting — predictions may be unreliable"
                )
            elif diagnostics_verdict.lower() in ("underfitting", "underfit"):
                score -= 5
                warnings.append("Model diagnostics: underfitting")
            elif diagnostics_verdict.lower() == "noisy":
                score -= 5
                warnings.append("Model diagnostics: noisy training")

        # --- Clamp & decide -----------------------------------------------
        score = float(np.clip(score, 0, 100))
        details["score"] = score

        if score >= self.buy_threshold and cum_return > 0:
            action = "BUY"
        elif score <= self.avoid_threshold or cum_return < -self.min_expected_return:
            action = "AVOID"
        else:
            action = "HOLD"

        return Recommendation(
            action=action,
            confidence=score,
            rationale=rationale,
            warnings=warnings,
            details=details,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _trend_filter(self, df: pd.DataFrame) -> tuple[float, List[str]]:
        """Return (score_delta, rationale_messages) from SMA analysis."""
        score = 0.0
        msgs: List[str] = []
        if df.empty:
            return score, msgs

        last = df.iloc[-1]

        has_sma50 = "sma_50" in df.columns
        has_sma200 = "sma_200" in df.columns
        close = float(last.get("Close", 0))

        if has_sma50 and has_sma200:
            sma50 = float(last["sma_50"])
            sma200 = float(last["sma_200"])

            above_sma200 = close > sma200
            golden_cross = sma50 > sma200

            if above_sma200 and golden_cross:
                score += 15
                msgs.append("Bullish trend: price > SMA-200 and SMA-50 > SMA-200")
            elif above_sma200:
                score += 5
                msgs.append("Price above SMA-200 but SMA-50 < SMA-200")
            elif not above_sma200 and not golden_cross:
                score -= 15
                msgs.append("Bearish trend: price < SMA-200 and SMA-50 < SMA-200")
            else:
                score -= 5
                msgs.append("Mixed trend: price < SMA-200 but SMA-50 > SMA-200")
        elif has_sma200:
            sma200 = float(last["sma_200"])
            if close > sma200:
                score += 10
                msgs.append("Price above SMA-200")
            else:
                score -= 10
                msgs.append("Price below SMA-200")

        return score, msgs

    def _volatility_filter(
        self, df: pd.DataFrame
    ) -> tuple[float, List[str], List[str]]:
        """Return (score_delta, rationale, warnings) from ATR analysis."""
        score = 0.0
        msgs: List[str] = []
        warns: List[str] = []
        if df.empty or "atr_pct" not in df.columns:
            return score, msgs, warns

        atr_pct = float(df["atr_pct"].iloc[-1])
        details_str = f"ATR% = {atr_pct:.4f}"

        if atr_pct <= self.max_volatility:
            score += 5
            msgs.append(f"Volatility within limits ({details_str})")
        elif atr_pct <= self.max_volatility * 1.5:
            score -= 5
            warns.append(f"Moderate volatility ({details_str})")
        else:
            score -= 10
            warns.append(f"High volatility ({details_str})")

        return score, msgs, warns
