"""Rule-based decision engine for trade recommendations.

Combines forecast trajectories, trend filters, volatility analysis,
market regime detection, model health gating, and **asset-class-aware
risk adjustments** to produce a **BUY / HOLD / AVOID** recommendation
with confidence, risk metrics, stop-loss / take-profit levels, and
rationale.

.. warning::
    This is a **decision-support tool**, not financial advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gldpred.config.assets import AssetInfo


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class RiskMetrics:
    """Quantitative risk assessment accompanying a recommendation."""

    stop_loss_pct: float = 0.0      # suggested stop-loss (negative %)
    take_profit_pct: float = 0.0    # suggested take-profit (positive %)
    risk_reward_ratio: float = 0.0  # |take_profit / stop_loss|
    max_drawdown_pct: float = 0.0   # worst peak-to-trough in forecast
    volatility_regime: str = "normal"  # "low", "normal", "high"


@dataclass
class Recommendation:
    """Output of the decision engine."""

    action: str                          # "BUY", "HOLD", "AVOID"
    confidence: float                    # 0-100
    rationale: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    risk: RiskMetrics = field(default_factory=RiskMetrics)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    score_components: Dict[str, float] = field(default_factory=dict)  # NEW: breakdown
    score_rationale: Dict[str, str] = field(default_factory=dict)     # NEW: explanations


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
        asset_info: Optional["AssetInfo"] = None,
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
        asset_info : AssetInfo or None
            Asset metadata for risk-aware scoring adjustments.
        """
        score = 50.0
        rationale: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}
        score_components: Dict[str, float] = {"base": 50.0}  # NEW: track each contrib
        score_rationale: Dict[str, str] = {}                 # NEW: explain each contrib

        q_idx = {round(q, 2): i for i, q in enumerate(quantiles)}
        median_idx = q_idx.get(0.5, len(quantiles) // 2)
        lo_idx = q_idx.get(0.1, 0)
        hi_idx = q_idx.get(0.9, len(quantiles) - 1)

        # Window over decision horizon (clip to available steps)
        h = min(self.horizon_days, returns_quantiles.shape[0])

        # --- 1. Expected return (median) ---------------------------------
        cum_return = float(np.sum(returns_quantiles[:h, median_idx]))
        details["cumulative_return_median"] = cum_return

        expected_return_score = 0.0
        if cum_return >= self.min_expected_return:
            expected_return_score = 20.0
            score += 20
            rationale.append(
                f"Expected {h}-day return {cum_return:+.2%} exceeds "
                f"threshold {self.min_expected_return:.2%}"
            )
            score_rationale["expected_return"] = f"Median +{cum_return:.2%} over {h}d"
        elif cum_return <= -self.min_expected_return:
            expected_return_score = -20.0
            score -= 20
            rationale.append(
                f"Expected {h}-day return {cum_return:+.2%} is negative"
            )
            score_rationale["expected_return"] = f"Median {cum_return:.2%} (negative)"
        else:
            rationale.append(
                f"Expected {h}-day return {cum_return:+.2%} is near zero"
            )
            score_rationale["expected_return"] = f"Median {cum_return:.2%} (neutral)"
        score_components["expected_return"] = expected_return_score

        # --- 2. Uncertainty width ----------------------------------------
        spread = float(
            np.mean(
                returns_quantiles[:h, hi_idx] - returns_quantiles[:h, lo_idx]
            )
        )
        details["mean_uncertainty_spread"] = spread
        uncertainty_score = 0.0
        if spread > 0.03:
            uncertainty_score = -10.0
            score -= 10
            warnings.append(
                f"High forecast uncertainty (avg daily spread {spread:.4f})"
            )
            score_rationale["uncertainty"] = f"High spread {spread:.3f}"
        elif spread < 0.01:
            uncertainty_score = 5.0
            score += 5
            rationale.append("Low forecast uncertainty — tight prediction band")
            score_rationale["uncertainty"] = f"Low spread {spread:.3f}"
        else:
            score_rationale["uncertainty"] = f"Moderate spread {spread:.3f}"
        score_components["uncertainty"] = uncertainty_score

        # --- 3. Trend filter (SMA) ----------------------------------------
        trend_score, trend_msgs = self._trend_filter(df)
        score += trend_score
        rationale.extend(trend_msgs)
        score_components["trend"] = float(trend_score)
        if trend_msgs:
            score_rationale["trend"] = trend_msgs[0][:50]  # truncate for display

        # --- 4. Volatility filter (ATR %) ---------------------------------
        vol_score, vol_msgs, vol_warns = self._volatility_filter(df)
        score += vol_score
        rationale.extend(vol_msgs)
        warnings.extend(vol_warns)
        score_components["volatility"] = float(vol_score)
        if vol_msgs or vol_warns:
            score_rationale["volatility"] = (vol_msgs + vol_warns)[0][:50]

        # --- 5. Market regime detection -----------------------------------
        regime = self._detect_regime(df)
        details["market_regime"] = regime
        regime_score = 0.0
        if regime == "high_volatility":
            regime_score = -5.0
            score -= 5
            warnings.append("Market regime: high volatility environment")
            score_rationale["regime"] = "High volatility"
        elif regime == "trending_up":
            regime_score = 5.0
            score += 5
            rationale.append("Market regime: trending up")
            score_rationale["regime"] = "Trending up"
        elif regime == "trending_down":
            regime_score = -5.0
            score -= 5
            rationale.append("Market regime: trending down")
            score_rationale["regime"] = "Trending down"
        else:
            score_rationale["regime"] = regime.replace("_", " ").title()
        score_components["regime"] = regime_score

        # --- 6. Conflicting signals check ---------------------------------
        conflict_penalty, conflict_msgs = self._check_conflicting_signals(
            cum_return, trend_score, vol_score, spread,
        )
        score += conflict_penalty
        warnings.extend(conflict_msgs)
        score_components["conflicts"] = float(conflict_penalty)
        if conflict_msgs:
            score_rationale["conflicts"] = conflict_msgs[0][:50]

        # --- 7. Diagnostics gate ------------------------------------------
        diag_score = 0.0
        if diagnostics_verdict:
            details["diagnostics_verdict"] = diagnostics_verdict
            if diagnostics_verdict.lower() == "healthy":
                diag_score = 5.0
                score += 5
                rationale.append("Model diagnostics: healthy")
                score_rationale["diagnostics"] = "Healthy"
            elif diagnostics_verdict.lower() in ("overfitting", "overfit"):
                diag_score = -10.0
                score -= 10
                warnings.append(
                    "Model diagnostics: overfitting — predictions may be unreliable"
                )
                score_rationale["diagnostics"] = "Overfitting"
            elif diagnostics_verdict.lower() in ("underfitting", "underfit"):
                diag_score = -5.0
                score -= 5
                warnings.append("Model diagnostics: underfitting")
                score_rationale["diagnostics"] = "Underfitting"
            elif diagnostics_verdict.lower() == "noisy":
                diag_score = -5.0
                score -= 5
                warnings.append("Model diagnostics: noisy training")
                score_rationale["diagnostics"] = "Noisy"
        score_components["diagnostics"] = diag_score

        # --- 8. Asset-class risk modifier ---------------------------------
        asset_class_score = 0.0
        if asset_info is not None:
            asset_class_score, ac_msgs, ac_warns = self._asset_class_modifier(
                asset_info, cum_return, spread,
            )
            score += asset_class_score
            rationale.extend(ac_msgs)
            warnings.extend(ac_warns)
            details["risk_level"] = asset_info.risk_level
            details["asset_role"] = asset_info.role
            details["volatility_profile"] = asset_info.volatility_profile
            if ac_msgs or ac_warns:
                score_rationale["asset_class"] = (ac_msgs + ac_warns)[0][:60]
        score_components["asset_class"] = asset_class_score

        # --- Clamp & decide -----------------------------------------------
        score = float(np.clip(score, 0, 100))
        details["score"] = score

        if score >= self.buy_threshold and cum_return > 0:
            action = "BUY"
        elif score <= self.avoid_threshold or cum_return < -self.min_expected_return:
            action = "AVOID"
        else:
            action = "HOLD"

        # --- Risk metrics -------------------------------------------------
        risk = self._compute_risk_metrics(
            returns_quantiles, h, lo_idx, median_idx, hi_idx, df,
        )

        return Recommendation(
            action=action,
            confidence=score,
            rationale=rationale,
            warnings=warnings,
            details=details,
            risk=risk,
            score_components=score_components,
            score_rationale=score_rationale,
        )

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------
    def _compute_risk_metrics(
        self,
        returns_q: np.ndarray,
        horizon: int,
        lo_idx: int,
        med_idx: int,
        hi_idx: int,
        df: pd.DataFrame,
    ) -> RiskMetrics:
        """Compute stop-loss, take-profit, and related risk metrics."""
        cum_lo = float(np.sum(returns_q[:horizon, lo_idx]))
        cum_hi = float(np.sum(returns_q[:horizon, hi_idx]))

        # Stop-loss: pessimistic scenario with a buffer
        stop_loss = cum_lo * 1.2 if cum_lo < 0 else cum_lo - 0.005
        stop_loss = min(stop_loss, -0.002)  # at least -0.2 %

        # Take-profit: optimistic scenario with conservative trim
        take_profit = cum_hi * 0.8 if cum_hi > 0 else cum_hi + 0.005
        take_profit = max(take_profit, 0.002)

        # Risk-reward ratio
        risk_reward = (
            abs(take_profit / stop_loss) if abs(stop_loss) > 1e-8 else 0.0
        )

        # Max drawdown from the pessimistic price path
        cum_rets_lo = np.cumsum(returns_q[:horizon, lo_idx])
        running_max = np.maximum.accumulate(1.0 + cum_rets_lo)
        drawdowns = (1.0 + cum_rets_lo) / running_max - 1.0
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Volatility regime from recent ATR%
        vol_regime = "normal"
        if not df.empty and "atr_pct" in df.columns:
            recent_atr = float(df["atr_pct"].iloc[-1])
            if recent_atr <= self.max_volatility * 0.7:
                vol_regime = "low"
            elif recent_atr > self.max_volatility * 1.3:
                vol_regime = "high"

        return RiskMetrics(
            stop_loss_pct=round(stop_loss * 100, 2),
            take_profit_pct=round(take_profit * 100, 2),
            risk_reward_ratio=round(risk_reward, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            volatility_regime=vol_regime,
        )

    # ------------------------------------------------------------------
    # Market regime detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_regime(df: pd.DataFrame) -> str:
        """Classify market regime from recent price data.

        Returns one of: ``"trending_up"``, ``"trending_down"``,
        ``"ranging"``, ``"high_volatility"``, ``"unknown"``.
        """
        if df.empty or len(df) < 20:
            return "unknown"

        close = df["Close"].values
        tail = close[-20:]

        x = np.arange(len(tail), dtype=float)
        slope = float(np.polyfit(x, tail, 1)[0])
        norm_slope = slope / float(np.mean(tail))

        if "volatility_20" in df.columns:
            vol = float(df["volatility_20"].iloc[-1])
        else:
            vol = float(np.std(np.diff(tail) / tail[:-1]))

        if vol > 0.025:
            return "high_volatility"
        if norm_slope > 0.002:
            return "trending_up"
        if norm_slope < -0.002:
            return "trending_down"
        return "ranging"

    # ------------------------------------------------------------------
    # Conflicting signals
    # ------------------------------------------------------------------
    @staticmethod
    def _check_conflicting_signals(
        cum_return: float,
        trend_score: float,
        vol_score: float,
        spread: float,
    ) -> tuple[float, List[str]]:
        """Detect when forecast and technical signals contradict."""
        msgs: List[str] = []
        penalty = 0.0

        if cum_return > 0.005 and trend_score < -10:
            penalty -= 5
            msgs.append(
                "Conflicting signals: forecast is positive but trend is bearish"
            )
        if cum_return < -0.005 and trend_score > 10:
            penalty -= 5
            msgs.append(
                "Conflicting signals: forecast is negative but trend is bullish"
            )
        if spread > 0.02 and abs(cum_return) > 0.01:
            penalty -= 3
            msgs.append(
                "Wide uncertainty band despite a strong directional forecast"
            )

        return penalty, msgs

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

    # ------------------------------------------------------------------
    # Asset-class risk modifier
    # ------------------------------------------------------------------
    @staticmethod
    def _asset_class_modifier(
        asset_info: "AssetInfo",
        cum_return: float,
        spread: float,
    ) -> tuple[float, List[str], List[str]]:
        """Adjust score based on asset risk classification.

        High-risk assets receive a penalty unless their return justifies it.
        Low-risk assets get a small stability bonus.  The benchmark role
        is treated neutrally.

        Returns ``(score_delta, rationale_msgs, warning_msgs)``.
        """
        score = 0.0
        msgs: List[str] = []
        warns: List[str] = []

        risk = asset_info.risk_level
        role = asset_info.role
        vol_profile = asset_info.volatility_profile

        # Risk-tier adjustments
        if risk == "high":
            # High-risk assets need a stronger signal to justify BUY
            if cum_return < 0.005:
                score -= 5
                warns.append(
                    f"High-risk asset ({asset_info.ticker}) — modest return "
                    f"does not compensate risk"
                )
            elif spread > 0.02:
                score -= 3
                warns.append(
                    f"High-risk asset with wide uncertainty — caution advised"
                )
            else:
                msgs.append(
                    f"High-risk asset with strong return signal — "
                    f"risk may be justified"
                )
        elif risk == "low":
            # Low-risk / stable assets get a small confidence bonus
            score += 3
            msgs.append(f"Low-risk asset — stability factor applied")

        # Volatile profile extra scrutiny
        if vol_profile == "volatile" and spread > 0.015:
            score -= 2
            warns.append(
                f"Volatile asset class with elevated forecast uncertainty"
            )

        # Speculative role: require conviction
        if role == "speculative" and cum_return < 0.01:
            score -= 3
            warns.append(
                f"Speculative asset — requires higher conviction "
                f"(return {cum_return:+.2%})"
            )

        return score, msgs, warns


# ======================================================================
# Recommendation history helper
# ======================================================================

class RecommendationHistory:
    """In-memory history of recommendations for the current session."""

    def __init__(self) -> None:
        self._history: List[Dict[str, Any]] = []

    def add(self, asset: str, reco: Recommendation) -> None:
        """Record a recommendation."""
        self._history.append({
            "asset": asset,
            "action": reco.action,
            "confidence": reco.confidence,
            "timestamp": reco.timestamp,
            "risk": {
                "stop_loss_pct": reco.risk.stop_loss_pct,
                "take_profit_pct": reco.risk.take_profit_pct,
                "risk_reward_ratio": reco.risk.risk_reward_ratio,
                "max_drawdown_pct": reco.risk.max_drawdown_pct,
                "volatility_regime": reco.risk.volatility_regime,
            },
            "details": reco.details,
        })

    def get_history(self, asset: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return recommendations, optionally filtered by asset."""
        if asset:
            return [r for r in self._history if r["asset"] == asset]
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)
