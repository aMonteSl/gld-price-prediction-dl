"""Action planner — per-day BUY / HOLD / SELL / AVOID classification.

Uses a **state-machine** approach (NO_POSITION → BUY → HOLD → SELL → CLOSED)
to produce a coherent daily action timeline with entry-window detection,
risk-adjusted best-exit computation, and multi-factor decision rationale.

All public functions are **pure** (no Streamlit dependency) so they can be
tested and reused outside the GUI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gldpred.decision.scenario_analyzer import ScenarioAnalysis


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class DayRecommendation:
    """Action recommendation for a single forecast day."""

    day: int                   # 1-indexed
    date: str                  # ISO date string
    action: str                # BUY | HOLD | SELL | AVOID
    confidence: float          # 0-100
    rationale: str             # human-readable explanation
    price_p10: float
    price_p50: float
    price_p90: float
    ret_p10: float             # cumulative return vs entry (decimal)
    ret_p50: float
    ret_p90: float
    uncertainty_width: float   # (ret_p90 − ret_p10) spread (decimal)
    risk_score: float          # risk-adjusted score


@dataclass
class EntryWindow:
    """Optimal entry range — consecutive days favourable for buying."""

    start_day: int             # 1-indexed
    end_day: int               # 1-indexed
    avg_score: float           # average risk-adjusted score in window
    rationale: str


@dataclass
class ExitPoint:
    """Optimal exit point — the single best day to close the position."""

    day: int                   # 1-indexed
    date: str
    expected_return_pct: float
    risk_score: float
    rationale: str


@dataclass
class DecisionRationale:
    """Explainable multi-factor decision justification."""

    trend_confirmation: str    # e.g. "Bullish: SMA-50 > SMA-200"
    volatility_regime: str     # e.g. "Normal (ATR% = 0.014)"
    quantile_risk: str         # e.g. "P10 drawdown within limits"
    today_assessment: str      # Why today may / may not be optimal


@dataclass
class ActionPlan:
    """Complete time-based action plan."""

    daily_actions: List[DayRecommendation]
    entry_window: Optional[EntryWindow]
    best_exit: Optional[ExitPoint]
    scenarios: ScenarioAnalysis
    rationale: DecisionRationale
    narrative: str             # e.g. "Buy today → hold 5 days → sell day 6"
    overall_signal: str        # BUY | HOLD | SELL | AVOID
    overall_confidence: float  # 0-100
    asset: str
    model_id: str
    entry_price: float
    horizon: int
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(),
    )


# ======================================================================
# Public API
# ======================================================================

def build_action_plan(
    price_paths: np.ndarray,
    dates: List,
    quantiles: List[float],
    scenarios: ScenarioAnalysis,
    *,
    horizon: int,
    take_profit_pct: float = 5.0,
    stop_loss_pct: float = 3.0,
    min_expected_return_pct: float = 1.0,
    risk_aversion_lambda: float = 0.5,
    entry_price: float,
    df: Optional[pd.DataFrame] = None,
    model_id: str = "",
    asset: str = "",
) -> ActionPlan:
    """Build a complete time-based action plan.

    Parameters
    ----------
    price_paths : ndarray ``(K+1, Q)``
        Row 0 = entry price; rows 1..K = forecast prices.
    dates : list
        Future dates (length ≥ *horizon*).
    quantiles : list[float]
        Quantile levels (e.g. ``[0.1, 0.5, 0.9]``).
    scenarios : ScenarioAnalysis
        Pre-computed scenario analysis.
    horizon : int
        Planning horizon in days.
    take_profit_pct, stop_loss_pct : float
        Risk thresholds in percent.
    min_expected_return_pct : float
        Minimum median return to consider a BUY (%).
    risk_aversion_lambda : float
        Downside penalty for risk-adjusted score.
    entry_price : float
        Last known close price.
    df : DataFrame, optional
        Market data with technical indicators for rationale.
    model_id, asset : str
        Metadata for persistence.

    Returns
    -------
    ActionPlan
    """
    K = price_paths.shape[0] - 1
    h = min(max(horizon, 1), K)

    # Resolve quantile column indices
    q_map = {round(q, 2): i for i, q in enumerate(quantiles)}
    idx = {
        "p10": q_map.get(0.1, 0),
        "p50": q_map.get(0.5, 1),
        "p90": q_map.get(0.9, 2),
    }

    # Cumulative returns relative to entry price
    r10 = (price_paths[1:h + 1, idx["p10"]] - entry_price) / entry_price
    r50 = (price_paths[1:h + 1, idx["p50"]] - entry_price) / entry_price
    r90 = (price_paths[1:h + 1, idx["p90"]] - entry_price) / entry_price

    tp_frac = take_profit_pct / 100.0
    sl_frac = stop_loss_pct / 100.0
    lam = risk_aversion_lambda

    # Risk-adjusted scores
    risk_scores = r50 - lam * np.maximum(0.0, -r10)

    # ── Classify each day ────────────────────────────────────────────
    day_actions = _classify_days(
        price_paths, dates, idx, h,
        r10, r50, r90, risk_scores,
        tp_frac, sl_frac, entry_price,
    )

    # ── Entry window ─────────────────────────────────────────────────
    entry_window = _find_entry_window(day_actions, risk_scores)

    # ── Best exit ────────────────────────────────────────────────────
    best_exit = _find_best_exit(r50, risk_scores, dates, h)

    # ── Decision rationale ───────────────────────────────────────────
    rationale = _build_rationale(
        df, r50, r10, r90, risk_scores, h, sl_frac, entry_window,
    )

    # ── Overall signal ───────────────────────────────────────────────
    overall_signal, overall_conf = _compute_overall_signal(
        day_actions, r50, r10, h,
        min_expected_return_pct / 100.0,
    )

    # ── Narrative ────────────────────────────────────────────────────
    narrative = _build_narrative(day_actions, best_exit)

    params = {
        "take_profit_pct": take_profit_pct,
        "stop_loss_pct": stop_loss_pct,
        "min_expected_return_pct": min_expected_return_pct,
        "risk_aversion_lambda": risk_aversion_lambda,
    }

    return ActionPlan(
        daily_actions=day_actions,
        entry_window=entry_window,
        best_exit=best_exit,
        scenarios=scenarios,
        rationale=rationale,
        narrative=narrative,
        overall_signal=overall_signal,
        overall_confidence=round(overall_conf, 1),
        asset=asset,
        model_id=model_id,
        entry_price=entry_price,
        horizon=h,
        params=params,
    )


def summarize_action_plan(plan: ActionPlan) -> Dict[str, Any]:
    """Create a JSON-serialisable summary of the action plan."""
    daily = []
    for d in plan.daily_actions:
        daily.append({
            "day": d.day,
            "date": d.date,
            "action": d.action,
            "confidence": d.confidence,
            "rationale": d.rationale,
            "price_p10": d.price_p10,
            "price_p50": d.price_p50,
            "price_p90": d.price_p90,
            "ret_p10_pct": round(d.ret_p10 * 100, 4),
            "ret_p50_pct": round(d.ret_p50 * 100, 4),
            "ret_p90_pct": round(d.ret_p90 * 100, 4),
            "uncertainty_width_pct": round(d.uncertainty_width * 100, 4),
            "risk_score": round(d.risk_score, 4),
        })

    result: Dict[str, Any] = {
        "overall_signal": plan.overall_signal,
        "overall_confidence": plan.overall_confidence,
        "narrative": plan.narrative,
        "asset": plan.asset,
        "model_id": plan.model_id,
        "entry_price": plan.entry_price,
        "horizon": plan.horizon,
        "params": plan.params,
        "timestamp": plan.timestamp,
        "rationale": {
            "trend": plan.rationale.trend_confirmation,
            "volatility": plan.rationale.volatility_regime,
            "quantile_risk": plan.rationale.quantile_risk,
            "today": plan.rationale.today_assessment,
        },
        "scenarios": {
            "optimistic": {
                "return_pct": plan.scenarios.optimistic.return_pct,
                "final_price": plan.scenarios.optimistic.final_price,
                "value_impact": plan.scenarios.optimistic.value_impact,
            },
            "base": {
                "return_pct": plan.scenarios.base.return_pct,
                "final_price": plan.scenarios.base.final_price,
                "value_impact": plan.scenarios.base.value_impact,
            },
            "pessimistic": {
                "return_pct": plan.scenarios.pessimistic.return_pct,
                "final_price": plan.scenarios.pessimistic.final_price,
                "value_impact": plan.scenarios.pessimistic.value_impact,
            },
        },
        "daily_plan": daily,
    }

    if plan.entry_window:
        result["entry_window"] = {
            "start_day": plan.entry_window.start_day,
            "end_day": plan.entry_window.end_day,
            "avg_score": plan.entry_window.avg_score,
            "rationale": plan.entry_window.rationale,
        }

    if plan.best_exit:
        result["best_exit"] = {
            "day": plan.best_exit.day,
            "date": plan.best_exit.date,
            "expected_return_pct": plan.best_exit.expected_return_pct,
            "risk_score": plan.best_exit.risk_score,
            "rationale": plan.best_exit.rationale,
        }

    return result


# ======================================================================
# Private helpers — day classification (state machine)
# ======================================================================

def _classify_days(
    price_paths: np.ndarray,
    dates: List,
    idx: Dict[str, int],
    horizon: int,
    r10: np.ndarray,
    r50: np.ndarray,
    r90: np.ndarray,
    risk_scores: np.ndarray,
    tp_frac: float,
    sl_frac: float,
    entry_price: float,
) -> List[DayRecommendation]:
    """Classify each day using a state-machine approach.

    States: NO_POSITION → (BUY) → POSITIONED → (SELL) → CLOSED.
    """
    actions: List[DayRecommendation] = []
    state = "NO_POSITION"
    peak_score = float("-inf")

    for t in range(horizon):
        day = t + 1
        date_str = (
            str(dates[t].date()) if hasattr(dates[t], "date")
            else str(dates[t])
        )

        p10 = float(price_paths[t + 1, idx["p10"]])
        p50 = float(price_paths[t + 1, idx["p50"]])
        p90 = float(price_paths[t + 1, idx["p90"]])
        ret10 = float(r10[t])
        ret50 = float(r50[t])
        ret90 = float(r90[t])
        uw = ret90 - ret10
        score = float(risk_scores[t])

        if state == "NO_POSITION":
            action, conf, reason = _classify_no_position(
                t, ret50, ret10, uw, score, risk_scores, sl_frac,
            )
            if action == "BUY":
                state = "POSITIONED"
                peak_score = score

        elif state == "POSITIONED":
            peak_score = max(peak_score, score)
            action, conf, reason = _classify_positioned(
                t, ret50, ret10, score, peak_score, risk_scores,
                tp_frac, sl_frac,
            )
            if action == "SELL":
                state = "CLOSED"

        else:  # CLOSED
            action = "AVOID"
            conf = 0.0
            reason = "Position closed"

        actions.append(DayRecommendation(
            day=day, date=date_str,
            action=action,
            confidence=round(conf, 1),
            rationale=reason,
            price_p10=round(p10, 2),
            price_p50=round(p50, 2),
            price_p90=round(p90, 2),
            ret_p10=ret10, ret_p50=ret50, ret_p90=ret90,
            uncertainty_width=uw,
            risk_score=score,
        ))

    return actions


def _classify_no_position(
    t: int,
    ret_p50: float,
    ret_p10: float,
    uw: float,
    score: float,
    all_scores: np.ndarray,
    sl_frac: float,
) -> Tuple[str, float, str]:
    """Classify when no position is held."""
    # Negative expected return → AVOID
    if ret_p50 < 0:
        return "AVOID", 70.0, f"Negative expected return ({ret_p50:+.2%})"

    # Excessive downside risk → AVOID
    if ret_p10 < -sl_frac:
        return "AVOID", 75.0, (
            f"Downside risk exceeds stop-loss "
            f"(P10 {ret_p10:+.2%} < -{sl_frac:.2%})"
        )

    # Excessive uncertainty → AVOID
    if uw > 0.10:
        return "AVOID", 65.0, f"Excessive uncertainty (spread {uw:.2%})"

    # Check risk-adjusted outlook
    improving = t == 0 or score >= float(all_scores[t - 1])

    if score > 0 and ret_p50 > 0 and ret_p10 > -sl_frac * 0.7:
        conf = min(90.0, 50.0 + abs(score) * 300)
        if improving:
            return "BUY", conf, (
                f"Favourable entry — return {ret_p50:+.2%}, "
                f"rising risk-adjusted outlook"
            )
        else:
            return "BUY", conf * 0.8, (
                f"Favourable entry — return {ret_p50:+.2%}"
            )

    # Mildly positive but not convincing → HOLD (wait)
    if ret_p50 > 0:
        return "HOLD", 45.0, (
            f"Mildly positive ({ret_p50:+.2%}) — "
            f"waiting for clearer signal"
        )

    return "AVOID", 60.0, "Conditions not favourable for entry"


def _classify_positioned(
    t: int,
    ret_p50: float,
    ret_p10: float,
    score: float,
    peak_score: float,
    all_scores: np.ndarray,
    tp_frac: float,
    sl_frac: float,
) -> Tuple[str, float, str]:
    """Classify when a position is held."""
    # Take-profit
    if ret_p50 >= tp_frac:
        return "SELL", 90.0, (
            f"Take-profit — return {ret_p50:+.2%} ≥ {tp_frac:+.2%}"
        )

    # Stop-loss
    if ret_p10 <= -sl_frac:
        return "SELL", 85.0, (
            f"Stop-loss — P10 return {ret_p10:+.2%} ≤ {-sl_frac:+.2%}"
        )

    # Significant score decline from peak → exit
    if (peak_score > 0
            and score < peak_score * 0.2
            and t > 0
            and score < float(all_scores[t - 1])):
        return "SELL", 65.0, "Momentum fading — risk-adjusted score declining"

    # Normal hold
    conf = min(80.0, 50.0 + max(0, score * 200))
    reason = f"Within limits — return {ret_p50:+.2%}"
    if peak_score > 0 and abs(score - peak_score) < 0.001:
        reason += " ★ peak"
    return "HOLD", conf, reason


# ======================================================================
# Private helpers — entry window, exit, rationale
# ======================================================================

def _find_entry_window(
    actions: List[DayRecommendation],
    risk_scores: np.ndarray,
) -> Optional[EntryWindow]:
    """Find the best contiguous range of BUY-classified days."""
    buy_runs: List[List[int]] = []
    current_run: List[int] = []

    for d in actions:
        if d.action == "BUY":
            current_run.append(d.day)
        else:
            if current_run:
                buy_runs.append(current_run)
            current_run = []
    if current_run:
        buy_runs.append(current_run)

    if not buy_runs:
        return None

    # Pick the run with the highest average risk-adjusted score
    best_run = None
    best_avg = float("-inf")
    for run in buy_runs:
        scores = [float(risk_scores[d - 1]) for d in run]
        avg = sum(scores) / len(scores)
        if avg > best_avg:
            best_avg = avg
            best_run = run

    if best_run is None:
        return None

    return EntryWindow(
        start_day=best_run[0],
        end_day=best_run[-1],
        avg_score=round(best_avg, 4),
        rationale=(
            f"Days {best_run[0]}–{best_run[-1]}: "
            f"favourable risk-adjusted outlook (avg score {best_avg:+.4f})"
        ),
    )


def _find_best_exit(
    r50: np.ndarray,
    risk_scores: np.ndarray,
    dates: List,
    horizon: int,
) -> Optional[ExitPoint]:
    """Find the optimal exit day based on risk-adjusted score."""
    if horizon == 0:
        return None

    best_idx = int(np.argmax(risk_scores[:horizon]))
    best_day = best_idx + 1
    date_str = (
        str(dates[best_idx].date())
        if hasattr(dates[best_idx], "date")
        else str(dates[best_idx])
    )

    return ExitPoint(
        day=best_day,
        date=date_str,
        expected_return_pct=round(float(r50[best_idx]) * 100, 2),
        risk_score=round(float(risk_scores[best_idx]), 4),
        rationale=(
            f"Day {best_day}: peak risk-adjusted score "
            f"({float(risk_scores[best_idx]):+.4f}), "
            f"expected return {float(r50[best_idx]):+.2%}"
        ),
    )


def _build_rationale(
    df: Optional[pd.DataFrame],
    r50: np.ndarray,
    r10: np.ndarray,
    r90: np.ndarray,
    risk_scores: np.ndarray,
    horizon: int,
    sl_frac: float,
    entry_window: Optional[EntryWindow],
) -> DecisionRationale:
    """Build multi-factor decision rationale from market data + forecast."""
    # ── Trend confirmation ───────────────────────────────────────────
    trend = "No market data available"
    if df is not None and not df.empty and len(df) >= 20:
        last = df.iloc[-1]
        close = float(last.get("Close", 0))
        has50 = "sma_50" in df.columns
        has200 = "sma_200" in df.columns

        if has50 and has200:
            s50 = float(last["sma_50"])
            s200 = float(last["sma_200"])
            if close > s200 and s50 > s200:
                trend = (
                    f"Bullish — price (${close:.2f}) > SMA-200 (${s200:.2f}) "
                    f"and SMA-50 > SMA-200 (golden cross)"
                )
            elif close > s200:
                trend = (
                    f"Moderately bullish — price > SMA-200 "
                    f"but SMA-50 < SMA-200"
                )
            elif close < s200 and s50 < s200:
                trend = (
                    f"Bearish — price (${close:.2f}) < SMA-200 (${s200:.2f}) "
                    f"and SMA-50 < SMA-200 (death cross)"
                )
            else:
                trend = (
                    f"Mixed — price < SMA-200 but SMA-50 > SMA-200"
                )
        elif has200:
            s200 = float(last["sma_200"])
            trend = (
                f"Price {'above' if close > s200 else 'below'} "
                f"SMA-200 (${s200:.2f})"
            )

    # ── Volatility regime ────────────────────────────────────────────
    vol = "No volatility data"
    if df is not None and not df.empty and "atr_pct" in df.columns:
        atr = float(df["atr_pct"].iloc[-1])
        if atr < 0.01:
            vol = f"Low volatility (ATR% = {atr:.4f}) — favourable"
        elif atr < 0.025:
            vol = f"Normal volatility (ATR% = {atr:.4f})"
        else:
            vol = f"High volatility (ATR% = {atr:.4f}) — elevated risk"

    # ── Quantile risk assessment ─────────────────────────────────────
    min_down = float(np.min(r10[:horizon]))
    max_spread = float(np.max(r90[:horizon] - r10[:horizon]))
    if min_down > -sl_frac * 0.5:
        q_risk = (
            f"Low risk — worst P10 drawdown {min_down:+.2%}, "
            f"well within stop-loss"
        )
    elif min_down > -sl_frac:
        q_risk = (
            f"Moderate risk — P10 drawdown {min_down:+.2%} "
            f"approaches stop-loss ({-sl_frac:+.2%})"
        )
    else:
        q_risk = (
            f"High risk — P10 drawdown {min_down:+.2%} "
            f"exceeds stop-loss ({-sl_frac:+.2%})"
        )
    q_risk += f". Max prediction spread: {max_spread:.2%}"

    # ── Today assessment ─────────────────────────────────────────────
    s1 = float(risk_scores[0])
    best_s = float(np.max(risk_scores[:horizon]))
    best_d = int(np.argmax(risk_scores[:horizon])) + 1
    r1 = float(r50[0])

    if entry_window and entry_window.start_day == 1:
        today = (
            f"Today is within the entry window "
            f"(days {entry_window.start_day}–{entry_window.end_day})"
        )
    elif s1 < best_s * 0.7 and best_d > 1:
        today = (
            f"Today's score ({s1:+.4f}) is below "
            f"the peak on day {best_d} ({best_s:+.4f}) — "
            f"waiting may yield a better entry"
        )
    elif r1 < 0:
        today = (
            f"Negative expected return on day 1 ({r1:+.2%}) — "
            f"not optimal for entry"
        )
    else:
        today = (
            f"Acceptable for entry (score {s1:+.4f}, "
            f"return {r1:+.2%})"
        )

    return DecisionRationale(
        trend_confirmation=trend,
        volatility_regime=vol,
        quantile_risk=q_risk,
        today_assessment=today,
    )


def _compute_overall_signal(
    actions: List[DayRecommendation],
    r50: np.ndarray,
    r10: np.ndarray,
    horizon: int,
    mer_frac: float,
) -> Tuple[str, float]:
    """Derive the aggregate signal from the daily plan."""
    if not actions:
        return "AVOID", 0.0

    counts: Dict[str, int] = {"BUY": 0, "HOLD": 0, "SELL": 0, "AVOID": 0}
    for a in actions:
        counts[a.action] = counts.get(a.action, 0) + 1

    final_ret = float(r50[horizon - 1])
    min_down = float(np.min(r10[:horizon]))

    # Mostly unfavourable
    if final_ret < mer_frac and counts["BUY"] == 0:
        return "AVOID", min(80.0, 30.0 + counts["AVOID"] / horizon * 50)

    # Has entry opportunities with decent return
    if counts["BUY"] > 0 and final_ret >= mer_frac:
        conf = min(90.0, 50.0 + final_ret * 300 + counts["BUY"] / horizon * 20)
        return "BUY", conf

    # Sell signals dominate
    if counts["SELL"] > 0 and counts["BUY"] == 0:
        return "SELL", 70.0

    # Has some positive days
    if final_ret >= mer_frac:
        return "HOLD", 55.0

    return "HOLD", 50.0


def _build_narrative(
    actions: List[DayRecommendation],
    best_exit: Optional[ExitPoint],
) -> str:
    """Build a one-line narrative: 'Buy today → hold → sell on day 6'."""
    if not actions:
        return "No action plan available"

    first_buy = None
    sell_day = None
    for d in actions:
        if d.action == "BUY" and first_buy is None:
            first_buy = d.day
        if d.action == "SELL":
            sell_day = d.day
            break

    if first_buy is None:
        return "Avoid entry — conditions not favourable"

    buy_label = "Buy today" if first_buy == 1 else f"Wait → buy on day {first_buy}"

    if sell_day:
        return f"{buy_label} → hold → sell on day {sell_day}"

    if best_exit:
        return f"{buy_label} → hold → best exit on day {best_exit.day}"

    return f"{buy_label} → hold through horizon"
