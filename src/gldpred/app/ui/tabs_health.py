"""Model Health tab — staleness, prediction accuracy, and recalibration advice.

Renders a per-model health dashboard showing freshness badges, trade-log
accuracy, training quality, and actionable recommendations.
"""
from __future__ import annotations

from typing import Dict, List

import streamlit as st

from gldpred.services.health_service import HealthService, ModelHealthReport


# ── colour / emoji maps ─────────────────────────────────────────────────
_STALENESS_EMOJI = {
    "fresh": "🟢",
    "aging": "🟡",
    "stale": "🔴",
    "expired": "⚫",
}

_VERDICT_EMOJI = {
    "healthy": "✅",
    "overfitting": "⚠️",
    "underfitting": "⚠️",
    "noisy": "⚠️",
    "unknown": "❓",
}


def render(t: Dict[str, str], lang: str = "en") -> None:
    """Render the Model Health tab."""
    st.header(t.get("health_header", "Model Health & Accountability"))
    st.caption(t.get("health_subtitle", "Monitor model freshness, prediction accuracy, and recalibration needs."))

    svc = HealthService()
    reports = svc.report_all_models()

    if not reports:
        st.info(t.get("health_no_models", "No models found in the registry. Train a model first."))
        return

    # ── Summary metrics row ──────────────────────────────────────────
    assigned_reports = [r for r in reports if r.is_primary]
    stale_count = sum(1 for r in assigned_reports if r.staleness in ("stale", "expired"))
    total_closed = sum(r.closed_trades for r in reports)
    avg_win = 0.0
    if total_closed > 0:
        total_wins = sum(
            r.closed_trades * r.win_rate / 100 for r in reports if r.closed_trades > 0
        )
        avg_win = total_wins / total_closed * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t.get("health_total_models", "Total Models"), len(reports))
    c2.metric(t.get("health_assigned", "Assigned (Primary)"), len(assigned_reports))
    c3.metric(
        t.get("health_stale_alert", "Stale / Expired"),
        stale_count,
        delta=None if stale_count == 0 else f"-{stale_count}",
        delta_color="inverse",
    )
    c4.metric(
        t.get("health_avg_win_rate", "Avg Win Rate"),
        f"{avg_win:.0f}%" if total_closed > 0 else "N/A",
    )

    st.divider()

    # ── Per-model cards ──────────────────────────────────────────────
    for report in reports:
        _render_model_card(report, t)


def _render_model_card(r: ModelHealthReport, t: Dict[str, str]) -> None:
    """Render a single model health card as an expander."""
    emoji = _STALENESS_EMOJI.get(r.staleness, "❓")
    primary_badge = " ⭐" if r.is_primary else ""
    title = f"{emoji} {r.label} ({r.architecture}){primary_badge}"

    with st.expander(title, expanded=r.is_primary):
        # Row 1: identity & freshness
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**{t.get('health_asset', 'Asset')}:** {r.asset}")
        col2.markdown(
            f"**{t.get('health_age', 'Age')}:** {r.age_days} "
            f"{t.get('health_days', 'days')}"
        )
        col3.markdown(
            f"**{t.get('health_freshness', 'Freshness')}:** "
            f"{emoji} {t.get(f'health_status_{r.staleness}', r.staleness.title())}"
        )

        # Row 2: training quality
        verdict_emoji = _VERDICT_EMOJI.get(r.training_verdict, "❓")
        col4, col5, col6 = st.columns(3)
        col4.markdown(
            f"**{t.get('health_training', 'Training')}:** "
            f"{verdict_emoji} {r.training_verdict.title()}"
        )
        col5.markdown(
            f"**{t.get('health_epochs', 'Epochs')}:** {r.total_epochs}"
        )
        col6.markdown(
            f"**{t.get('health_best_loss', 'Best Val Loss')}:** "
            f"{r.best_val_loss:.4f}" if r.best_val_loss > 0 else
            f"**{t.get('health_best_loss', 'Best Val Loss')}:** N/A"
        )

        st.divider()

        # Row 3: prediction accuracy
        st.markdown(f"##### {t.get('health_accuracy_header', 'Prediction Accuracy')}")
        if r.closed_trades > 0:
            ca, cb, cc, cd = st.columns(4)
            ca.metric(t.get("health_trades_total", "Trades"), r.total_trades)
            cb.metric(t.get("health_trades_closed", "Closed"), r.closed_trades)
            cc.metric(t.get("health_win_rate_label", "Win Rate"), f"{r.win_rate:.0f}%")

            bias_str = f"{r.prediction_bias:+.1f}pp"
            cd.metric(
                t.get("health_bias", "Pred. Bias"),
                bias_str,
            )

            # Predicted vs actual mini-summary
            st.markdown(
                f"📊 {t.get('health_avg_predicted', 'Avg predicted return')}: "
                f"**{r.avg_predicted_return:+.2f}%** → "
                f"{t.get('health_avg_actual', 'Avg actual')}: "
                f"**{r.avg_actual_return:+.2f}%**"
            )
        else:
            st.info(
                t.get(
                    "health_no_closed",
                    "No closed trades for this model yet. "
                    "Archive recommendations in the Portfolio tab and close "
                    "them with actual outcomes to see accuracy here.",
                )
            )

        st.divider()

        # Row 4: recommendations
        st.markdown(f"##### {t.get('health_recs_header', 'Recommendations')}")
        for rec in r.recommendations:
            st.markdown(f"• {rec}")
