"""Dashboard tab — the primary landing page.

Shows a quick investment decision board: one-line signal per asset
with a leaderboard ranked by expected return.
"""
from __future__ import annotations

import traceback
from datetime import datetime
from typing import Dict

import pandas as pd
import streamlit as st

from gldpred.app import state
from gldpred.app.components.empty_states import show_empty_no_model
from gldpred.app.components.walkthrough import render_walkthrough_banner
from gldpred.app.controllers.dashboard_controller import (
    DashboardAssetResult,
    DashboardResult,
    run_dashboard_analysis,
)
from gldpred.registry import ModelAssignments, ModelRegistry


# Colour helpers
_SIGNAL_COLOURS = {
    "BUY": "green",
    "HOLD": "orange",
    "SELL": "red",
    "AVOID": "gray",
}

_SIGNAL_EMOJI = {
    "BUY": "🟢",
    "HOLD": "🟡",
    "SELL": "🔴",
    "AVOID": "⚫",
}


def render(t: Dict[str, str]) -> None:
    """Render the Dashboard tab."""
    render_walkthrough_banner(t, "tab_dashboard")
    st.header(t["dash_header"])
    st.caption(t["dash_subtitle"])

    # Quick params
    c1, c2 = st.columns(2)
    investment = c1.number_input(
        t["dash_investment_label"],
        min_value=0.01,
        max_value=1_000_000.0,
        value=10_000.0,
        step=500.0,
        key="dash_investment",
    )
    horizon = c2.slider(
        t["dash_horizon_label"],
        min_value=5,
        max_value=60,
        value=20,
        key="dash_horizon",
    )

    # Check if any models exist at all
    registry = ModelRegistry()
    total = len(registry.list_models())
    if total == 0:
        show_empty_no_model(t)
        return

    # Run analysis
    if st.button(t["dash_run_analysis"], type="primary"):
        with st.spinner(t["dash_analysing_all"]):
            try:
                result = run_dashboard_analysis(
                    investment=investment,
                    horizon=horizon,
                )
                st.session_state["_dashboard_result"] = result
            except Exception as exc:
                st.error(str(exc))
                st.code(traceback.format_exc())
                return

    result: DashboardResult | None = st.session_state.get("_dashboard_result")
    if result is None:
        st.info(t["dash_click_run"])
        return

    if not result.items:
        st.warning(t["dash_no_models"])
        return

    # ── Featured asset (top ranked) ──────────────────────────────
    best = result.items[0]
    if best.error is None:
        _render_hero_card(best, t, investment)

    # ── Leaderboard ──────────────────────────────────────────────
    st.subheader(t["dash_leaderboard"])
    _render_leaderboard(result, t, investment)

    # ── Per-asset expandable details ─────────────────────────────
    for item in result.items:
        if item.error:
            st.warning(f"{item.asset}: {item.error}")
            continue
        _render_asset_expander(item, t, investment)

    # Timestamp
    st.caption(f"{t['dash_last_update']}: {datetime.now().strftime('%H:%M:%S')}")


# ======================================================================
# Private renderers
# ======================================================================

def _render_hero_card(item: DashboardAssetResult, t: Dict, inv: float) -> None:
    """Large signal card for the top-ranked asset."""
    emoji = _SIGNAL_EMOJI.get(item.signal, "⚫")
    colour = _SIGNAL_COLOURS.get(item.signal, "gray")

    st.markdown(
        f"### {emoji} **{item.asset}** — "
        f":{colour}[**{item.signal}**]  "
        f"({t['dash_confidence']}: {item.confidence:.0f}%)"
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric(
        t["dash_expected_return"],
        f"{item.expected_return_pct:+.2f}%",
    )
    mc2.metric(
        t["dash_max_risk"],
        f"{item.max_risk_pct:+.2f}%",
    )
    pnl_sign = "+" if item.pnl_median >= 0 else ""
    mc3.metric(
        t["dash_pnl"],
        f"{pnl_sign}${item.pnl_median:,.2f}",
    )
    mc4.metric(t["dash_model_label"], item.model_label)

    info_cols = st.columns(2)
    if item.entry_window:
        info_cols[0].success(f"📅 {t['dash_entry_window']}: {item.entry_window}")
    if item.best_exit:
        info_cols[1].info(f"🎯 {t['dash_best_exit']}: {item.best_exit}")

    st.divider()


def _render_leaderboard(result: DashboardResult, t: Dict, inv: float) -> None:
    """Tabular leaderboard of all analysed assets."""
    rows = []
    for i, item in enumerate(result.items):
        if item.error:
            continue
        emoji = _SIGNAL_EMOJI.get(item.signal, "⚫")
        rows.append({
            t["dash_rank"]: i + 1,
            t["dash_asset"]: item.asset,
            t["dash_signal"]: f"{emoji} {item.signal}",
            t["dash_confidence"]: f"{item.confidence:.0f}%",
            t["dash_expected_return"]: f"{item.expected_return_pct:+.2f}%",
            t["dash_max_risk"]: f"{item.max_risk_pct:+.2f}%",
            t["dash_pnl"]: f"${item.pnl_median:+,.2f}",
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


def _render_asset_expander(
    item: DashboardAssetResult, t: Dict, inv: float,
) -> None:
    """Expandable detail card per asset."""
    emoji = _SIGNAL_EMOJI.get(item.signal, "⚫")
    header = f"{emoji} {item.asset} — {item.signal} ({item.confidence:.0f}%)"

    with st.expander(header, expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric(t["dash_expected_return"], f"{item.expected_return_pct:+.2f}%")
        c2.metric(t["dash_max_risk"], f"{item.max_risk_pct:+.2f}%")
        pnl_sign = "+" if item.pnl_median >= 0 else ""
        c3.metric(t["dash_pnl"], f"{pnl_sign}${item.pnl_median:,.2f}")

        if item.entry_window:
            st.success(f"📅 {t['dash_entry_window']}: {item.entry_window}")
        if item.best_exit:
            st.info(f"🎯 {t['dash_best_exit']}: {item.best_exit}")

        if item.plan and item.plan.narrative:
            st.markdown(f"*{item.plan.narrative}*")

        st.caption(f"{t['dash_model_label']}: {item.model_label}")
