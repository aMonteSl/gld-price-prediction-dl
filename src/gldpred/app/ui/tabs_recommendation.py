"""Recommendation tab rendering — action-plan based."""
from __future__ import annotations

import traceback
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gldpred.app import state
from gldpred.app.controllers.model_loader import auto_select_model, get_active_model
from gldpred.app.controllers.trade_plan_controller import generate_action_plan
from gldpred.app.glossary import info_term
from gldpred.decision.action_planner import ActionPlan


# ── Colour palette for actions ──────────────────────────────────────
_ACTION_COLOURS = {
    "BUY": ("#27ae60", "green"),
    "HOLD": ("#f39c12", "orange"),
    "SELL": ("#e74c3c", "red"),
    "AVOID": ("#7f8c8d", "gray"),
}


def render(t: Dict[str, str], lang: str) -> None:
    """Render the Recommendation tab with full action-plan UI."""
    st.header(t["reco_header"])
    st.markdown(t["reco_disclaimer"])
    st.info(t["ap_info"])

    forecast = state.get(state.KEY_FORECAST)
    df = state.get(state.KEY_RAW_DF)
    asset = state.get(state.KEY_ASSET, "GLD")

    # Model availability check
    model = get_active_model()
    if model is None:
        model = auto_select_model(asset)
    if model is None:
        st.warning(t["reco_warn_no_model"])
        return
    if forecast is None:
        st.warning(t["ap_no_forecast"])
        return

    try:
        # Read sidebar parameters
        horizon = st.session_state.get("tp_horizon", 10)
        tp_pct = st.session_state.get("tp_take_profit", 5.0)
        sl_pct = st.session_state.get("tp_stop_loss", 3.0)
        mer_pct = st.session_state.get("tp_min_return", 1.0)
        lam = st.session_state.get("tp_risk_aversion", 0.5)
        investment = st.session_state.get("tp_investment", 10_000.0)

        # Clamp horizon to available forecast days
        K = forecast.returns_quantiles.shape[0]
        horizon = min(horizon, K)

        # Generate button
        if st.button(t["ap_generate"], type="primary"):
            plan = generate_action_plan(
                forecast,
                horizon=horizon,
                take_profit_pct=tp_pct,
                stop_loss_pct=sl_pct,
                min_expected_return_pct=mer_pct,
                risk_aversion_lambda=lam,
                investment=investment,
                model_id=getattr(model, "model_id", ""),
                asset=asset,
                df=df,
            )
            st.session_state["_action_plan"] = plan
            st.toast(t["ap_plan_saved"])

        plan: ActionPlan | None = st.session_state.get("_action_plan")
        if plan is None:
            st.caption(
                "Press **" + t["ap_generate"] + "** to create an action plan."
                if lang == "en"
                else "Pulsa **" + t["ap_generate"] + "** para crear un plan."
            )
            return

        # ── Overall signal badge ─────────────────────────────────────
        _render_signal_badge(plan, t)

        # ── Decision rationale ───────────────────────────────────────
        _render_rationale(plan, t)

        # ── Scenario analysis ────────────────────────────────────────
        _render_scenarios(plan, t)

        # ── Entry & exit optimization ────────────────────────────────
        _render_entry_exit(plan, t, lang)

        # ── Daily timeline ───────────────────────────────────────────
        _render_timeline(plan, t, lang)

        # ── Chart ────────────────────────────────────────────────────
        st.subheader(t["ap_chart_title"])
        fig = _create_action_plan_chart(plan, forecast, t, lang)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.error(t["reco_error"].format(err=exc))
        st.code(traceback.format_exc())


# ======================================================================
# Private helpers
# ======================================================================

def _render_signal_badge(plan: ActionPlan, t: Dict[str, str]) -> None:
    """Large signal badge + narrative summary."""
    signal = plan.overall_signal
    badge_key = f"ap_signal_{signal.lower()}"
    badge = t.get(badge_key, signal)
    colour = _ACTION_COLOURS.get(signal, ("#7f8c8d", "gray"))[1]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            f"### {t['ap_overall_signal']}: :{colour}[**{badge}**]"
        )
        if plan.narrative:
            st.markdown(f"*{plan.narrative}*")
    with col2:
        st.metric(t["ap_confidence"], f"{plan.overall_confidence:.0f}%")


def _render_rationale(plan: ActionPlan, t: Dict[str, str]) -> None:
    """Expandable decision rationale with 4 factors."""
    r = plan.rationale
    with st.expander(t["ap_rationale_header"], expanded=False):
        st.markdown(f"**{t['ap_trend']}:** {r.trend_confirmation}")
        st.markdown(f"**{t['ap_volatility']}:** {r.volatility_regime}")
        st.markdown(f"**{t['ap_quantile_risk']}:** {r.quantile_risk}")
        st.markdown(f"**{t['ap_today']}:** {r.today_assessment}")


def _render_scenarios(plan: ActionPlan, t: Dict[str, str]) -> None:
    """Three-card scenario analysis (Pessimistic / Base / Optimistic)."""
    st.subheader(t["ap_scenarios_header"])
    sc = plan.scenarios

    c1, c2, c3 = st.columns(3)

    for col, scenario, label_key in [
        (c1, sc.pessimistic, "ap_scenario_pessimistic"),
        (c2, sc.base, "ap_scenario_base"),
        (c3, sc.optimistic, "ap_scenario_optimistic"),
    ]:
        with col:
            st.markdown(f"**{t[label_key]}**")
            ret_sign = "+" if scenario.return_pct >= 0 else ""
            st.metric(
                t["ap_return"],
                f"{ret_sign}{scenario.return_pct:.2f}%",
            )
            st.metric(t["ap_final_price"], f"${scenario.final_price:,.2f}")
            pnl_sign = "+" if scenario.value_impact >= 0 else ""
            inv_label = t["ap_investment_label"].format(
                amount=f"${sc.investment:,.0f}"
            )
            st.metric(
                f"{t['ap_pnl']} ({inv_label})",
                f"{pnl_sign}${scenario.value_impact:,.2f}",
            )


def _render_entry_exit(
    plan: ActionPlan, t: Dict[str, str], lang: str,
) -> None:
    """Entry window and best exit metrics."""
    st.subheader(t["ap_entry_exit_header"])
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**{t['ap_entry_window']}**")
        if plan.entry_window:
            ew = plan.entry_window
            st.success(
                f"Days {ew.start_day}–{ew.end_day}  |  "
                f"Avg score: {ew.avg_score:+.4f}"
            )
            st.caption(ew.rationale)
        else:
            st.warning(t["ap_no_entry"])

    with c2:
        st.markdown(f"**{t['ap_best_exit']}**")
        if plan.best_exit:
            bx = plan.best_exit
            ret_sign = "+" if bx.expected_return_pct >= 0 else ""
            st.success(
                f"Day {bx.day} ({bx.date})  |  "
                f"Return: {ret_sign}{bx.expected_return_pct:.2f}%"
            )
            st.caption(bx.rationale)
        else:
            st.info("—")


def _render_timeline(
    plan: ActionPlan, t: Dict[str, str], lang: str,
) -> None:
    """Colour-coded daily timeline + expandable per-day details."""
    st.subheader(t["ap_timeline_header"])

    actions = plan.daily_actions
    if not actions:
        return

    # ── Colour-coded row of day boxes ────────────────────────────────
    n = len(actions)
    # Use up to 20 columns; if more days, group
    max_cols = min(n, 20)
    cols = st.columns(max_cols)
    for i, d in enumerate(actions[:max_cols]):
        hex_col = _ACTION_COLOURS.get(d.action, ("#7f8c8d", "gray"))[0]
        with cols[i]:
            st.markdown(
                f"<div style='background-color:{hex_col};color:white;"
                f"text-align:center;border-radius:6px;padding:4px 2px;"
                f"font-size:0.75rem;margin-bottom:2px'>"
                f"D{d.day}<br><b>{d.action}</b></div>",
                unsafe_allow_html=True,
            )

    # ── Expandable per-day details ───────────────────────────────────
    for d in actions:
        action_label = t.get(f"ap_signal_{d.action.lower()}", d.action)
        header = t["ap_day_details"].format(day=d.day, action=action_label)
        with st.expander(header, expanded=False):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(t["ap_col_price"], f"${d.price_p50:,.2f}")
            mc2.metric(t["ap_col_ret"], f"{d.ret_p50:+.2%}")
            mc3.metric(t["ap_col_risk"], f"{d.risk_score:+.4f}")
            st.caption(
                f"P10 ${d.price_p10:,.2f} → P90 ${d.price_p90:,.2f} "
                f"| {t['ap_confidence']}: {d.confidence:.0f}%"
            )
            st.markdown(f"_{d.rationale}_")


def _create_action_plan_chart(
    plan: ActionPlan,
    forecast,
    t: Dict[str, str],
    lang: str,
) -> go.Figure:
    """Plotly chart with P10/P50/P90, entry window shading, and exit marker."""
    fig = go.Figure()

    days = [d.day for d in plan.daily_actions]
    dates = [d.date for d in plan.daily_actions]
    p10 = [d.price_p10 for d in plan.daily_actions]
    p50 = [d.price_p50 for d in plan.daily_actions]
    p90 = [d.price_p90 for d in plan.daily_actions]

    # Uncertainty band (P10–P90)
    fig.add_trace(go.Scatter(
        x=dates, y=p90, mode="lines",
        line=dict(width=0), showlegend=False, name="P90",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=p10, mode="lines",
        fill="tonexty",
        fillcolor="rgba(52, 152, 219, 0.15)",
        line=dict(width=0), name="P10–P90",
    ))

    # P50 line
    fig.add_trace(go.Scatter(
        x=dates, y=p50, mode="lines+markers",
        line=dict(color="#2980b9", width=2),
        name="P50 (Median)",
    ))

    # Entry price
    fig.add_hline(
        y=plan.entry_price,
        line_dash="dash", line_color="gray",
        annotation_text=f"Entry ${plan.entry_price:.2f}",
    )

    # TP / SL levels
    tp_pct = plan.params.get("take_profit_pct", 5.0)
    sl_pct = plan.params.get("stop_loss_pct", 3.0)
    tp_price = plan.entry_price * (1 + tp_pct / 100)
    sl_price = plan.entry_price * (1 - sl_pct / 100)
    fig.add_hline(
        y=tp_price, line_dash="dot", line_color="green",
        annotation_text=f"TP ${tp_price:.2f}",
    )
    fig.add_hline(
        y=sl_price, line_dash="dot", line_color="red",
        annotation_text=f"SL ${sl_price:.2f}",
    )

    # Entry window shading
    if plan.entry_window and len(dates) >= plan.entry_window.end_day:
        ew = plan.entry_window
        x0 = dates[ew.start_day - 1]
        x1 = dates[ew.end_day - 1]
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="rgba(39, 174, 96, 0.10)",
            line_width=0,
            annotation_text="Entry Window",
            annotation_position="top left",
        )

    # Best exit marker
    if plan.best_exit and plan.best_exit.day <= len(dates):
        idx = plan.best_exit.day - 1
        fig.add_trace(go.Scatter(
            x=[dates[idx]], y=[p50[idx]],
            mode="markers+text",
            marker=dict(size=14, color="gold", symbol="star"),
            text=["★ Best Exit"],
            textposition="top center",
            name="Best Exit",
        ))

    # Day-action colour markers
    for d in plan.daily_actions:
        if d.action == "SELL":
            idx = d.day - 1
            if idx < len(dates):
                fig.add_trace(go.Scatter(
                    x=[dates[idx]], y=[d.price_p50],
                    mode="markers",
                    marker=dict(size=10, color="red", symbol="x"),
                    showlegend=False,
                ))

    fig.update_layout(
        xaxis_title=t.get("axis_date", "Date"),
        yaxis_title=t.get("axis_price", "Price ($)"),
        template="plotly_dark",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    return fig
