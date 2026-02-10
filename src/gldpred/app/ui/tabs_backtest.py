"""Backtest tab — walk-forward simulation of model predictions vs actuals.

Lets users select a model and date range, runs historical inference, and
visualises predicted vs actual returns with coverage and accuracy metrics.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import streamlit as st

from gldpred.app import state
from gldpred.app.components.empty_states import show_empty_no_data, show_empty_no_model
from gldpred.services.backtest_engine import BacktestEngine, BacktestSummary


def render(t: Dict[str, str], lang: str = "en") -> None:
    """Render the Backtest tab."""
    st.header(t.get("backtest_header", "Walk-Forward Backtest"))
    st.caption(t.get(
        "backtest_subtitle",
        "Simulate how the model would have performed on historical data.",
    ))

    # ── Guards ────────────────────────────────────────────────────────
    df = state.get(state.KEY_RAW_DF)
    if df is None:
        show_empty_no_data(t)
        return

    bundle = state.get(state.KEY_ACTIVE_MODEL)
    if bundle is None:
        show_empty_no_model(t)
        return

    # ── Feature engineering ───────────────────────────────────────────
    from gldpred.features import FeatureEngineering

    fe = FeatureEngineering()
    df_feat = fe.add_technical_indicators(df)
    feat_df = fe.select_features(df_feat)
    feature_names = list(feat_df.columns)
    df_feat = df_feat.dropna()

    n = len(df_feat)
    seq_length = bundle.config.get("seq_length", 20)
    forecast_steps = bundle.config.get("forecast_steps", 20)
    min_rows = seq_length + forecast_steps + 1

    if n < min_rows:
        st.warning(t.get(
            "backtest_not_enough_data",
            f"Not enough data rows ({n}) for backtesting. Need at least {min_rows}.",
        ))
        return

    # ── Parameters ────────────────────────────────────────────────────
    st.markdown(f"##### {t.get('backtest_params', 'Parameters')}")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_points = (n - seq_length - forecast_steps) // forecast_steps
        num_points = st.slider(
            t.get("backtest_num_points", "Backtest points"),
            min_value=5,
            max_value=min(max_points, 100),
            value=min(20, max_points),
            step=1,
        )
    with c2:
        investment = st.number_input(
            t.get("backtest_investment", "Investment ($)"),
            min_value=1,
            max_value=1_000_000,
            value=10_000,
            step=1000,
        )
    with c3:
        st.markdown(f"**{t.get('backtest_model', 'Model')}:** {bundle.label}")
        st.markdown(f"**{t.get('backtest_asset', 'Asset')}:** {bundle.asset}")

    # ── Run ───────────────────────────────────────────────────────────
    if st.button(t.get("backtest_run", "🚀 Run Backtest"), type="primary"):
        total_range = n - seq_length - forecast_steps
        step = max(total_range // num_points, 1)
        start_idx = seq_length

        engine = BacktestEngine()
        with st.spinner(t.get("backtest_running", "Running walk-forward backtest...")):
            summary = engine.run_walk_forward(
                df=df_feat,
                bundle=bundle,
                feature_names=feature_names,
                start_idx=start_idx,
                step=step,
                seq_length=seq_length,
                forecast_steps=forecast_steps,
                investment=investment,
            )

        if summary.total_points == 0:
            st.warning(t.get("backtest_no_results", "No backtest results produced."))
            return

        st.session_state["_backtest_summary"] = summary

    # ── Display results ───────────────────────────────────────────────
    summary: BacktestSummary | None = st.session_state.get("_backtest_summary")
    if summary is None or summary.total_points == 0:
        st.info(t.get(
            "backtest_click_run",
            "Configure parameters and click Run to start the backtest.",
        ))
        return

    _render_summary(summary, t)
    _render_chart(summary, t)
    _render_table(summary, t)


# ──────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────

def _render_summary(s: BacktestSummary, t: Dict[str, str]) -> None:
    """Show aggregate backtest metrics."""
    st.markdown(f"### {t.get('backtest_results_header', 'Results')}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t.get("backtest_points", "Points"), s.total_points)
    c2.metric(
        t.get("backtest_dir_accuracy", "Dir. Accuracy"),
        f"{s.directional_accuracy:.0f}%",
    )
    c3.metric(
        t.get("backtest_coverage", "P10–P90 Coverage"),
        f"{s.coverage:.0f}%",
    )
    c4.metric(
        t.get("backtest_mae", "MAE"),
        f"{s.mae * 100:.2f}%",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        t.get("backtest_avg_pred", "Avg Predicted"),
        f"{s.avg_predicted_return * 100:+.2f}%",
    )
    c6.metric(
        t.get("backtest_avg_actual", "Avg Actual"),
        f"{s.avg_actual_return * 100:+.2f}%",
    )
    c7.metric(
        t.get("backtest_bias", "Bias"),
        f"{s.prediction_bias * 100:+.2f}%",
    )
    c8.metric(
        t.get("backtest_avg_pnl", "Avg P&L"),
        f"${s.avg_actual_pnl:+,.0f}",
    )


def _render_chart(s: BacktestSummary, t: Dict[str, str]) -> None:
    """Predicted vs actual return scatter + line chart."""
    import plotly.graph_objects as go

    dates = [r.backtest_date[:10] for r in s.results]
    pred = [r.predicted_return_p50 * 100 for r in s.results]
    actual = [r.actual_return * 100 for r in s.results]
    p10 = [r.predicted_return_p10 * 100 for r in s.results]
    p90 = [r.predicted_return_p90 * 100 for r in s.results]

    fig = go.Figure()

    # P10–P90 band
    fig.add_trace(go.Scatter(
        x=dates, y=p90,
        mode="lines", line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=p10,
        mode="lines", line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(135, 206, 250, 0.2)",
        name=t.get("backtest_band", "P10–P90 Band"),
    ))

    # Predicted P50
    fig.add_trace(go.Scatter(
        x=dates, y=pred,
        mode="lines+markers",
        name=t.get("backtest_predicted", "Predicted (P50)"),
        line=dict(color="#3498db", width=2),
        marker=dict(size=5),
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        mode="lines+markers",
        name=t.get("backtest_actual", "Actual"),
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=5),
    ))

    fig.update_layout(
        title=t.get("backtest_chart_title", "Predicted vs Actual Returns"),
        xaxis_title=t.get("backtest_chart_x", "Date"),
        yaxis_title=t.get("backtest_chart_y", "Return (%)"),
        hovermode="x unified",
        height=420,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    st.plotly_chart(fig, use_container_width=True)


def _render_table(s: BacktestSummary, t: Dict[str, str]) -> None:
    """Expandable detail table."""
    with st.expander(t.get("backtest_detail_table", "📋 Detail Table"), expanded=False):
        import pandas as pd

        rows = []
        for r in s.results:
            rows.append({
                t.get("backtest_col_date", "Date"): r.backtest_date[:10],
                t.get("backtest_col_entry", "Entry"): f"${r.entry_price:.2f}",
                t.get("backtest_col_pred", "Pred P50"): f"{r.predicted_return_p50 * 100:+.2f}%",
                t.get("backtest_col_actual", "Actual"): f"{r.actual_return * 100:+.2f}%",
                t.get("backtest_col_error", "Error"): f"{r.prediction_error * 100:.2f}%",
                t.get("backtest_col_band", "In Band"): "✅" if r.within_band else "❌",
                t.get("backtest_col_pnl", "P&L"): f"${r.actual_pnl:+,.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
