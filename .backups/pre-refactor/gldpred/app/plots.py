"""Reusable Plotly chart builders for the Streamlit GUI.

All functions return a ``plotly.graph_objects.Figure`` — the caller is
responsible for rendering via ``st.plotly_chart()``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ======================================================================
# Loss curve with diagnostic markers
# ======================================================================

def create_loss_chart(
    train_losses: List[float],
    val_losses: List[float],
    *,
    best_epoch: Optional[int] = None,
    verdict: Optional[str] = None,
) -> go.Figure:
    """Create training / validation loss chart with diagnostic markers.

    Args:
        train_losses: per-epoch training pinball loss.
        val_losses: per-epoch validation pinball loss.
        best_epoch: 0-based index of the epoch with minimum val loss.
            Drawn as a vertical dashed line.
        verdict: diagnostics verdict string.  When ``"overfitting"``,
            epochs *after* ``best_epoch`` are shaded as an overfit zone.

    Returns:
        Plotly Figure ready for rendering.
    """
    fig = go.Figure()
    epochs_x = list(range(1, len(train_losses) + 1))

    fig.add_trace(go.Scatter(
        x=epochs_x, y=train_losses, mode="lines",
        name="Train", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=epochs_x, y=val_losses, mode="lines",
        name="Validation", line=dict(color="#ff7f0e", width=2),
    ))

    # ── Best-epoch marker ────────────────────────────────────────────
    if best_epoch is not None and 0 <= best_epoch < len(val_losses):
        best_x = best_epoch + 1  # 1-based for display
        fig.add_vline(
            x=best_x,
            line_dash="dash",
            line_color="#2ca02c",
            line_width=1.5,
            annotation_text=f"Best epoch ({best_x})",
            annotation_position="top right",
            annotation_font_color="#2ca02c",
        )

    # ── Overfit shading ──────────────────────────────────────────────
    if (
        verdict == "overfitting"
        and best_epoch is not None
        and best_epoch + 1 < len(val_losses)
    ):
        fig.add_vrect(
            x0=best_epoch + 1,           # 1-based
            x1=len(val_losses),
            fillcolor="rgba(255, 0, 0, 0.08)",
            layer="below",
            line_width=0,
            annotation_text="Overfit zone",
            annotation_position="top left",
            annotation_font_color="#d62728",
        )

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss (pinball)",
        template="plotly_dark",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ======================================================================
# Fan chart (price trajectory with uncertainty bands)
# ======================================================================

def create_fan_chart(
    df: pd.DataFrame,
    forecast: Any,
    *,
    x_label: str = "Date",
    y_label: str = "Price (USD)",
    hist_tail: int = 60,
) -> go.Figure:
    """Create a Plotly fan chart with historical tail + forecast bands.

    Args:
        df: DataFrame with at least a ``"Close"`` column (DatetimeIndex).
        forecast: :class:`TrajectoryForecast` object.
        x_label: X-axis label.
        y_label: Y-axis label.
        hist_tail: number of historical trading days to show before the
            forecast begins.

    Returns:
        Plotly Figure ready for rendering.
    """
    quantiles = forecast.quantiles
    q_idx: Dict[float, int] = {round(q, 2): i for i, q in enumerate(quantiles)}
    lo_i = q_idx.get(0.1, 0)
    med_i = q_idx.get(0.5, 1)
    hi_i = q_idx.get(0.9, len(quantiles) - 1)

    # Historical tail
    tail = df["Close"].iloc[-hist_tail:]

    # Forecast dates (include last known price as anchor)
    fc_dates = [forecast.last_date] + list(forecast.dates)
    prices_med = forecast.price_paths[:, med_i]
    prices_lo = forecast.price_paths[:, lo_i]
    prices_hi = forecast.price_paths[:, hi_i]

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=tail.index, y=tail.values,
        mode="lines", name="Historical",
        line=dict(color="#FFD700", width=2),
    ))

    # P10–P90 band (upper, then lower with fill between)
    fig.add_trace(go.Scatter(
        x=fc_dates, y=prices_hi.tolist(),
        mode="lines", line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=prices_lo.tolist(),
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(255,215,0,0.15)",
        name="P10–P90",
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=fc_dates, y=prices_med.tolist(),
        mode="lines+markers", name="P50 (Median)",
        line=dict(color="#00BFFF", width=2),
    ))

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_dark",
        height=500,
    )
    return fig
