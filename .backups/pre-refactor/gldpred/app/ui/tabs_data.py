"""Data tab rendering."""
from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gldpred.app import state
from gldpred.app.data_controller import LoadedData, fetch_asset_data, invalidate_cache


def render(t: Dict[str, str]) -> None:
    """Render the Data tab."""
    st.header(t["data_header"])
    st.info(t["data_info"])

    asset = state.get(state.KEY_ASSET, "GLD")

    # Auto-load: fetch data on first render or asset change
    loaded_asset = state.get(state.KEY_DATA_LOADED_ASSET)

    if loaded_asset != asset or state.get(state.KEY_RAW_DF) is None:
        try:
            with st.spinner(t["data_loading_spinner"]):
                data = fetch_asset_data(asset)
            _store_loaded_data(asset, data)
        except Exception as exc:
            st.error(t["data_load_error"].format(err=exc))
            return

    # Refresh button
    if st.button(t["data_refresh_btn"], key="btn_refresh_data"):
        invalidate_cache()
        try:
            with st.spinner(t["data_loading_spinner"]):
                data = fetch_asset_data(asset)
            _store_loaded_data(asset, data)
            st.rerun()
        except Exception as exc:
            st.error(t["data_load_error"].format(err=exc))
            return

    df = state.get(state.KEY_RAW_DF)
    if df is None:
        return

    # Metrics
    n = len(df)
    start = str(df.index[0].date()) if hasattr(df.index[0], "date") else str(df.index[0])
    end = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])
    st.success(t["data_load_success"].format(n=n, asset=asset, start=start, end=end))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t["data_metric_records"], f"{n:,}")
    latest_price = df["Close"].iloc[-1]
    c2.metric(t["data_metric_price"], f"${latest_price:.2f}")
    pct_change = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100 if n > 1 else 0
    c3.metric(t["data_metric_change"], f"{pct_change:+.2f}%")
    feature_names = state.get(state.KEY_FEATURE_NAMES, [])
    c4.metric(t["data_metric_features"], len(feature_names))

    # Price chart
    st.subheader(t["data_price_history"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines", name=asset,
    ))
    if "sma_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["sma_50"], mode="lines",
            name="SMA 50", line=dict(dash="dash"),
        ))
    if "sma_200" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["sma_200"], mode="lines",
            name="SMA 200", line=dict(dash="dot"),
        ))
    fig.update_layout(
        xaxis_title=t["axis_date"],
        yaxis_title=t["axis_price"],
        template="plotly_dark",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data preview
    with st.expander(t["data_preview"]):
        st.dataframe(df.tail(20), use_container_width=True)


def _store_loaded_data(asset: str, data: LoadedData) -> None:
    """Push fetched data into session state."""
    state.put(state.KEY_RAW_DF, data.df)
    state.put(state.KEY_DAILY_RETURNS, data.daily_returns)
    state.put(state.KEY_FEATURE_NAMES, data.feature_names)
    state.put(state.KEY_DATA_LOADED_ASSET, asset)
