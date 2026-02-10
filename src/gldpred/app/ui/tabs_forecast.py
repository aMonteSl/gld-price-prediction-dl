"""Forecast tab rendering."""
from __future__ import annotations

import traceback
from typing import Dict

import pandas as pd
import streamlit as st

from gldpred.app import state
from gldpred.app.controllers.forecasting_controller import generate_forecast
from gldpred.app.controllers.model_loader import auto_select_model, get_active_model
from gldpred.app.glossary import info_term
from gldpred.app.plots import create_fan_chart


def render(t: Dict[str, str], lang: str) -> None:
    """Render the Forecast tab."""
    st.header(t["forecast_header"])
    st.info(t["forecast_info"])

    df = state.get(state.KEY_RAW_DF)
    if df is None:
        st.warning(t["forecast_warn_no_model"])
        return

    # Get model: active model > auto-select from registry
    asset = state.get(state.KEY_ASSET, "GLD")
    model = get_active_model()
    if model is None:
        model = auto_select_model(asset)
    if model is None:
        st.warning(t["forecast_warn_no_model"])
        return

    # Validate asset compatibility
    if model.asset != asset:
        st.warning(
            t.get("sidebar_model_mismatch", "").format(
                model_asset=model.asset, asset=asset,
            )
        )

    try:
        feature_names = state.get(state.KEY_FEATURE_NAMES, [])
        seq_length = model.config.get(
            "seq_length", st.session_state.get("seq_length", 20),
        )
        quantiles = model.quantiles

        forecast = generate_forecast(
            model, df, feature_names, seq_length, asset,
        )

        # Fan chart
        st.subheader(t["forecast_fan_chart"])
        info_term(t["forecast_fan_chart"], "fan_chart", lang)
        fig = create_fan_chart(
            df, forecast,
            x_label=t["axis_date"],
            y_label=t["axis_price"],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader(t["forecast_table"])
        info_term("Quantiles (P10 / P50 / P90)", "quantiles", lang)
        K = forecast.returns_quantiles.shape[0]
        q_idx = {round(q, 2): i for i, q in enumerate(quantiles)}
        lo_i = q_idx.get(0.1, 0)
        med_i = q_idx.get(0.5, 1)
        hi_i = q_idx.get(0.9, 2)

        rows = []
        for k in range(K):
            rows.append({
                t["forecast_col_day"]: k + 1,
                t["forecast_col_date"]: (
                    str(forecast.dates[k].date())
                    if hasattr(forecast.dates[k], "date")
                    else str(forecast.dates[k])
                ),
                t["forecast_col_p10"]: f"${forecast.price_paths[k + 1, lo_i]:.2f}",
                t["forecast_col_p50"]: f"${forecast.price_paths[k + 1, med_i]:.2f}",
                t["forecast_col_p90"]: f"${forecast.price_paths[k + 1, hi_i]:.2f}",
                t["forecast_col_return"]: f"{forecast.returns_quantiles[k, med_i]:+.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    except Exception as exc:
        st.error(t["forecast_error"].format(err=exc))
        st.code(traceback.format_exc())
