"""Recommendation controller for Streamlit orchestration."""
from __future__ import annotations

import streamlit as st

from gldpred.app import state
from gldpred.config import DecisionConfig
from gldpred.decision import DecisionEngine, RecommendationHistory


def generate_recommendation(
    forecast,
    df,
    asset: str,
    diagnostics_verdict: str | None,
    horizon: int,
):
    """Generate a recommendation and update history."""
    decision_cfg = DecisionConfig()
    max_vol = decision_cfg.max_volatility.get(
        asset, decision_cfg.max_volatility["default"],
    )
    engine = DecisionEngine(
        horizon_days=horizon,
        min_expected_return=decision_cfg.min_expected_return,
        max_volatility=max_vol,
    )
    reco = engine.recommend(
        forecast.returns_quantiles,
        df,
        quantiles=forecast.quantiles,
        diagnostics_verdict=diagnostics_verdict,
    )

    if (
        state.KEY_RECO_HISTORY not in st.session_state
        or st.session_state[state.KEY_RECO_HISTORY] is None
    ):
        st.session_state[state.KEY_RECO_HISTORY] = RecommendationHistory()
    st.session_state[state.KEY_RECO_HISTORY].add(asset, reco)
    return reco


def get_recommendation_history() -> RecommendationHistory | None:
    """Return recommendation history if present."""
    return st.session_state.get(state.KEY_RECO_HISTORY)
