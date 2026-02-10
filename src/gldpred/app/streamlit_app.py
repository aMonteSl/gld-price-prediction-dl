"""Streamlit application -- 13-tab GUI for multi-asset price prediction.

Tabs: Dashboard . Data . Train . Models . Forecast . Recommendation .
      Evaluation . Compare . Portfolio . Health . Backtest . Data Hub .
      Tutorial

Default language: Spanish (ES). English available via language selector.

Run with::

    streamlit run app.py
"""
from __future__ import annotations

from typing import Dict, Type

import streamlit as st
import torch.nn as nn

from gldpred.app import state
from gldpred.app.ui.tabs_dashboard import render as render_dashboard_tab
from gldpred.app.ui.tabs_train import render as render_train_tab
from gldpred.app.ui.sidebar import render_sidebar
from gldpred.app.ui.tabs_data import render as render_data_tab
from gldpred.app.ui.tabs_models import render as render_models_tab
from gldpred.app.ui.tabs_forecast import render as render_forecast_tab
from gldpred.app.ui.tabs_recommendation import render as render_recommendation_tab
from gldpred.app.ui.tabs_evaluation import render as render_evaluation_tab
from gldpred.app.ui.tabs_compare import render as render_compare_tab
from gldpred.app.ui.tabs_portfolio import render as render_portfolio_tab
from gldpred.app.ui.tabs_health import render as render_health_tab
from gldpred.app.ui.tabs_backtest import render as render_backtest_tab
from gldpred.app.ui.tabs_datahub import render as render_datahub_tab
from gldpred.app.ui.tabs_tutorial import render as render_tutorial_tab
from gldpred.app.components.onboarding import (
    should_show_onboarding,
    show_onboarding,
)
from gldpred.app.components.walkthrough import render_walkthrough_complete_banner
from gldpred.config import AppConfig
from gldpred.i18n import STRINGS
from gldpred.i18n import DEFAULT_LANGUAGE as _DEFAULT_LANG
from gldpred.models import GRUForecaster, LSTMForecaster, TCNForecaster

# -- Architecture map --
_ARCH_MAP: Dict[str, Type[nn.Module]] = {
    "GRU": GRUForecaster,
    "LSTM": LSTMForecaster,
    "TCN": TCNForecaster,
}

# -- i18n Helper --
def _t() -> dict:
    """Return the current language translation dict."""
    code = st.session_state.get(state.KEY_LANGUAGE, "es")
    return STRINGS.get(code, STRINGS["es"])


def _lang() -> str:
    """Current ISO language code."""
    return st.session_state.get(state.KEY_LANGUAGE, "es")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    cfg = AppConfig()
    st.set_page_config(page_title=cfg.page_title, layout="wide")

    state.init_state()

    # Restore language from query_params (persists across page reloads)
    qp = st.query_params.get("lang")
    if qp and qp in ("es", "en"):
        state.put(state.KEY_LANGUAGE, qp)

    t = _t()
    st.title(t["app_title"])
    st.caption(t["app_subtitle"])

    # Guided onboarding for first-time users
    if should_show_onboarding():
        show_onboarding(t)
        return  # Don't render tabs while onboarding is active

    # Walkthrough completion banner (shows once, then auto-clears)
    render_walkthrough_complete_banner(t)

    render_sidebar(list(_ARCH_MAP.keys()))

    # 13 tabs — Dashboard first (Decision-Primary flow)
    (tab_dash, tab_data, tab_train, tab_models, tab_forecast,
     tab_reco, tab_eval, tab_compare, tab_portfolio, tab_health,
     tab_backtest, tab_datahub, tab_tutorial) = st.tabs([
        t["tab_dashboard"],
        t["tab_data"],
        t["tab_train"],
        t["tab_models"],
        t["tab_forecast"],
        t["tab_recommendation"],
        t["tab_evaluation"],
        t["tab_compare"],
        t["tab_portfolio"],
        t["tab_health"],
        t["tab_backtest"],
        t["tab_datahub"],
        t["tab_tutorial"],
    ])

    with tab_dash:
        render_dashboard_tab(_t())
    with tab_data:
        render_data_tab(_t())
    with tab_train:
        render_train_tab(_t(), _lang(), _ARCH_MAP)
    with tab_models:
        render_models_tab(_t())
    with tab_forecast:
        render_forecast_tab(_t(), _lang())
    with tab_reco:
        render_recommendation_tab(_t(), _lang())
    with tab_eval:
        render_evaluation_tab(_t(), _lang())
    with tab_compare:
        render_compare_tab(_t())
    with tab_portfolio:
        render_portfolio_tab(_t(), _lang())
    with tab_health:
        render_health_tab(_t(), _lang())
    with tab_backtest:
        render_backtest_tab(_t(), _lang())
    with tab_datahub:
        render_datahub_tab(_t(), _lang())
    with tab_tutorial:
        render_tutorial_tab(_t())


# -- Run --
if __name__ == "__main__":
    main()
