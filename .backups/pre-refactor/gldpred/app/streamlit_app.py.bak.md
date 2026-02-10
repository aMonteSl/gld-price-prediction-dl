"""Streamlit application -- 8-tab GUI for multi-asset price prediction.

Tabs: Data . Train . Models . Forecast . Recommendation . Evaluation .
      Compare . Tutorial

Run with::

    streamlit run app.py
"""
from __future__ import annotations

import traceback
from typing import Any, Dict, List, Type

import streamlit as st
import torch.nn as nn

from gldpred.app import state
from gldpred.app.ui.tabs_train import render as render_train_tab
from gldpred.app.ui.sidebar import render_sidebar
from gldpred.app.ui.tabs_data import render as render_data_tab
from gldpred.app.ui.tabs_models import render as render_models_tab
from gldpred.app.ui.tabs_forecast import render as render_forecast_tab
from gldpred.app.ui.tabs_recommendation import render as render_recommendation_tab
from gldpred.app.ui.tabs_evaluation import render as render_evaluation_tab
from gldpred.app.ui.tabs_compare import render as render_compare_tab
from gldpred.config import AppConfig
from gldpred.i18n import STRINGS
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
    code = st.session_state.get(state.KEY_LANGUAGE, "en")
    return STRINGS.get(code, STRINGS["en"])


def _lang() -> str:
    """Current ISO language code."""
    return st.session_state.get(state.KEY_LANGUAGE, "en")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    cfg = AppConfig()
    st.set_page_config(page_title=cfg.page_title, layout="wide")

    state.init_state()

    t = _t()
    st.title(t["app_title"])
    st.caption(t["app_subtitle"])

    render_sidebar(list(_ARCH_MAP.keys()))

    # 8 tabs
    (tab_data, tab_train, tab_models, tab_forecast,
     tab_reco, tab_eval, tab_compare, tab_tutorial) = st.tabs([
        t["tab_data"],
        t["tab_train"],
        t["tab_models"],
        t["tab_forecast"],
        t["tab_recommendation"],
        t["tab_evaluation"],
        t["tab_compare"],
        t["tab_tutorial"],
    ])

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
    with tab_tutorial:
        _tab_tutorial()


# ======================================================================
# TAB 8 -- TUTORIAL
# ======================================================================
def _tab_tutorial() -> None:
    t = _t()
    st.header(t["tut_header"])
    st.markdown(t["tut_disclaimer"])

    sections = [
        ("tut_s1_title", "tut_s1_body"),
        ("tut_s2_title", "tut_s2_body"),
        ("tut_s3_title", "tut_s3_body"),
        ("tut_s4_title", "tut_s4_body"),
        ("tut_s5_title", "tut_s5_body"),
        ("tut_s6_title", "tut_s6_body"),
        ("tut_s7_title", "tut_s7_body"),
        ("tut_s8_title", "tut_s8_body"),
        ("tut_s9_title", "tut_s9_body"),
        ("tut_s10_title", "tut_s10_body"),
    ]
    for title_key, body_key in sections:
        with st.expander(t.get(title_key, title_key)):
            st.markdown(t.get(body_key, ""))


# -- Run --
if __name__ == "__main__":
    main()
