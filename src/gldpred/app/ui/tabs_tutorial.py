"""Tutorial tab rendering."""
from __future__ import annotations

from typing import Dict

import streamlit as st

from gldpred.app.components.onboarding import restart_onboarding
from gldpred.app.components.walkthrough import start_walkthrough


def render(t: Dict[str, str]) -> None:
    """Render the Tutorial tab."""
    st.header(t["tut_header"])
    st.markdown(t["tut_disclaimer"])

    # Restart buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            t.get("onb_restart", "📘 Restart Guided Tutorial"),
            key="tut_restart_onboarding",
            type="primary",
            use_container_width=True,
        ):
            restart_onboarding()
            st.rerun()

    with col2:
        if st.button(
            t.get("wt_restart", "🚀 Restart Hands-On Walkthrough"),
            key="tut_restart_walkthrough",
            use_container_width=True,
        ):
            start_walkthrough()
            st.rerun()

    st.divider()

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
