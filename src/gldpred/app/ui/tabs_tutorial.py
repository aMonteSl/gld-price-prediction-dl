"""Tutorial tab rendering."""
from __future__ import annotations

from typing import Dict

import streamlit as st

from gldpred.app.components.onboarding import restart_onboarding


def render(t: Dict[str, str]) -> None:
    """Render the Tutorial tab."""
    st.header(t["tut_header"])
    st.markdown(t["tut_disclaimer"])

    # Restart guided onboarding button
    if st.button(
        t.get("onb_restart", "📘 Restart Guided Tutorial"),
        key="tut_restart_onboarding",
        type="primary",
    ):
        restart_onboarding()
        st.success(t.get("onb_restart_done", "Tutorial restarted!"))
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
