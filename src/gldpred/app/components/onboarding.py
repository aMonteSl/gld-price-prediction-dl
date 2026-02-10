"""Guided onboarding – step-by-step tutorial for first-time users.

Triggers automatically when the app is opened for the first time
(``tutorial_done`` is False in session state). Users can also relaunch
it manually from the Tutorial tab via the "📘 Tutorial Guiado" button.

Implementation uses ``st.container`` with prominent styling and
Next / Back / Skip navigation. Each step explains a major area of
the application in plain, non-technical Spanish with real money
examples (€1 000).
"""
from __future__ import annotations

from typing import Dict

import streamlit as st

from gldpred.app import state


# ── Step definitions ─────────────────────────────────────────────────
# Each step: (title_key, body_key)
_STEPS = [
    ("onb_step1_title", "onb_step1_body"),
    ("onb_step2_title", "onb_step2_body"),
    ("onb_step3_title", "onb_step3_body"),
    ("onb_step4_title", "onb_step4_body"),
    ("onb_step5_title", "onb_step5_body"),
    ("onb_step6_title", "onb_step6_body"),
    ("onb_step7_title", "onb_step7_body"),
    ("onb_step8_title", "onb_step8_body"),
]

TOTAL_STEPS = len(_STEPS)


def should_show_onboarding() -> bool:
    """Return True if the onboarding should auto-trigger."""
    return not state.get(state.KEY_TUTORIAL_DONE, False)


def show_onboarding(t: Dict[str, str]) -> None:
    """Render the guided onboarding overlay.

    Shows the current step with Next/Back/Skip buttons.
    When the user completes or skips, sets tutorial_done = True.
    """
    current = state.get(state.KEY_ONBOARDING_STEP, 0)

    # Safety clamp
    if current < 0:
        current = 0
    if current >= TOTAL_STEPS:
        _finish_onboarding()
        return

    title_key, body_key = _STEPS[current]
    title = t.get(title_key, f"Step {current + 1}")
    body = t.get(body_key, "")

    # ── Render the onboarding card ────────────────────────────────
    st.markdown("---")
    with st.container():
        # Progress bar
        progress = (current + 1) / TOTAL_STEPS
        st.progress(progress, text=f"{t.get('onb_progress', 'Paso')} {current + 1} / {TOTAL_STEPS}")

        st.markdown(f"## 🧭 {title}")
        st.markdown(body)

        # Navigation buttons
        nav_cols = st.columns([1, 1, 1, 2])

        # Back
        if current > 0:
            if nav_cols[0].button(
                t.get("onb_back", "⬅️ Atrás"),
                key="onb_back_btn",
                use_container_width=True,
            ):
                state.put(state.KEY_ONBOARDING_STEP, current - 1)
                st.rerun()
        else:
            nav_cols[0].write("")  # placeholder

        # Next / Finish
        is_last = current == TOTAL_STEPS - 1
        next_label = t.get("onb_finish", "✅ Empezar") if is_last else t.get("onb_next", "Siguiente ➡️")
        if nav_cols[1].button(
            next_label,
            key="onb_next_btn",
            type="primary",
            use_container_width=True,
        ):
            if is_last:
                _finish_onboarding()
            else:
                state.put(state.KEY_ONBOARDING_STEP, current + 1)
            st.rerun()

        # Skip
        if nav_cols[2].button(
            t.get("onb_skip", "⏭️ Saltar"),
            key="onb_skip_btn",
            use_container_width=True,
        ):
            _finish_onboarding()
            st.rerun()

    st.markdown("---")


def restart_onboarding() -> None:
    """Reset onboarding state so it triggers again."""
    state.put(state.KEY_TUTORIAL_DONE, False)
    state.put(state.KEY_ONBOARDING_STEP, 0)


def _finish_onboarding() -> None:
    """Mark onboarding as completed."""
    state.put(state.KEY_TUTORIAL_DONE, True)
    state.put(state.KEY_ONBOARDING_STEP, 0)
