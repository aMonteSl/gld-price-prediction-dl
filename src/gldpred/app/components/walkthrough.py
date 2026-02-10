"""Interactive guided walkthrough — hands-on usage example.

After completing the conceptual onboarding (8 steps), users can opt into
a **guided walkthrough** that directs them tab-by-tab through a real
usage session.  Each walkthrough step renders a prominent info banner at
the top of the relevant tab explaining what to do, with a
"✅ Done — Next step" button that advances to the next step.

Walkthrough steps
-----------------
1. **Data tab** — Pick an asset and observe how data loads automatically.
2. **Train tab** — Train a quick model (or load an existing one).
3. **Forecast tab** — View the fan-chart forecast.
4. **Recommendation tab** — Generate a recommendation and read the action plan.
5. **Dashboard tab** — See the full decision board with all assets.
6. **Completed** — Congratulations! Walkthrough ends.

State
-----
``KEY_WALKTHROUGH_STEP``:
  - ``0`` → walkthrough is **off**
  - ``1–5`` → active walkthrough (step number)
  - ``6`` → just finished (briefly shows success, then auto-resets to 0)
"""
from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from gldpred.app import state

# ── Step metadata ────────────────────────────────────────────────────────
# Each tuple: (tab_id, title_key, instruction_key)
# tab_id must match the tab names used in streamlit_app.py
_WALKTHROUGH_STEPS = [
    ("tab_data",           "wt_step1_title", "wt_step1_body"),
    ("tab_train",          "wt_step2_title", "wt_step2_body"),
    ("tab_forecast",       "wt_step3_title", "wt_step3_body"),
    ("tab_recommendation", "wt_step4_title", "wt_step4_body"),
    ("tab_dashboard",      "wt_step5_title", "wt_step5_body"),
]

TOTAL_WT_STEPS = len(_WALKTHROUGH_STEPS)


# ── Public API ───────────────────────────────────────────────────────────

def is_walkthrough_active() -> bool:
    """Return True if the walkthrough is currently active (step 1-5)."""
    step = state.get(state.KEY_WALKTHROUGH_STEP, 0)
    return 1 <= step <= TOTAL_WT_STEPS


def start_walkthrough() -> None:
    """Begin the walkthrough (called after onboarding ends)."""
    state.put(state.KEY_WALKTHROUGH_STEP, 1)


def stop_walkthrough() -> None:
    """Cancel the walkthrough entirely."""
    state.put(state.KEY_WALKTHROUGH_STEP, 0)


def get_walkthrough_tab_key() -> Optional[str]:
    """Return the i18n tab key the user should be on, or None."""
    step = state.get(state.KEY_WALKTHROUGH_STEP, 0)
    if 1 <= step <= TOTAL_WT_STEPS:
        return _WALKTHROUGH_STEPS[step - 1][0]
    return None


def render_walkthrough_banner(t: Dict[str, str], current_tab_key: str) -> None:
    """Render the walkthrough banner if appropriate for this tab.

    Call this at the **top** of each tab's ``render()`` function.
    If the walkthrough is active and this is the correct tab, a styled
    info box with instructions and a "Done — next" button is shown.

    If this is NOT the active tab but the walkthrough is active, a small
    nudge message is shown redirecting the user.

    Parameters
    ----------
    t : dict
        Translation dictionary.
    current_tab_key : str
        The i18n key of the tab being rendered (e.g. ``"tab_data"``).
    """
    step = state.get(state.KEY_WALKTHROUGH_STEP, 0)
    if step < 1 or step > TOTAL_WT_STEPS:
        return  # walkthrough not active

    expected_tab, title_key, body_key = _WALKTHROUGH_STEPS[step - 1]

    if current_tab_key == expected_tab:
        # ── Active step: show full instruction banner ────────────
        _render_active_banner(t, step, title_key, body_key)
    else:
        # ── Wrong tab: show a gentle nudge ───────────────────────
        target_label = t.get(expected_tab, expected_tab)
        nudge = t.get("wt_go_to_tab", "👉 El tutorial te espera en la pestaña **{tab}**").format(
            tab=target_label
        )
        st.info(nudge, icon="🧭")


def render_walkthrough_complete_banner(t: Dict[str, str]) -> None:
    """Show a success banner when the walkthrough just finished (step 6).

    The banner auto-clears on the next interaction.
    """
    step = state.get(state.KEY_WALKTHROUGH_STEP, 0)
    if step == TOTAL_WT_STEPS + 1:
        st.balloons()
        st.success(t.get("wt_complete", "🎉 ¡Tutorial completado!"), icon="🏆")
        st.markdown(t.get("wt_complete_body", ""))
        # Auto-clear so it doesn't persist
        state.put(state.KEY_WALKTHROUGH_STEP, 0)


# ── Private helpers ──────────────────────────────────────────────────────

def _render_active_banner(
    t: Dict[str, str],
    step: int,
    title_key: str,
    body_key: str,
) -> None:
    """Render the walkthrough instruction card for the current step."""
    title = t.get(title_key, f"Step {step}")
    body = t.get(body_key, "")

    progress = step / TOTAL_WT_STEPS
    progress_text = t.get("wt_progress", "Tutorial práctico").format(
        step=step, total=TOTAL_WT_STEPS,
    )

    with st.container():
        st.info(f"🧭 **{progress_text}**", icon="📚")
        st.progress(progress)
        st.markdown(f"### {title}")
        st.markdown(body)

        # Navigation
        cols = st.columns([1, 1, 2])

        is_last = step == TOTAL_WT_STEPS

        next_label = (
            t.get("wt_finish", "🏁 Terminar tutorial")
            if is_last
            else t.get("wt_done_next", "✅ Listo — Siguiente paso")
        )

        if cols[0].button(
            next_label,
            key=f"wt_next_{step}",
            type="primary",
            use_container_width=True,
        ):
            if is_last:
                state.put(state.KEY_WALKTHROUGH_STEP, TOTAL_WT_STEPS + 1)
            else:
                state.put(state.KEY_WALKTHROUGH_STEP, step + 1)
            st.rerun()

        if cols[1].button(
            t.get("wt_skip", "⏭️ Saltar tutorial"),
            key=f"wt_skip_{step}",
            use_container_width=True,
        ):
            stop_walkthrough()
            st.rerun()

        st.markdown("---")
