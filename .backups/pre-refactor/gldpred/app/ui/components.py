"""Shared UI helpers for the Streamlit app."""
from __future__ import annotations

from typing import Dict, List

import streamlit as st

from gldpred.app import state
from gldpred.app.glossary import info_term
from gldpred.app.plots import create_loss_chart

HIDDEN_SIZE_OPTIONS = [32, 48, 64, 96, 128]
BATCH_SIZE_OPTIONS = [16, 32, 64, 128, 256]
LR_OPTIONS = [0.0001, 0.0005, 0.001, 0.005, 0.01]


def show_diagnostics(t: Dict[str, str], lang: str) -> None:
    """Show training diagnostics, loss chart, and Apply Suggestions."""
    diag = state.get(state.KEY_DIAG_RESULT)
    train_losses = state.get(state.KEY_TRAIN_LOSSES)
    val_losses = state.get(state.KEY_VAL_LOSSES)
    if diag is None:
        return

    st.subheader(t["diag_header"])

    verdict_map = {
        "healthy": t["diag_verdict_healthy"],
        "overfitting": t["diag_verdict_overfitting"],
        "underfitting": t["diag_verdict_underfitting"],
        "noisy": t["diag_verdict_noisy"],
    }
    v_label = verdict_map.get(diag.verdict, diag.verdict)

    if diag.verdict in ("overfitting", "underfitting"):
        info_term(t["diag_verdict"], diag.verdict, lang)
    else:
        info_term(t["diag_verdict"], "pinball_loss", lang)

    c1, c2, c3 = st.columns(3)
    c1.metric(t["diag_verdict"], v_label)
    c2.metric(t["diag_best_epoch"], diag.best_epoch + 1)
    c3.metric(t["diag_gen_gap"], f"{diag.generalization_gap:.4f}")

    st.markdown(f"**{t['diag_explanation']}:** {diag.explanation}")
    if diag.suggestions:
        st.markdown(f"**{t['diag_suggestions']}:**")
        for s in diag.suggestions:
            st.markdown(f"- {s}")

    # Apply Suggestions button
    if diag.verdict != "healthy":
        already = state.get(state.KEY_SUGGESTIONS_APPLIED, False)
        if already:
            st.info(t["diag_applied_success"])
        else:
            if st.button(t["diag_apply_btn"], key="btn_apply_suggestions"):
                apply_suggestions(diag.verdict, diag.best_epoch)
                state.put(state.KEY_SUGGESTIONS_APPLIED, True)
                st.rerun()

    # Loss chart
    if train_losses and val_losses:
        fig = create_loss_chart(
            train_losses, val_losses,
            best_epoch=diag.best_epoch, verdict=diag.verdict,
        )
        st.plotly_chart(fig, use_container_width=True)


def apply_suggestions(verdict: str, best_epoch: int) -> None:
    """Store suggested config values in temporary _sugg_* keys."""
    ss = st.session_state

    if verdict == "overfitting":
        ss["_sugg_epochs"] = min(max(best_epoch + 5, 10), 200)
        current_hs = ss.get("hidden_size", 64)
        ss["_sugg_hidden_size"] = step_value(
            current_hs, HIDDEN_SIZE_OPTIONS, -1,
        )
        nl = ss.get("num_layers", 2)
        if nl > 1:
            ss["_sugg_num_layers"] = nl - 1

    elif verdict == "underfitting":
        ss["_sugg_epochs"] = min(ss.get("epochs", 50) + 50, 200)
        current_hs = ss.get("hidden_size", 64)
        ss["_sugg_hidden_size"] = step_value(
            current_hs, HIDDEN_SIZE_OPTIONS, +1,
        )
        nl = ss.get("num_layers", 2)
        if nl < 4:
            ss["_sugg_num_layers"] = nl + 1
        current_lr = ss.get("learning_rate", 0.001)
        ss["_sugg_learning_rate"] = step_value(
            current_lr, LR_OPTIONS, +1,
        )

    elif verdict == "noisy":
        current_lr = ss.get("learning_rate", 0.001)
        ss["_sugg_learning_rate"] = step_value(
            current_lr, LR_OPTIONS, -1,
        )
        current_bs = ss.get("batch_size", 32)
        ss["_sugg_batch_size"] = step_value(
            current_bs, BATCH_SIZE_OPTIONS, +1,
        )
        sl = ss.get("seq_length", 20)
        ss["_sugg_seq_length"] = min(sl + 10, 60)


def step_value(current: float, options: List[float], direction: int) -> float:
    """Step a select-slider value up (+1) or down (-1) within *options*."""
    try:
        idx = options.index(current)
    except ValueError:
        idx = min(
            range(len(options)),
            key=lambda i: abs(options[i] - current),
        )
    new_idx = max(0, min(len(options) - 1, idx + direction))
    return options[new_idx]
