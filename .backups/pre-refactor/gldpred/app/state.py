"""Centralised session-state management for the Streamlit app.

All session-state keys, defaults, and helpers live here so that every tab
reads/writes the same canonical names. No magic strings scattered across
the GUI code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import streamlit as st


# ── Canonical key names ──────────────────────────────────────────────────
# Data pipeline
KEY_ASSET = "asset"
KEY_RAW_DF = "raw_df"
KEY_DAILY_RETURNS = "daily_returns"
KEY_FEATURE_NAMES = "feature_names"
KEY_DATA_LOADED_ASSET = "data_loaded_asset"  # tracks which asset was loaded

# Training pipeline
KEY_TRAINER = "trainer"
KEY_TRAIN_LOSSES = "train_losses"
KEY_VAL_LOSSES = "val_losses"
KEY_DIAG_RESULT = "diag_result"
KEY_TRAJ_METRICS = "traj_metrics"
KEY_QUANT_METRICS = "quant_metrics"
KEY_LAST_MODEL_ID = "last_model_id"
KEY_SUGGESTIONS_APPLIED = "suggestions_applied"

# Forecast / Recommendation
KEY_FORECAST = "forecast"
KEY_RECO_HISTORY = "reco_history"

# Active model (loaded from registry for inference)
KEY_ACTIVE_MODEL = "active_model"          # ModelBundle or None
KEY_ACTIVE_MODEL_ID = "active_model_id"    # str or None

# Compare
KEY_COMPARE_RESULT = "compare_result"

# UI
KEY_LANGUAGE = "language"


# ── Defaults ─────────────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    KEY_ASSET: "GLD",
    KEY_RAW_DF: None,
    KEY_DAILY_RETURNS: None,
    KEY_FEATURE_NAMES: [],
    KEY_DATA_LOADED_ASSET: None,
    KEY_TRAINER: None,
    KEY_TRAIN_LOSSES: None,
    KEY_VAL_LOSSES: None,
    KEY_DIAG_RESULT: None,
    KEY_TRAJ_METRICS: None,
    KEY_QUANT_METRICS: None,
    KEY_LAST_MODEL_ID: None,
    KEY_SUGGESTIONS_APPLIED: False,
    KEY_FORECAST: None,
    KEY_RECO_HISTORY: None,
    KEY_ACTIVE_MODEL: None,
    KEY_ACTIVE_MODEL_ID: None,
    KEY_COMPARE_RESULT: None,
}


def init_state() -> None:
    """Ensure every canonical key exists in ``st.session_state``."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get(key: str, default: Any = None) -> Any:
    """Read a session-state value (safe)."""
    return st.session_state.get(key, default)


def put(key: str, value: Any) -> None:
    """Write a session-state value."""
    st.session_state[key] = value


def clear_training_state() -> None:
    """Reset all training-related state (used when asset changes)."""
    for k in (
        KEY_TRAINER,
        KEY_TRAIN_LOSSES,
        KEY_VAL_LOSSES,
        KEY_DIAG_RESULT,
        KEY_TRAJ_METRICS,
        KEY_QUANT_METRICS,
        KEY_LAST_MODEL_ID,
        KEY_SUGGESTIONS_APPLIED,
        KEY_FORECAST,
        KEY_ACTIVE_MODEL,
        KEY_ACTIVE_MODEL_ID,
    ):
        st.session_state[k] = _DEFAULTS.get(k)


def clear_data_state() -> None:
    """Reset data + downstream state (used when asset changes)."""
    for k in (KEY_RAW_DF, KEY_DAILY_RETURNS, KEY_FEATURE_NAMES, KEY_DATA_LOADED_ASSET):
        st.session_state[k] = _DEFAULTS.get(k)
    clear_training_state()
