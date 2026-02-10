"""Guided empty-state components for when data/model/forecast are missing.

Instead of a bare ``st.warning(...)`` these provide step-by-step guidance
so the user knows exactly what to do next.
"""
from __future__ import annotations

from typing import Dict

import streamlit as st


def show_empty_no_data(t: Dict[str, str]) -> None:
    """Shown when no market data has been loaded."""
    st.info(
        t.get(
            "empty_no_data",
            "📊 No market data loaded yet. Go to the **Data** tab "
            "and click **Load Data** to get started.",
        )
    )


def show_empty_no_model(t: Dict[str, str]) -> None:
    """Shown when no model is available for the selected asset."""
    st.warning(
        t.get(
            "empty_no_model",
            "🤖 No model available for this asset. You can:\n"
            "1. **Train** a new model in the Train tab\n"
            "2. **Load** an existing model from the sidebar\n"
            "3. **Assign** a primary model in the Models tab",
        )
    )


def show_empty_no_forecast(t: Dict[str, str]) -> None:
    """Shown when a model exists but no forecast has been generated."""
    st.info(
        t.get(
            "empty_no_forecast",
            "🔮 No forecast generated yet. Go to the **Forecast** tab "
            "to generate predictions, then come back here.",
        )
    )
