"""Sidebar rendering for the Streamlit app."""
from __future__ import annotations

from typing import Dict, List

import streamlit as st

from gldpred.app import state
from gldpred.app.config_suggester import suggest_training_config, get_config_rationale
from gldpred.app.controllers.model_loader import (
    get_active_model,
    list_asset_models,
    load_model_from_registry,
)
from gldpred.app.glossary import info_term
from gldpred.app.ui.components import (
    BATCH_SIZE_OPTIONS,
    HIDDEN_SIZE_OPTIONS,
    LR_OPTIONS,
)
from gldpred.config import SUPPORTED_ASSETS
from gldpred.i18n import LANGUAGES, STRINGS
from gldpred.registry import ModelAssignments


def _t() -> Dict[str, str]:
    code = st.session_state.get(state.KEY_LANGUAGE, "en")
    return STRINGS.get(code, STRINGS["en"])


def _lang() -> str:
    return st.session_state.get(state.KEY_LANGUAGE, "en")


def render_sidebar(arch_options: List[str]) -> None:
    """Render the configuration sidebar."""
    t = _t()
    ss = st.session_state

    with st.sidebar:
        st.header(t["sidebar_config"])

        # Language
        lang_label = st.selectbox(
            "Language / Idioma",
            list(LANGUAGES.keys()),
            index=0,
            key="lang_select",
        )
        ss[state.KEY_LANGUAGE] = LANGUAGES[lang_label]
        t = _t()

        # Asset selection
        prev_asset = ss.get(state.KEY_ASSET, "GLD")
        asset = st.selectbox(
            t["sidebar_asset"], SUPPORTED_ASSETS, key="_asset_select",
        )
        if asset != prev_asset:
            state.clear_data_state()
        ss[state.KEY_ASSET] = asset

        # Data settings
        st.subheader(t["sidebar_data_settings"])
        st.caption(t["sidebar_date_range"])

        # Active Model selection
        st.subheader(t["sidebar_active_model"])
        _render_model_selector(t, asset)

        # Model settings
        st.subheader(t["sidebar_model_settings"])
        info_term(t["sidebar_model_arch"], "architecture", _lang())
        st.selectbox(
            t["sidebar_model_arch"],
            arch_options,
            index=2,
            key="architecture",
            label_visibility="collapsed",
        )
        info_term(t["sidebar_forecast_steps"], "forecast_steps", _lang())
        st.slider(
            t["sidebar_forecast_steps"], 5, 60, 20,
            key="forecast_steps",
            label_visibility="collapsed",
        )

        # Training settings
        st.subheader(t["sidebar_training_settings"])
        
        # Auto-Config button
        df = ss.get(state.KEY_RAW_DF)
        if df is not None and len(df) > 0:
            if st.button(
                "🤖 Auto-Config" if _lang() == "en" else "🤖 Configuración Automática",
                help="Sugerir hiperparámetros basados en asset y volatilidad" if _lang() == "es" else "Suggest hyperparameters based on asset and volatility",
            ):
                suggested = suggest_training_config(asset, df)
                rationale = get_config_rationale(asset, df, suggested)
                
                # Update session state with suggestions
                for key, value in suggested.items():
                    ss[key] = value
                
                st.success(
                    f"✅ Configuración aplicada" if _lang() == "es" else f"✅ Configuration applied"
                )
                st.caption(rationale)
                st.rerun()

        def _consume(key, default):
            return ss.pop(f"_sugg_{key}", ss.get(key, default))

        info_term(t["sidebar_seq_length"], "seq_length", _lang())
        st.slider(
            t["sidebar_seq_length"], 10, 60,
            _consume("seq_length", 20),
            key="seq_length",
            label_visibility="collapsed",
        )

        info_term(t["sidebar_hidden_size"], "hidden_size", _lang())
        st.select_slider(
            t["sidebar_hidden_size"],
            options=HIDDEN_SIZE_OPTIONS,
            value=_consume("hidden_size", 64),
            key="hidden_size",
            label_visibility="collapsed",
        )

        info_term(t["sidebar_num_layers"], "num_layers", _lang())
        st.slider(
            t["sidebar_num_layers"], 1, 4,
            _consume("num_layers", 2),
            key="num_layers",
            label_visibility="collapsed",
        )

        info_term(t["sidebar_epochs"], "epochs", _lang())
        st.slider(
            t["sidebar_epochs"], 10, 200,
            _consume("epochs", 50),
            key="epochs",
            label_visibility="collapsed",
        )

        info_term(t["sidebar_batch_size"], "batch_size", _lang())
        st.select_slider(
            t["sidebar_batch_size"],
            options=BATCH_SIZE_OPTIONS,
            value=_consume("batch_size", 32),
            key="batch_size",
            label_visibility="collapsed",
        )

        info_term(t["sidebar_learning_rate"], "learning_rate", _lang())
        st.select_slider(
            t["sidebar_learning_rate"],
            options=LR_OPTIONS,
            value=_consume("learning_rate", 0.001),
            key="learning_rate",
            label_visibility="collapsed",
        )

        # Action Plan settings
        st.subheader(t["sidebar_action_plan"])
        st.slider(
            t["sidebar_tp_horizon"],
            1, 60, 10,
            key="tp_horizon",
        )
        st.number_input(
            t["sidebar_tp_take_profit"],
            min_value=0.5, max_value=50.0, value=5.0, step=0.5,
            key="tp_take_profit",
        )
        st.number_input(
            t["sidebar_tp_stop_loss"],
            min_value=0.5, max_value=50.0, value=3.0, step=0.5,
            key="tp_stop_loss",
        )
        st.number_input(
            t["sidebar_tp_min_return"],
            min_value=0.0, max_value=50.0, value=1.0, step=0.5,
            key="tp_min_return",
        )
        st.slider(
            t["sidebar_tp_risk_aversion"],
            0.0, 2.0, 0.5, step=0.1,
            key="tp_risk_aversion",
        )
        st.number_input(
            t["sidebar_tp_investment"],
            min_value=100.0, max_value=1_000_000.0, value=10_000.0,
            step=500.0,
            key="tp_investment",
        )

        # About
        st.divider()
        st.subheader(t["sidebar_about"])
        st.caption(t["sidebar_about_text"])


def _render_model_selector(t: Dict[str, str], asset: str) -> None:
    """Render saved-model selector within the sidebar."""
    models = list_asset_models(asset)
    if not models:
        st.caption(t["sidebar_no_models"])
        return

    # Build display labels
    assignments = ModelAssignments()
    primary_id = assignments.get(asset)
    labels: List[str] = []
    ids: List[str] = []
    default_idx = 0

    for i, meta in enumerate(models):
        mid = meta["model_id"]
        lab = meta.get("label", mid)
        arch = meta.get("architecture", "?")
        badge = "⭐ " if mid == primary_id else ""
        labels.append(f"{badge}{lab} ({arch})")
        ids.append(mid)

        # Pre-select: active model > primary > first
        active_id = state.get(state.KEY_ACTIVE_MODEL_ID)
        if active_id == mid:
            default_idx = i
        elif active_id is None and mid == primary_id:
            default_idx = i

    chosen_label = st.selectbox(
        t["sidebar_select_model"],
        labels,
        index=default_idx,
        key="_sidebar_model_select",
        label_visibility="collapsed",
    )
    chosen_id = ids[labels.index(chosen_label)]

    # Load if different from current active
    active = get_active_model()
    if active is None or active.model_id != chosen_id:
        if st.button(
            "📥 " + (
                "Cargar modelo" if _lang() == "es" else "Load model"
            ),
            key="btn_load_sidebar_model",
        ):
            try:
                bundle = load_model_from_registry(chosen_id)
                st.success(t["sidebar_model_loaded"].format(label=bundle.label))
                st.rerun()
            except Exception as e:
                st.error(str(e))
    else:
        st.caption(
            "✅ " + (
                f"Activo: {active.label}"
                if _lang() == "es"
                else f"Active: {active.label}"
            )
        )
