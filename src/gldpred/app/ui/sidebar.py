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
from gldpred.config.assets import ASSET_CATALOG, BENCHMARK_ASSET, RISK_LEVELS
from gldpred.i18n import LANGUAGES, STRINGS
from gldpred.registry import ModelAssignments


def _t() -> Dict[str, str]:
    code = st.session_state.get(state.KEY_LANGUAGE, "es")
    return STRINGS.get(code, STRINGS["es"])


def _lang() -> str:
    return st.session_state.get(state.KEY_LANGUAGE, "es")


def render_sidebar(arch_options: List[str]) -> None:
    """Render the configuration sidebar."""
    t = _t()
    ss = st.session_state

    with st.sidebar:
        st.header(t["sidebar_config"])

        # Language — persisted via query_params
        _CODE_TO_LABEL = {v: k for k, v in LANGUAGES.items()}
        # Restore from query params if available
        qp_lang = st.query_params.get("lang")
        if qp_lang in LANGUAGES.values() and state.KEY_LANGUAGE not in ss:
            ss[state.KEY_LANGUAGE] = qp_lang
        current_code = ss.get(state.KEY_LANGUAGE, "es")
        current_label = _CODE_TO_LABEL.get(current_code, "Español")
        lang_keys = list(LANGUAGES.keys())
        default_idx = lang_keys.index(current_label) if current_label in lang_keys else 0
        lang_label = st.selectbox(
            "🌐 Idioma / Language",
            lang_keys,
            index=default_idx,
            key="lang_select",
        )
        new_code = LANGUAGES[lang_label]
        if new_code != ss.get(state.KEY_LANGUAGE):
            ss[state.KEY_LANGUAGE] = new_code
            st.query_params["lang"] = new_code
        t = _t()

        # Asset selection — grouped by risk tier
        risk_filter = st.selectbox(
            t["sidebar_risk_tier"],
            [t["sidebar_all_tiers"]] + [
                t[f"risk_level_{r}"] for r in RISK_LEVELS
            ],
            index=0,
            key="_risk_tier_filter",
        )

        # Build filtered asset list
        if risk_filter == t["sidebar_all_tiers"]:
            visible_assets = list(SUPPORTED_ASSETS)
        else:
            # Reverse-lookup the risk level from translated label
            _risk_map = {t[f"risk_level_{r}"]: r for r in RISK_LEVELS}
            selected_risk = _risk_map.get(risk_filter, "medium")
            visible_assets = [
                a for a in SUPPORTED_ASSETS
                if ASSET_CATALOG.get(a, None) is not None
                and ASSET_CATALOG[a].risk_level == selected_risk
            ]

        # Format display labels: "SPY — SPDR S&P 500 ETF [📊]"
        def _label(ticker: str) -> str:
            info = ASSET_CATALOG.get(ticker)
            if info is None:
                return ticker
            badge = " 📊" if ticker == BENCHMARK_ASSET else ""
            return f"{ticker} — {info.name}{badge}"

        asset_labels = [_label(a) for a in visible_assets]
        prev_asset = ss.get(state.KEY_ASSET, "SPY")

        # Default index: previous selection or first
        if prev_asset in visible_assets:
            default_asset_idx = visible_assets.index(prev_asset)
        else:
            default_asset_idx = 0

        chosen_label = st.selectbox(
            t["sidebar_asset"], asset_labels,
            index=default_asset_idx, key="_asset_select",
        )
        # Extract ticker from label
        asset = chosen_label.split(" — ")[0].strip() if " — " in chosen_label else chosen_label

        if asset != prev_asset:
            state.clear_data_state()
        ss[state.KEY_ASSET] = asset

        # Show asset metadata badge
        _info = ASSET_CATALOG.get(asset)
        if _info is not None:
            risk_label = t.get(f"risk_level_{_info.risk_level}", _info.risk_level)
            horizon_labels = ", ".join(
                t.get(f"horizon_{h}", h) for h in _info.investment_horizon
            )
            role_label = t.get(f"role_{_info.role}", _info.role)
            st.caption(
                f"⚡ {risk_label} · 📅 {horizon_labels} · 🎯 {role_label}"
            )

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
                t["sidebar_auto_config"],
                help=t["sidebar_auto_config_help"],
            ):
                suggested = suggest_training_config(asset, df)
                rationale = get_config_rationale(asset, df, suggested)
                
                # Update session state with suggestions
                for key, value in suggested.items():
                    ss[key] = value
                
                st.success(t["sidebar_config_applied"])
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
            min_value=0.01, max_value=1_000_000.0, value=10_000.0,
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
            t["sidebar_load_model"],
            key="btn_load_sidebar_model",
        ):
            try:
                bundle = load_model_from_registry(chosen_id)
                st.success(t["sidebar_model_loaded"].format(label=bundle.label))
                st.rerun()
            except Exception as e:
                st.error(str(e))
    else:
        st.caption(t["sidebar_model_active"].format(label=active.label))
