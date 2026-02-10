"""Training tab rendering."""
from __future__ import annotations

import traceback
from typing import Dict, Type

import streamlit as st
import torch.nn as nn

from gldpred.app import state
from gldpred.app.components.walkthrough import render_walkthrough_banner
from gldpred.app.controllers.training_controller import run_training
from gldpred.app.ui.components import show_diagnostics
from gldpred.registry import ModelRegistry


def render(
    t: Dict[str, str],
    lang: str,
    arch_map: Dict[str, Type[nn.Module]],
) -> None:
    """Render the Train tab."""
    render_walkthrough_banner(t, "tab_train")
    st.header(t["train_header"])
    st.info(t["train_info"])

    df = state.get(state.KEY_RAW_DF)
    if df is None:
        st.warning(t["train_warn_no_data"])
        return

    asset = state.get(state.KEY_ASSET, "GLD")
    feature_names = state.get(state.KEY_FEATURE_NAMES, [])

    # Training mode
    mode = st.radio(
        t["train_mode"],
        [t["train_mode_new"], t["train_mode_finetune"]],
        horizontal=True,
        key="train_mode_radio",
    )
    is_finetune = mode == t["train_mode_finetune"]

    # Custom label
    label_input = st.text_input(
        t["train_label"], "", key="model_label_input",
        help=t["train_label_help"],
    )

    # Early stopping option
    use_early_stopping = st.checkbox(
        "⏹️ Early Stopping (detener si no mejora)" if lang == "es" else "⏹️ Early Stopping (stop if no improvement)",
        value=True,
        key="early_stopping_checkbox",
        help="Detiene el entrenamiento automáticamente cuando el loss de validación deja de mejorar" if lang == "es" else "Stops training automatically when validation loss stops improving",
    )
    patience = 5
    if use_early_stopping:
        patience = st.slider(
            "Paciencia (epochs sin mejora)" if lang == "es" else "Patience (epochs without improvement)",
            min_value=3,
            max_value=15,
            value=5,
            key="early_stopping_patience",
        )

    # Architecture params from sidebar
    arch_name = st.session_state.get("architecture", "TCN")
    hidden_size = st.session_state.get("hidden_size", 64)
    num_layers = st.session_state.get("num_layers", 2)
    forecast_steps = st.session_state.get("forecast_steps", 20)
    seq_length = st.session_state.get("seq_length", 20)
    epochs = st.session_state.get("epochs", 50)
    batch_size = st.session_state.get("batch_size", 32)
    learning_rate = st.session_state.get("learning_rate", 0.001)
    quantiles = (0.1, 0.5, 0.9)

    # -- Fine-tune: select base model --
    base_model_id = None
    if is_finetune:
        registry = ModelRegistry()
        saved = registry.list_models(asset=asset)
        if not saved:
            st.warning(t["registry_no_models"])
            return
        labels = [
            f"{m.get('label', m['model_id'])} -- {m.get('created_at', '?')[:16]}"
            for m in saved
        ]
        choice = st.selectbox(t["train_select_model"], labels, key="finetune_select")
        base_model_id = saved[labels.index(choice)]["model_id"]
        epochs = st.slider(
            t["train_finetune_epochs"], 5, 100, 20, key="finetune_epochs",
        )

    # -- Train button --
    btn_label = t["train_finetune_btn"] if is_finetune else t["train_btn"]
    if not st.button(btn_label, key="btn_train"):
        show_diagnostics(t, lang)
        return

    try:
        with st.spinner(t["train_spinner"]):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _on_epoch(epoch, total, history):
                pct = (epoch + 1) / total
                progress_bar.progress(pct)
                train_loss = history["train_loss"][-1]
                val_loss = history["val_loss"][-1]
                status_text.text(
                    f"Epoch {epoch + 1}/{total} -- "
                    f"train: {train_loss:.5f} | val: {val_loss:.5f}"
                )

            result = run_training(
                df=df,
                feature_names=feature_names,
                daily_returns=state.get(state.KEY_DAILY_RETURNS),
                arch_map=arch_map,
                arch_name=arch_name,
                hidden_size=hidden_size,
                num_layers=num_layers,
                forecast_steps=forecast_steps,
                seq_length=seq_length,
                quantiles=quantiles,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                is_finetune=is_finetune,
                base_model_id=base_model_id,
                label_input=label_input,
                on_epoch=_on_epoch,
                early_stopping=use_early_stopping,
                patience=patience,
            )

            st.success(t["train_success"].format(model_id=result.model_id))
            st.info(t["train_label_saved_as"].format(label=result.label))
            if use_early_stopping and hasattr(state.get(state.KEY_TRAINER), 'best_epoch'):
                trainer = state.get(state.KEY_TRAINER)
                st.info(f"🎯 Mejor epoch: {trainer.best_epoch + 1} (loss: {trainer.best_val_loss:.6f})")

    except Exception as exc:
        st.error(t["train_error"].format(err=exc))
        st.code(traceback.format_exc())
        return

    show_diagnostics(t, lang)
