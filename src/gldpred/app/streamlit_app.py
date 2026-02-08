"""Streamlit GUI — Multi-Asset Price Prediction with Deep Learning.

Seven tabs: Data · Train · Forecast · Recommendation · Evaluation ·
Compare · Tutorial.
"""
from __future__ import annotations

import traceback
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from gldpred.app.plots import create_fan_chart, create_loss_chart
from gldpred.config import (
    ASSET_CATALOG,
    SUPPORTED_ASSETS,
    AppConfig,
    DecisionConfig,
    ModelConfig,
    TrainingConfig,
)
from gldpred.data import AssetDataLoader
from gldpred.decision import (
    DecisionEngine,
    PortfolioComparator,
    Recommendation,
    RecommendationHistory,
    RiskMetrics,
)
from gldpred.diagnostics import DiagnosticsAnalyzer
from gldpred.evaluation import ModelEvaluator
from gldpred.features import FeatureEngineering
from gldpred.i18n import LANGUAGES, STRINGS
from gldpred.inference import TrajectoryPredictor
from gldpred.models import GRUForecaster, LSTMForecaster, TCNForecaster
from gldpred.registry import ModelAssignments, ModelRegistry
from gldpred.training import ModelTrainer

# ── Architecture lookup ──────────────────────────────────────────────────
_ARCH_MAP = {
    "TCN": TCNForecaster,
    "GRU": GRUForecaster,
    "LSTM": LSTMForecaster,
}

# ── Sidebar allowed values (used by Apply Suggestions) ───────────────────
_HIDDEN_SIZE_OPTIONS: List[int] = [32, 64, 128]
_BATCH_SIZE_OPTIONS: List[int] = [16, 32, 64, 128]
_LR_OPTIONS: List[float] = [0.0001, 0.0005, 0.001, 0.005, 0.01]

# ── i18n helper ──────────────────────────────────────────────────────────


def _t() -> Dict[str, str]:
    lang = st.session_state.get("lang", "en")
    return STRINGS[lang]


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    """Entry-point called from ``app.py``."""
    t = _t()

    st.set_page_config(
        page_title=t["page_title"],
        page_icon="📈",
        layout="wide",
    )
    st.title(t["app_title"])
    st.caption(t["app_subtitle"])

    _sidebar()

    tabs = st.tabs([
        t["tab_data"],
        t["tab_train"],
        t["tab_forecast"],
        t["tab_recommendation"],
        t["tab_evaluation"],
        t["tab_compare"],
        t["tab_tutorial"],
    ])

    with tabs[0]:
        _tab_data()
    with tabs[1]:
        _tab_train()
    with tabs[2]:
        _tab_forecast()
    with tabs[3]:
        _tab_recommendation()
    with tabs[4]:
        _tab_evaluation()
    with tabs[5]:
        _tab_compare()
    with tabs[6]:
        _tab_tutorial()


# ======================================================================
# SIDEBAR
# ======================================================================
def _sidebar() -> None:
    t = _t()
    with st.sidebar:
        # Language selector
        lang_label = st.selectbox(
            "🌐 Language",
            list(LANGUAGES.keys()),
            index=0,
        )
        st.session_state["lang"] = LANGUAGES[lang_label]
        t = _t()  # refresh after selection

        st.header(t["sidebar_config"])

        # ── Asset ────────────────────────────────────────────────────
        st.selectbox(
            t["sidebar_asset"],
            list(SUPPORTED_ASSETS),
            index=0,
            key="asset",
        )
        st.caption(t["sidebar_date_range"])

        # ── Model ────────────────────────────────────────────────────
        st.subheader(t["sidebar_model_settings"])
        st.selectbox(
            t["sidebar_model_arch"], ["TCN", "GRU", "LSTM"],
            index=0, key="architecture",
        )
        st.slider(
            t["sidebar_forecast_steps"], 5, 60, 20, key="forecast_steps",
        )

        # ── Training ─────────────────────────────────────────────────
        st.subheader(t["sidebar_training_settings"])
        
        # Apply suggested values if they exist (from diagnostics)
        seq_length_val = st.session_state.pop("_sugg_seq_length", 20)
        hidden_size_val = st.session_state.pop("_sugg_hidden_size", 64)
        num_layers_val = st.session_state.pop("_sugg_num_layers", 2)
        epochs_val = st.session_state.pop("_sugg_epochs", 50)
        batch_size_val = st.session_state.pop("_sugg_batch_size", 32)
        learning_rate_val = st.session_state.pop("_sugg_learning_rate", 0.001)
        
        st.slider(t["sidebar_seq_length"], 10, 60, seq_length_val, key="seq_length")
        st.select_slider(
            t["sidebar_hidden_size"], _HIDDEN_SIZE_OPTIONS,
            value=hidden_size_val, key="hidden_size",
        )
        st.slider(t["sidebar_num_layers"], 1, 4, num_layers_val, key="num_layers")
        st.slider(t["sidebar_epochs"], 10, 200, epochs_val, key="epochs")
        st.select_slider(
            t["sidebar_batch_size"], _BATCH_SIZE_OPTIONS,
            value=batch_size_val, key="batch_size",
        )
        st.select_slider(
            t["sidebar_learning_rate"], _LR_OPTIONS,
            value=learning_rate_val, key="learning_rate",
        )

        # ── About ────────────────────────────────────────────────────
        st.divider()
        st.subheader(t["sidebar_about"])
        st.info(t["sidebar_about_text"])


# ======================================================================
# TAB 1 — DATA
# ======================================================================
def _tab_data() -> None:
    t = _t()
    st.header(t["data_header"])
    st.info(t["data_info"])

    if st.button(t["data_load_btn"], key="btn_load"):
        try:
            with st.spinner(t["data_loading_spinner"]):
                asset = st.session_state.get("asset", "GLD")
                loader = AssetDataLoader(ticker=asset)
                df = loader.load_data()
                daily_ret = loader.daily_returns()

                eng = FeatureEngineering()
                df = eng.add_technical_indicators(df)
                feat_df = eng.select_features(df)
                feature_names = feat_df.columns.tolist()

                st.session_state["raw_df"] = df
                st.session_state["daily_returns"] = daily_ret
                st.session_state["feature_names"] = feature_names

                n = len(df)
                start = str(df.index[0].date())
                end = str(df.index[-1].date())
                st.success(t["data_load_success"].format(n=n, asset=asset, start=start, end=end))
        except Exception as exc:
            st.error(t["data_load_error"].format(err=exc))
            return

    df = st.session_state.get("raw_df")
    if df is not None and not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t["data_metric_records"], len(df))
        c2.metric(t["data_metric_price"], f"${df['Close'].iloc[-1]:.2f}")
        pct = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
        c3.metric(t["data_metric_change"], f"{pct:+.2f}%")
        c4.metric(t["data_metric_features"], len(st.session_state.get("feature_names", [])))

        # Price chart
        st.subheader(t["data_price_history"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], mode="lines",
            name="Close", line=dict(color="#FFD700", width=2),
        ))
        if "sma_50" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["sma_50"], mode="lines",
                name="SMA 50", line=dict(color="#1f77b4", width=1, dash="dash"),
            ))
        if "sma_200" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["sma_200"], mode="lines",
                name="SMA 200", line=dict(color="#ff7f0e", width=1, dash="dash"),
            ))
        fig.update_layout(
            xaxis_title=t["axis_date"],
            yaxis_title=t["axis_price"],
            template="plotly_dark",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Preview
        with st.expander(t["data_preview"]):
            st.dataframe(df.tail(20))


# ======================================================================
# TAB 2 — TRAIN
# ======================================================================
def _tab_train() -> None:
    t = _t()
    st.header(t["train_header"])
    st.info(t["train_info"])

    # Show "suggestions applied" banner (set by Apply Suggestions)
    if st.session_state.pop("suggestions_applied", False):
        st.success(t["diag_applied_success"])

    df = st.session_state.get("raw_df")
    if df is None:
        st.warning(t["train_warn_no_data"])
        return

    # Training mode
    mode = st.radio(
        t["train_mode"],
        [t["train_mode_new"], t["train_mode_finetune"]],
        horizontal=True,
    )
    is_finetune = mode == t["train_mode_finetune"]

    registry = ModelRegistry()
    selected_model_id: str | None = None
    selected_meta: Dict[str, Any] | None = None

    if is_finetune:
        asset = st.session_state.get("asset", "GLD")
        arch = st.session_state.get("architecture", "TCN")
        saved = registry.list_models(asset=asset, architecture=arch)
        if not saved:
            st.warning(t["registry_no_models"])
            return
        labels = [
            f"{m.get('label', m['model_id'])} — "
            f"{m.get('created_at', '?')[:16]}"
            for m in saved
        ]
        choice = st.selectbox(t["train_select_model"], labels)
        idx = labels.index(choice)
        selected_model_id = saved[idx]["model_id"]
        selected_meta = saved[idx]

    # Custom model label input
    model_label = st.text_input(
        t["train_label"],
        placeholder=f"{st.session_state.get('asset', 'GLD')}_{st.session_state.get('architecture', 'TCN')}_v1",
        help=t["train_label_help"],
        key="model_label_input",
    )

    btn_label = t["train_finetune_btn"] if is_finetune else t["train_btn"]
    if not st.button(btn_label, key="btn_train"):
        # Even when not training, show prior diagnostics if available
        _show_diagnostics()
        return

    try:
        with st.spinner(t["train_spinner"]):
            # ── Clear previous diagnostics state ──────────────────────
            st.session_state["suggestions_applied"] = False

            asset = st.session_state.get("asset", "GLD")
            arch = st.session_state.get("architecture", "TCN")
            forecast_steps = st.session_state.get("forecast_steps", 20)
            seq_length = st.session_state.get("seq_length", 20)
            hidden_size = st.session_state.get("hidden_size", 64)
            num_layers = st.session_state.get("num_layers", 2)
            epochs = st.session_state.get("epochs", 50)
            batch_size = st.session_state.get("batch_size", 32)
            lr = st.session_state.get("learning_rate", 0.001)
            quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

            daily_returns = st.session_state["daily_returns"]
            feature_names: list[str] = st.session_state["feature_names"]

            # ── For fine-tune: override with saved model config ───────
            if is_finetune and selected_meta:
                saved_cfg = selected_meta.get("config", {})
                forecast_steps = saved_cfg.get("forecast_steps", forecast_steps)
                seq_length = saved_cfg.get("seq_length", seq_length)
                hidden_size = saved_cfg.get("hidden_size", hidden_size)
                num_layers = saved_cfg.get("num_layers", num_layers)
                quantiles = tuple(
                    saved_cfg.get("quantiles", list(quantiles))
                )

                # Validate feature dimension match
                saved_features = selected_meta.get("feature_names", [])
                if saved_features and len(saved_features) != len(feature_names):
                    st.error(
                        t["train_feature_mismatch"].format(
                            expected=len(saved_features),
                            got=len(feature_names),
                        )
                    )
                    return

            # ── Create sequences ──────────────────────────────────────
            eng = FeatureEngineering()
            X, y = eng.create_sequences(
                df[feature_names].values,
                daily_returns.values,
                seq_length=seq_length,
                forecast_steps=forecast_steps,
            )

            input_size = X.shape[2]
            model_cls = _ARCH_MAP[arch]
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if is_finetune and selected_model_id:
                model, scaler, _meta = registry.load_model(
                    selected_model_id,
                    model_cls,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    forecast_steps=forecast_steps,
                    quantiles=quantiles,
                )
                trainer = ModelTrainer(
                    model, quantiles=quantiles, device=device,
                )
                trainer.scaler = scaler          # BEFORE prepare_data
                train_loader, val_loader = trainer.prepare_data(
                    X, y,
                    test_size=0.2, batch_size=batch_size,
                    refit_scaler=False,
                )
            else:
                model = model_cls(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    forecast_steps=forecast_steps,
                    quantiles=quantiles,
                )
                trainer = ModelTrainer(
                    model, quantiles=quantiles, device=device,
                )
                train_loader, val_loader = trainer.prepare_data(
                    X, y, test_size=0.2, batch_size=batch_size,
                )

            # Progress bar
            progress_bar = st.progress(0)
            epoch_placeholder = st.empty()

            def _on_epoch(epoch: int, total: int, history: Dict) -> None:
                progress_bar.progress((epoch + 1) / total)
                tl = history["train_loss"][-1]
                vl = history["val_loss"][-1]
                epoch_placeholder.text(
                    f"Epoch {epoch + 1}/{total}  — train: {tl:.6f}  val: {vl:.6f}"
                )

            history = trainer.train(
                train_loader, val_loader,
                epochs=epochs, learning_rate=lr,
                on_epoch=_on_epoch,
            )
            train_losses = history["train_loss"]
            val_losses = history["val_loss"]
            progress_bar.progress(1.0)

            # Diagnostics
            diag_result = DiagnosticsAnalyzer.analyze({
                "train_loss": train_losses,
                "val_loss": val_losses,
            })

            # Evaluation on val set
            split = int(len(X) * 0.8)
            X_val_raw = X[split:]
            y_val = y[split:]
            pred_val = trainer.predict(X_val_raw)

            evaluator = ModelEvaluator()
            median_idx = list(quantiles).index(0.5)
            traj_metrics = evaluator.evaluate_trajectory(y_val, pred_val[:, :, median_idx])
            quant_metrics = evaluator.evaluate_quantiles(y_val, pred_val, quantiles)

            # Save to registry
            config_dict = {
                "asset": asset,
                "architecture": arch,
                "forecast_steps": forecast_steps,
                "seq_length": seq_length,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "quantiles": list(quantiles),
            }
            training_summary = {
                "epochs": epochs,
                "learning_rate": lr,
                "batch_size": batch_size,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "diagnostics_verdict": diag_result.verdict,
            }
            eval_summary = {**traj_metrics, **quant_metrics}

            model_id = registry.save_model(
                trainer.model,
                trainer.scaler,
                config_dict,
                feature_names,
                training_summary,
                eval_summary,
                label=model_label.strip() or None,
            )

            # Get saved label from metadata
            saved_meta = registry.list_models()
            saved_label = next(
                (m["label"] for m in saved_meta if m["model_id"] == model_id),
                model_id
            )

            # Store in session
            st.session_state["trainer"] = trainer
            st.session_state["train_losses"] = train_losses
            st.session_state["val_losses"] = val_losses
            st.session_state["diag_result"] = diag_result
            st.session_state["traj_metrics"] = traj_metrics
            st.session_state["quant_metrics"] = quant_metrics
            st.session_state["last_model_id"] = model_id

            st.success(t["train_label_saved_as"].format(label=saved_label))

    except Exception as exc:
        st.error(t["train_error"].format(err=exc))
        st.code(traceback.format_exc())
        return

    # Show diagnostics
    _show_diagnostics()

    # Asset model assignment section
    st.divider()
    _show_asset_assignment()

    # Registry management section
    st.divider()
    with st.expander(t["registry_delete_header"]):
        _show_registry_deletion()


def _show_asset_assignment() -> None:
    """Show controls to assign a model as the primary for this asset."""
    t = _t()
    asset = st.session_state.get("asset", "GLD")
    registry = ModelRegistry()
    assignments = ModelAssignments()

    saved = registry.list_models(asset=asset)
    if not saved:
        return

    with st.expander(t["assign_header"]):
        current_id = assignments.get(asset)
        if current_id:
            # Find label of current assignment
            label = current_id
            for m in saved:
                if m["model_id"] == current_id:
                    label = m.get("label", current_id)
                    break
            st.success(f"{t['assign_current']}: **{label}**")

            if st.button(t["assign_unassign_btn"], key="btn_unassign"):
                assignments.unassign(asset)
                st.success(t["assign_removed"].format(asset=asset))
                st.rerun()
        else:
            st.info(t["assign_none"])

        # Select model to assign
        labels = [
            f"{m.get('label', m['model_id'])} — "
            f"{m.get('created_at', '?')[:16]}"
            for m in saved
        ]
        choice = st.selectbox(
            t["assign_header"], labels,
            key="assign_model_select",
        )
        idx = labels.index(choice)
        model_id = saved[idx]["model_id"]
        model_label = saved[idx].get("label", model_id)

        if st.button(t["assign_btn"], key="btn_assign"):
            assignments.assign(asset, model_id)
            st.success(t["assign_success"].format(asset=asset, label=model_label))
            st.rerun()


def _show_diagnostics() -> None:
    t = _t()
    diag = st.session_state.get("diag_result")
    train_losses = st.session_state.get("train_losses")
    val_losses = st.session_state.get("val_losses")
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

    c1, c2, c3 = st.columns(3)
    c1.metric(t["diag_verdict"], v_label)
    c2.metric(t["diag_best_epoch"], diag.best_epoch + 1)
    c3.metric(t["diag_gen_gap"], f"{diag.generalization_gap:.4f}")

    st.markdown(f"**{t['diag_explanation']}:** {diag.explanation}")
    if diag.suggestions:
        st.markdown(f"**{t['diag_suggestions']}:**")
        for s in diag.suggestions:
            st.markdown(f"- {s}")

    # ── "Apply Suggestions" button ────────────────────────────────────
    if diag.verdict != "healthy":
        already_applied = st.session_state.get("suggestions_applied", False)
        if already_applied:
            st.info(t["diag_applied_success"])
        else:
            if st.button(t["diag_apply_btn"], key="btn_apply_suggestions"):
                _apply_suggestions(diag.verdict, diag.best_epoch)
                st.session_state["suggestions_applied"] = True
                st.rerun()

    # ── Loss chart with markers ───────────────────────────────────────
    if train_losses and val_losses:
        fig = create_loss_chart(
            train_losses,
            val_losses,
            best_epoch=diag.best_epoch,
            verdict=diag.verdict,
        )
        st.plotly_chart(fig, use_container_width=True)


def _apply_suggestions(verdict: str, best_epoch: int) -> None:
    """Store suggested config values for sidebar widgets.

    Stores suggestions in temporary keys (_sugg_*) that the sidebar will
    consume on next rerun. This avoids modifying widget keys after widgets
    have been instantiated.
    """
    ss = st.session_state

    if verdict == "overfitting":
        # Reduce epochs to best_epoch + small buffer
        ss["_sugg_epochs"] = min(max(best_epoch + 5, 10), 200)
        # Step hidden_size down
        current_hs = ss.get("hidden_size", 64)
        ss["_sugg_hidden_size"] = _step_value(current_hs, _HIDDEN_SIZE_OPTIONS, -1)
        # Reduce layers
        nl = ss.get("num_layers", 2)
        if nl > 1:
            ss["_sugg_num_layers"] = nl - 1

    elif verdict == "underfitting":
        # More epochs
        ss["_sugg_epochs"] = min(ss.get("epochs", 50) + 50, 200)
        # Step hidden_size up
        current_hs = ss.get("hidden_size", 64)
        ss["_sugg_hidden_size"] = _step_value(current_hs, _HIDDEN_SIZE_OPTIONS, +1)
        # More layers
        nl = ss.get("num_layers", 2)
        if nl < 4:
            ss["_sugg_num_layers"] = nl + 1
        # Step LR up
        current_lr = ss.get("learning_rate", 0.001)
        ss["_sugg_learning_rate"] = _step_value(current_lr, _LR_OPTIONS, +1)

    elif verdict == "noisy":
        # Step LR down
        current_lr = ss.get("learning_rate", 0.001)
        ss["_sugg_learning_rate"] = _step_value(current_lr, _LR_OPTIONS, -1)
        # Step batch_size up
        current_bs = ss.get("batch_size", 32)
        ss["_sugg_batch_size"] = _step_value(current_bs, _BATCH_SIZE_OPTIONS, +1)
        # Increase sequence length
        sl = ss.get("seq_length", 20)
        ss["_sugg_seq_length"] = min(sl + 10, 60)


def _step_value(current: float, options: list, direction: int) -> float:
    """Step a select-slider value up (+1) or down (-1) within options.
    
    Returns the new value after stepping.
    """
    try:
        idx = options.index(current)
    except ValueError:
        # Snap to the closest allowed value
        idx = min(
            range(len(options)), key=lambda i: abs(options[i] - current),
        )
    new_idx = max(0, min(len(options) - 1, idx + direction))
    return options[new_idx]


def _show_registry_deletion() -> None:
    """Show model deletion controls."""
    t = _t()
    registry = ModelRegistry()
    asset = st.session_state.get("asset", "GLD")
    
    all_models = registry.list_models()
    asset_models = registry.list_models(asset=asset)
    
    if not all_models:
        st.info(t["registry_no_models"])
        return
    
    st.markdown(f"**{len(all_models)}** total models · **{len(asset_models)}** {asset} models")
    
    # Delete single model
    st.subheader(t["registry_delete_single"])
    
    model_labels = [
        f"{m.get('label', m['model_id'])} ({m['architecture']}, {m.get('created_at', '?')[:10]})"
        for m in all_models
    ]
    selected = st.selectbox(
        t["registry_model_details"],
        model_labels,
        key="delete_model_select"
    )
    selected_model = all_models[model_labels.index(selected)]
    
    # Show details
    col1, col2, col3 = st.columns(3)
    col1.metric("Asset", selected_model["asset"])
    col2.metric("Architecture", selected_model["architecture"])
    col3.metric("Created", selected_model.get("created_at", "?")[:10])
    
    confirm_single = st.text_input(
        t["registry_confirm_single"],
        placeholder="DELETE",
        key="confirm_single_delete"
    )
    
    if st.button(t["registry_delete_btn"], key="btn_delete_single"):
        if confirm_single.strip() == "DELETE":
            try:
                registry.delete_model(selected_model["model_id"])
                st.success(t["registry_delete_success"].format(count=1))
                st.rerun()
            except Exception as e:
                st.error(t["registry_delete_error"].format(err=str(e)))
        else:
            st.warning("Confirmation text must be exactly: DELETE")
    
    st.divider()
    
    # Delete all models (with asset filter option)
    st.subheader(t["registry_delete_all"])
    
    delete_scope = st.radio(
        "Scope",
        [f"All models ({len(all_models)})", f"Only {asset} models ({len(asset_models)})"],
        key="delete_scope",
        horizontal=True,
    )
    delete_asset_only = asset if "Only" in delete_scope else None
    delete_count = len(asset_models) if delete_asset_only else len(all_models)
    
    if delete_count == 0:
        st.info(t["registry_no_models"])
        return
    
    st.warning(t["registry_confirm_all"].format(count=delete_count))
    confirm_all = st.text_input(
        t["registry_confirm_input"],
        placeholder="DELETE ALL",
        key="confirm_all_delete"
    )
    
    if st.button(t["registry_delete_btn"], key="btn_delete_all"):
        if confirm_all.strip() == "DELETE ALL":
            try:
                deleted = registry.delete_all_models(asset=delete_asset_only, confirmed=True)
                st.success(t["registry_delete_success"].format(count=deleted))
                st.rerun()
            except Exception as e:
                st.error(t["registry_delete_error"].format(err=str(e)))
        else:
            st.warning("Confirmation text must be exactly: DELETE ALL")


# ======================================================================
# TAB 3 — FORECAST
# ======================================================================
def _tab_forecast() -> None:
    t = _t()
    st.header(t["forecast_header"])
    st.info(t["forecast_info"])

    trainer: ModelTrainer | None = st.session_state.get("trainer")
    df = st.session_state.get("raw_df")
    if trainer is None or df is None:
        st.warning(t["forecast_warn_no_model"])
        return

    try:
        asset = st.session_state.get("asset", "GLD")
        feature_names = st.session_state.get("feature_names", [])
        seq_length = st.session_state.get("seq_length", 20)
        quantiles = trainer.quantiles  # property → tuple

        predictor = TrajectoryPredictor(trainer)
        forecast = predictor.predict_trajectory(df, feature_names, seq_length, asset)

        st.session_state["forecast"] = forecast

        # ── Fan chart ─────────────────────────────────────────────────
        st.subheader(t["forecast_fan_chart"])
        fig = create_fan_chart(
            df, forecast,
            x_label=t["axis_date"],
            y_label=t["axis_price"],
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Table ─────────────────────────────────────────────────────
        st.subheader(t["forecast_table"])
        K = forecast.returns_quantiles.shape[0]
        q_idx = {round(q, 2): i for i, q in enumerate(quantiles)}
        lo_i = q_idx.get(0.1, 0)
        med_i = q_idx.get(0.5, 1)
        hi_i = q_idx.get(0.9, 2)

        rows = []
        for k in range(K):
            rows.append({
                t["forecast_col_day"]: k + 1,
                t["forecast_col_date"]: str(forecast.dates[k].date()) if hasattr(forecast.dates[k], "date") else str(forecast.dates[k]),
                t["forecast_col_p10"]: f"${forecast.price_paths[k + 1, lo_i]:.2f}",
                t["forecast_col_p50"]: f"${forecast.price_paths[k + 1, med_i]:.2f}",
                t["forecast_col_p90"]: f"${forecast.price_paths[k + 1, hi_i]:.2f}",
                t["forecast_col_return"]: f"{forecast.returns_quantiles[k, med_i]:+.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    except Exception as exc:
        st.error(t["forecast_error"].format(err=exc))
        st.code(traceback.format_exc())



# ======================================================================
# TAB 4 — RECOMMENDATION
# ======================================================================
def _tab_recommendation() -> None:
    t = _t()
    st.header(t["reco_header"])
    st.markdown(t["reco_disclaimer"])
    st.info(t["reco_info"])

    trainer = st.session_state.get("trainer")
    forecast = st.session_state.get("forecast")
    df = st.session_state.get("raw_df")
    diag = st.session_state.get("diag_result")

    if trainer is None or forecast is None:
        st.warning(t["reco_warn_no_model"])
        return

    try:
        asset = st.session_state.get("asset", "GLD")
        decision_cfg = DecisionConfig()
        max_vol = decision_cfg.max_volatility.get(asset, decision_cfg.max_volatility["default"])
        horizon = st.slider(t["reco_decision_window"], 1, forecast.returns_quantiles.shape[0], min(5, forecast.returns_quantiles.shape[0]))

        engine = DecisionEngine(
            horizon_days=horizon,
            min_expected_return=decision_cfg.min_expected_return,
            max_volatility=max_vol,
        )

        verdict_str = diag.verdict if diag else None
        reco = engine.recommend(
            forecast.returns_quantiles,
            df,
            quantiles=forecast.quantiles,
            diagnostics_verdict=verdict_str,
        )

        # Record in history
        if "reco_history" not in st.session_state:
            st.session_state["reco_history"] = RecommendationHistory()
        st.session_state["reco_history"].add(asset, reco)

        # Display action and confidence
        action_label = {
            "BUY": t["reco_buy"],
            "HOLD": t["reco_hold"],
            "AVOID": t["reco_avoid"],
        }

        col1, col2 = st.columns(2)
        col1.markdown(f"### {t['reco_action']}: {action_label.get(reco.action, reco.action)}")
        col2.metric(t["reco_confidence"], f"{reco.confidence:.0f} / 100")

        # Market regime
        regime_map = {
            "trending_up": t["regime_trending_up"],
            "trending_down": t["regime_trending_down"],
            "ranging": t["regime_ranging"],
            "high_volatility": t["regime_high_volatility"],
            "unknown": t["regime_unknown"],
        }
        regime = reco.details.get("market_regime", "unknown")
        st.markdown(f"**{t['regime_header']}:** {regime_map.get(regime, regime)}")

        # Risk metrics
        st.subheader(t["risk_header"])
        risk = reco.risk
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric(t["risk_stop_loss"], f"{risk.stop_loss_pct:+.2f}%")
        rc2.metric(t["risk_take_profit"], f"{risk.take_profit_pct:+.2f}%")
        rc3.metric(t["risk_reward_ratio"], f"{risk.risk_reward_ratio:.2f}")
        rc4.metric(t["risk_max_drawdown"], f"{risk.max_drawdown_pct:.2f}%")

        vol_regime_map = {
            "low": t["risk_regime_low"],
            "normal": t["risk_regime_normal"],
            "high": t["risk_regime_high"],
        }
        st.markdown(
            f"**{t['risk_volatility_regime']}:** "
            f"{vol_regime_map.get(risk.volatility_regime, risk.volatility_regime)}"
        )

        # Rationale and warnings
        if reco.rationale:
            st.markdown(f"**{t['reco_rationale']}**")
            for r in reco.rationale:
                st.markdown(f"- {r}")

        if reco.warnings:
            st.markdown(f"**{t['reco_warnings']}**")
            for w in reco.warnings:
                st.warning(w)

        # Recommendation history
        _show_recommendation_history()

    except Exception as exc:
        st.error(t["reco_error"].format(err=exc))
        st.code(traceback.format_exc())


def _show_recommendation_history() -> None:
    """Display the in-session recommendation history."""
    t = _t()
    history: RecommendationHistory | None = st.session_state.get("reco_history")
    if not history or len(history) == 0:
        return

    with st.expander(t["reco_history_header"]):
        records = history.get_history()
        if not records:
            st.info(t["reco_history_empty"])
            return

        rows = []
        for rec in reversed(records):  # most recent first
            rows.append({
                t["compare_asset"]: rec["asset"],
                t["compare_action"]: rec["action"],
                t["compare_confidence"]: f"{rec['confidence']:.0f}",
                "Timestamp": rec["timestamp"][:19],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if st.button(t["reco_history_clear"], key="btn_clear_history"):
            history.clear()
            st.rerun()


# ======================================================================
# TAB 5 — EVALUATION
# ======================================================================
def _tab_evaluation() -> None:
    t = _t()
    st.header(t["eval_header"])
    st.info(t["eval_info"])

    traj = st.session_state.get("traj_metrics")
    quant = st.session_state.get("quant_metrics")

    if traj is None:
        st.warning(t["eval_warn_no_model"])
        return

    try:
        # Trajectory metrics
        st.subheader(t["eval_trajectory_metrics"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MSE", f"{traj.get('mse', 0):.6f}")
        c2.metric("RMSE", f"{traj.get('rmse', 0):.6f}")
        c3.metric("MAE", f"{traj.get('mae', 0):.6f}")
        c4.metric("Dir. Accuracy", f"{traj.get('directional_accuracy', 0):.2%}")

        # Per-step breakdown
        mae_per_step = traj.get("mae_per_step", [])
        if mae_per_step:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(1, len(mae_per_step) + 1)), y=mae_per_step, name="MAE"))
            fig.update_layout(
                xaxis_title=t["axis_day"],
                yaxis_title="MAE",
                template="plotly_dark",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Quantile calibration
        if quant:
            st.subheader(t["eval_quantile_metrics"])
            cov_cols = st.columns(3)
            cov_cols[0].metric(
                "P10 Coverage",
                f"{quant.get('q10_coverage', 0):.2%}",
            )
            cov_cols[1].metric(
                "P50 Coverage",
                f"{quant.get('q50_coverage', 0):.2%}",
            )
            cov_cols[2].metric(
                "P90 Coverage",
                f"{quant.get('q90_coverage', 0):.2%}",
            )

            if "mean_interval_width" in quant:
                st.metric(
                    "Mean Interval Width (P10–P90)",
                    f"{quant['mean_interval_width']:.6f}",
                )

            # Average calibration error across quantiles
            cal_errs = [
                quant.get(f"q{int(q * 100)}_cal_error", 0)
                for q in (0.1, 0.5, 0.9)
            ]
            avg_cal = float(np.mean(cal_errs))
            st.metric("Avg Calibration Error", f"{avg_cal:.4f}")

        # All metrics
        with st.expander(t["eval_detailed"]):
            st.json({**traj, **(quant or {})})

    except Exception as exc:
        st.error(t["eval_error"].format(err=exc))
        st.code(traceback.format_exc())


# ======================================================================
# TAB 6 — COMPARE
# ======================================================================
def _tab_compare() -> None:
    t = _t()
    st.header(t["compare_header"])
    st.info(t["compare_info"])

    assignments = ModelAssignments()
    registry = ModelRegistry()
    all_assignments = assignments.get_all()

    if not all_assignments:
        st.warning(t["compare_no_models"])
        return

    # Controls
    c1, c2 = st.columns(2)
    investment = c1.number_input(
        t["compare_investment"], min_value=100.0, max_value=1_000_000.0,
        value=1000.0, step=100.0, key="compare_investment",
    )
    horizon = c2.slider(
        t["compare_horizon"], 1, 60, 5, key="compare_horizon",
    )

    # Show which assets have assignments
    assigned_tickers = list(all_assignments.keys())
    st.markdown(
        f"**Assets with primary models:** "
        f"{', '.join(assigned_tickers)} ({len(assigned_tickers)})"
    )

    if not st.button(t["compare_btn"], key="btn_compare"):
        # Show previous result if available
        prev_result = st.session_state.get("compare_result")
        if prev_result:
            _render_comparison(prev_result)
        return

    try:
        with st.spinner(t["compare_spinner"]):
            forecasts: Dict[str, Any] = {}
            dfs: Dict[str, pd.DataFrame] = {}
            diagnostics_verdicts: Dict[str, str] = {}
            max_vols: Dict[str, float] = {}

            for ticker, model_id in all_assignments.items():
                # Load data
                loader = AssetDataLoader(ticker=ticker)
                df = loader.load_data()
                daily_ret = loader.daily_returns()

                eng = FeatureEngineering()
                df = eng.add_technical_indicators(df)
                feat_df = eng.select_features(df)
                feature_names = feat_df.columns.tolist()

                # Load model from registry
                saved = registry.list_models()
                meta = next(
                    (m for m in saved if m["model_id"] == model_id), None,
                )
                if meta is None:
                    continue

                arch = meta.get("architecture", "TCN")
                cfg = meta.get("config", {})
                model_cls = _ARCH_MAP.get(arch, TCNForecaster)

                model, scaler, _m = registry.load_model(
                    model_id,
                    model_cls,
                    input_size=len(feature_names),
                    hidden_size=cfg.get("hidden_size", 64),
                    num_layers=cfg.get("num_layers", 2),
                    forecast_steps=cfg.get("forecast_steps", 20),
                    quantiles=tuple(cfg.get("quantiles", [0.1, 0.5, 0.9])),
                )

                quantiles = tuple(cfg.get("quantiles", [0.1, 0.5, 0.9]))
                trainer = ModelTrainer(model, quantiles=quantiles, device="cpu")
                trainer.scaler = scaler

                seq_length = cfg.get("seq_length", 20)
                predictor = TrajectoryPredictor(trainer)
                forecast = predictor.predict_trajectory(
                    df, feature_names, seq_length, ticker,
                )

                forecasts[ticker] = forecast
                dfs[ticker] = df

                # Get diagnostics verdict from training summary
                ts = meta.get("training_summary", {})
                diagnostics_verdicts[ticker] = ts.get("diagnostics_verdict", "")

                # Asset-specific volatility
                if ticker in ASSET_CATALOG:
                    max_vols[ticker] = ASSET_CATALOG[ticker].max_volatility
                else:
                    max_vols[ticker] = 0.02

            # Run comparison
            comparator = PortfolioComparator(
                horizon_days=horizon,
            )
            result = comparator.compare(
                forecasts=forecasts,
                dfs=dfs,
                investment=investment,
                diagnostics_verdicts=diagnostics_verdicts,
                max_volatilities=max_vols,
            )

            st.session_state["compare_result"] = result

        _render_comparison(result)

    except Exception as exc:
        st.error(t["compare_error"].format(err=exc))
        st.code(traceback.format_exc())


def _render_comparison(result) -> None:
    """Render comparison results."""
    t = _t()

    if not result.outcomes:
        st.warning(t["compare_no_models"])
        return

    # Best asset banner
    best = result.outcomes[0]
    action_label = {
        "BUY": t["reco_buy"],
        "HOLD": t["reco_hold"],
        "AVOID": t["reco_avoid"],
    }
    st.markdown(
        f"### {t['compare_best_asset']}: **{best.ticker}** — "
        f"{action_label.get(best.recommendation.action, best.recommendation.action)} "
        f"({best.recommendation.confidence:.0f}/100)"
    )

    # Leaderboard table
    st.subheader(t["compare_leaderboard"])
    rows = []
    for o in result.outcomes:
        rows.append({
            t["compare_rank"]: o.rank,
            t["compare_asset"]: o.ticker,
            t["compare_action"]: action_label.get(o.recommendation.action, o.recommendation.action),
            t["compare_confidence"]: f"{o.recommendation.confidence:.0f}",
            t["compare_pnl_p50"]: f"${o.pnl_p50:+,.2f}",
            t["compare_pnl_pct"]: f"{o.pnl_pct_p50:+.2f}%",
            t["compare_value_p10"]: f"${o.projected_value_p10:,.2f}",
            t["compare_value_p50"]: f"${o.projected_value_p50:,.2f}",
            t["compare_value_p90"]: f"${o.projected_value_p90:,.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Detailed per-asset expanders
    for o in result.outcomes:
        with st.expander(t["compare_outcome_header"].format(asset=o.ticker)):
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric(t["compare_current_price"], f"${o.current_price:,.2f}")
            cc2.metric(t["compare_shares"], f"{o.shares:.4f}")
            cc3.metric(t["compare_pnl_p50"], f"${o.pnl_p50:+,.2f}")

            # Risk metrics for this asset
            risk = o.recommendation.risk
            rr1, rr2, rr3 = st.columns(3)
            rr1.metric(t["risk_stop_loss"], f"{risk.stop_loss_pct:+.2f}%")
            rr2.metric(t["risk_take_profit"], f"{risk.take_profit_pct:+.2f}%")
            rr3.metric(t["risk_reward_ratio"], f"{risk.risk_reward_ratio:.2f}")

            # Rationale summary
            if o.recommendation.rationale:
                for r in o.recommendation.rationale[:3]:
                    st.markdown(f"- {r}")


# ======================================================================
# TAB 7 — TUTORIAL
# ======================================================================
def _tab_tutorial() -> None:
    t = _t()
    st.header(t["tut_header"])
    st.markdown(t["tut_disclaimer"])

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


# ── Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
