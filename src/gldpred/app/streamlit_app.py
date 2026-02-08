"""Streamlit GUI — Multi-Asset Price Prediction with Deep Learning.

Six tabs: Data · Train · Forecast · Recommendation · Evaluation · Tutorial.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from gldpred.config import (
    SUPPORTED_ASSETS,
    AppConfig,
    DecisionConfig,
    ModelConfig,
    TrainingConfig,
)
from gldpred.data import AssetDataLoader
from gldpred.decision import DecisionEngine, Recommendation
from gldpred.diagnostics import DiagnosticsAnalyzer
from gldpred.evaluation import ModelEvaluator
from gldpred.features import FeatureEngineering
from gldpred.i18n import LANGUAGES, STRINGS
from gldpred.inference import TrajectoryPredictor
from gldpred.models import GRUForecaster, LSTMForecaster, TCNForecaster
from gldpred.registry import ModelRegistry
from gldpred.training import ModelTrainer

# ── Architecture lookup ──────────────────────────────────────────────────
_ARCH_MAP = {
    "TCN": TCNForecaster,
    "GRU": GRUForecaster,
    "LSTM": LSTMForecaster,
}

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
        asset = st.selectbox(
            t["sidebar_asset"],
            list(SUPPORTED_ASSETS),
            index=0,
        )
        st.session_state["asset"] = asset

        # ── Data ─────────────────────────────────────────────────────
        st.subheader(t["sidebar_data_settings"])
        default_end = datetime.now()
        default_start = default_end - timedelta(days=365 * 5)
        start_date = st.date_input(t["sidebar_start_date"], value=default_start)
        end_date = st.date_input(t["sidebar_end_date"], value=default_end)
        st.session_state["start_date"] = str(start_date)
        st.session_state["end_date"] = str(end_date)

        # ── Model ────────────────────────────────────────────────────
        st.subheader(t["sidebar_model_settings"])
        arch = st.selectbox(t["sidebar_model_arch"], ["TCN", "GRU", "LSTM"], index=0)
        st.session_state["architecture"] = arch

        forecast_steps = st.slider(t["sidebar_forecast_steps"], 5, 60, 20)
        st.session_state["forecast_steps"] = forecast_steps

        # ── Training ─────────────────────────────────────────────────
        st.subheader(t["sidebar_training_settings"])
        seq_length = st.slider(t["sidebar_seq_length"], 10, 60, 20)
        hidden_size = st.select_slider(t["sidebar_hidden_size"], [32, 64, 128], value=64)
        num_layers = st.slider(t["sidebar_num_layers"], 1, 4, 2)
        epochs = st.slider(t["sidebar_epochs"], 10, 200, 50)
        batch_size = st.select_slider(t["sidebar_batch_size"], [16, 32, 64, 128], value=32)
        lr = st.select_slider(
            t["sidebar_learning_rate"],
            [0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
        )

        st.session_state.update({
            "seq_length": seq_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
        })

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
                loader = AssetDataLoader(
                    ticker=asset,
                    start_date=st.session_state.get("start_date"),
                    end_date=st.session_state.get("end_date"),
                )
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
        if "SMA_50" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["SMA_50"], mode="lines",
                name="SMA 50", line=dict(color="#1f77b4", width=1, dash="dash"),
            ))
        if "SMA_200" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["SMA_200"], mode="lines",
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

    df = st.session_state.get("raw_df")
    if df is None:
        st.warning(t["train_warn_no_data"])
        return

    # Training mode
    mode = st.radio(t["train_mode"], [t["train_mode_new"], t["train_mode_finetune"]], horizontal=True)
    is_finetune = mode == t["train_mode_finetune"]

    registry = ModelRegistry()
    selected_model_id: str | None = None

    if is_finetune:
        asset = st.session_state.get("asset", "GLD")
        arch = st.session_state.get("architecture", "TCN")
        saved = registry.list_models(asset=asset, architecture=arch)
        if not saved:
            st.warning(t["registry_no_models"])
            return
        labels = [f"{m['model_id']} ({m.get('created_at', '?')[:16]})" for m in saved]
        choice = st.selectbox(t["train_select_model"], labels)
        selected_model_id = saved[labels.index(choice)]["model_id"]
        epochs_label = t["train_finetune_epochs"]
    else:
        epochs_label = t["sidebar_epochs"]

    btn_label = t["train_finetune_btn"] if is_finetune else t["train_btn"]
    if not st.button(btn_label, key="btn_train"):
        return

    try:
        with st.spinner(t["train_spinner"]):
            asset = st.session_state.get("asset", "GLD")
            arch = st.session_state.get("architecture", "TCN")
            forecast_steps = st.session_state.get("forecast_steps", 20)
            seq_length = st.session_state.get("seq_length", 20)
            hidden_size = st.session_state.get("hidden_size", 64)
            num_layers = st.session_state.get("num_layers", 2)
            epochs = st.session_state.get("epochs", 50)
            batch_size = st.session_state.get("batch_size", 32)
            lr = st.session_state.get("learning_rate", 0.001)
            quantiles = (0.1, 0.5, 0.9)

            daily_returns = st.session_state["daily_returns"]
            feature_names = st.session_state["feature_names"]

            # Sequences
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
                trainer = ModelTrainer(model, quantiles=quantiles, device=device)
                train_loader, val_loader = trainer.prepare_data(
                    X, y, test_size=0.2, batch_size=batch_size, refit_scaler=False,
                )
                trainer.scaler = scaler
            else:
                model = model_cls(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    forecast_steps=forecast_steps,
                    quantiles=quantiles,
                )
                trainer = ModelTrainer(model, quantiles=quantiles, device=device)
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
            diag = DiagnosticsAnalyzer()
            diag_result = diag.analyze({
                "train_loss": train_losses,
                "val_loss": val_losses,
            })

            # Evaluation on val set
            X_val_tensor = torch.tensor(
                trainer.scaler.transform(X[int(len(X) * 0.8):].reshape(-1, X.shape[2])).reshape(-1, seq_length, X.shape[2]),
                dtype=torch.float32,
            )
            y_val = y[int(len(y) * 0.8):]
            pred_val = trainer.predict(X_val_tensor.numpy())

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
            )

            # Store in session
            st.session_state["trainer"] = trainer
            st.session_state["train_losses"] = train_losses
            st.session_state["val_losses"] = val_losses
            st.session_state["diag_result"] = diag_result
            st.session_state["traj_metrics"] = traj_metrics
            st.session_state["quant_metrics"] = quant_metrics
            st.session_state["last_model_id"] = model_id

            st.success(t["train_success"].format(model_id=model_id))

    except Exception as exc:
        st.error(t["train_error"].format(err=exc))
        st.code(traceback.format_exc())
        return

    # Show diagnostics
    _show_diagnostics()


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
    c2.metric(t["diag_best_epoch"], diag.best_epoch)
    c3.metric(t["diag_gen_gap"], f"{diag.generalization_gap:.4f}")

    st.markdown(f"**{t['diag_explanation']}:** {diag.explanation}")
    if diag.suggestions:
        st.markdown(f"**{t['diag_suggestions']}:**")
        for s in diag.suggestions:
            st.markdown(f"- {s}")

    # Loss curve
    if train_losses and val_losses:
        fig = go.Figure()
        epochs_x = list(range(1, len(train_losses) + 1))
        fig.add_trace(go.Scatter(x=epochs_x, y=train_losses, mode="lines", name="Train"))
        fig.add_trace(go.Scatter(x=epochs_x, y=val_losses, mode="lines", name="Validation"))
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_dark",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# TAB 3 — FORECAST
# ======================================================================
def _tab_forecast() -> None:
    t = _t()
    st.header(t["forecast_header"])
    st.info(t["forecast_info"])

    trainer = st.session_state.get("trainer")
    df = st.session_state.get("raw_df")
    if trainer is None or df is None:
        st.warning(t["forecast_warn_no_model"])
        return

    try:
        asset = st.session_state.get("asset", "GLD")
        feature_names = st.session_state.get("feature_names", [])
        seq_length = st.session_state.get("seq_length", 20)
        quantiles = tuple(trainer.quantiles.tolist())

        predictor = TrajectoryPredictor(trainer)
        forecast = predictor.predict_trajectory(df, feature_names, seq_length, asset)

        st.session_state["forecast"] = forecast

        # ── Fan chart ─────────────────────────────────────────────────
        st.subheader(t["forecast_fan_chart"])
        _plot_fan_chart(df, forecast, t)

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


def _plot_fan_chart(df: pd.DataFrame, forecast: Any, t: Dict[str, str]) -> None:
    """Plotly fan chart with historical tail + forecast bands."""
    quantiles = forecast.quantiles
    q_idx = {round(q, 2): i for i, q in enumerate(quantiles)}
    lo_i = q_idx.get(0.1, 0)
    med_i = q_idx.get(0.5, 1)
    hi_i = q_idx.get(0.9, 2)

    # Historical tail (last 60 days)
    hist_tail = df["Close"].iloc[-60:]

    # Forecast dates (include last known as starting anchor)
    fc_dates = [forecast.last_date] + list(forecast.dates)
    prices_med = forecast.price_paths[:, med_i]
    prices_lo = forecast.price_paths[:, lo_i]
    prices_hi = forecast.price_paths[:, hi_i]

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_tail.index, y=hist_tail.values,
        mode="lines", name="Historical",
        line=dict(color="#FFD700", width=2),
    ))

    # P10-P90 band
    fig.add_trace(go.Scatter(
        x=fc_dates, y=prices_hi.tolist(),
        mode="lines", line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=prices_lo.tolist(),
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(255,215,0,0.15)",
        name="P10–P90",
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=fc_dates, y=prices_med.tolist(),
        mode="lines+markers", name="P50 (Median)",
        line=dict(color="#00BFFF", width=2),
    ))

    fig.update_layout(
        xaxis_title=t["axis_date"],
        yaxis_title=t["axis_price"],
        template="plotly_dark",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


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

        # Display
        action_label = {
            "BUY": t["reco_buy"],
            "HOLD": t["reco_hold"],
            "AVOID": t["reco_avoid"],
        }

        col1, col2 = st.columns(2)
        col1.markdown(f"### {t['reco_action']}: {action_label.get(reco.action, reco.action)}")
        col2.metric(t["reco_confidence"], f"{reco.confidence:.0f} / 100")

        if reco.rationale:
            st.markdown(f"**{t['reco_rationale']}**")
            for r in reco.rationale:
                st.markdown(f"- {r}")

        if reco.warnings:
            st.markdown(f"**{t['reco_warnings']}**")
            for w in reco.warnings:
                st.warning(w)

    except Exception as exc:
        st.error(t["reco_error"].format(err=exc))
        st.code(traceback.format_exc())


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
            cov_cols[0].metric("P10 Coverage", f"{quant.get('q10_coverage', 0):.2%}")
            cov_cols[1].metric("P50 Coverage", f"{quant.get('q50_coverage', 0):.2%}")
            cov_cols[2].metric("P90 Coverage", f"{quant.get('q90_coverage', 0):.2%}")

            st.metric("Mean Interval Width (P10–P90)", f"{quant.get('mean_interval_width', 0):.6f}")
            st.metric("Calibration Error", f"{quant.get('calibration_error', 0):.4f}")

        # All metrics
        with st.expander(t["eval_detailed"]):
            st.json({**traj, **(quant or {})})

    except Exception as exc:
        st.error(t["eval_error"].format(err=exc))
        st.code(traceback.format_exc())


# ======================================================================
# TAB 6 — TUTORIAL
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
