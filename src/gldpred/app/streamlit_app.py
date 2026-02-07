"""
Streamlit GUI for GLD price prediction application.

Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
import os

from gldpred.data import GLDDataLoader
from gldpred.features import FeatureEngineering
from gldpred.models import (
    GRURegressor, LSTMRegressor, GRUClassifier, LSTMClassifier,
    TCNRegressor, TCNClassifier,
    GRUMultiTask, LSTMMultiTask, TCNMultiTask,
)
from gldpred.training import ModelTrainer
from gldpred.evaluation import ModelEvaluator
from gldpred.diagnostics import DiagnosticsAnalyzer
from gldpred.app.i18n import STRINGS, LANGUAGES


# ---------------------------------------------------------------------------
# Model registry: (model_type, task) → class
# ---------------------------------------------------------------------------
_MODEL_MAP = {
    ("GRU", "regression"): GRURegressor,
    ("GRU", "classification"): GRUClassifier,
    ("GRU", "multitask"): GRUMultiTask,
    ("LSTM", "regression"): LSTMRegressor,
    ("LSTM", "classification"): LSTMClassifier,
    ("LSTM", "multitask"): LSTMMultiTask,
    ("TCN", "regression"): TCNRegressor,
    ("TCN", "classification"): TCNClassifier,
    ("TCN", "multitask"): TCNMultiTask,
}


# ---------------------------------------------------------------------------
# Helper: resolve current language strings
# ---------------------------------------------------------------------------
def _t() -> dict[str, str]:
    """Return the translation dict for the active language."""
    lang_code = st.session_state.get("lang", "en")
    return STRINGS.get(lang_code, STRINGS["en"])


# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GLD Price Prediction",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — language selector (placed first so `t` is available everywhere)
# ---------------------------------------------------------------------------
selected_lang_label = st.sidebar.selectbox(
    "Language / Idioma",
    list(LANGUAGES.keys()),
    index=0,
    key="_lang_selector",
)
st.session_state["lang"] = LANGUAGES[selected_lang_label]

t = _t()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.title(t["app_title"])
st.markdown(t["app_subtitle"])

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
st.sidebar.header(t["sidebar_config"])

# Data settings
st.sidebar.subheader(t["sidebar_data_settings"])
start_date = st.sidebar.date_input(
    t["sidebar_start_date"],
    value=datetime.now() - timedelta(days=365 * 5),
)
end_date = st.sidebar.date_input(t["sidebar_end_date"], value=datetime.now())

# Model settings
st.sidebar.subheader(t["sidebar_model_settings"])
model_type = st.sidebar.selectbox(t["sidebar_model_arch"], ["GRU", "LSTM", "TCN"])

task_options = [
    t["sidebar_task_regression"],
    t["sidebar_task_classification"],
    t["sidebar_task_multitask"],
]
task_type = st.sidebar.selectbox(t["sidebar_task_type"], task_options)

horizon = st.sidebar.selectbox(t["sidebar_horizon"], [1, 5, 20])

# Resolve task string
if task_type == t["sidebar_task_regression"]:
    task = "regression"
elif task_type == t["sidebar_task_classification"]:
    task = "classification"
else:
    task = "multitask"

# Buy threshold (used for classification & multitask labels)
buy_threshold = 0.0
if task in ("classification", "multitask"):
    buy_threshold = st.sidebar.number_input(
        t["sidebar_buy_threshold"],
        min_value=0.0,
        max_value=0.05,
        value=0.003,
        step=0.001,
        format="%.4f",
    )

# Multi-task loss weights
w_reg, w_cls = 1.0, 1.0
if task == "multitask":
    w_reg = st.sidebar.slider(t["sidebar_w_reg"], 0.1, 5.0, 1.0, 0.1)
    w_cls = st.sidebar.slider(t["sidebar_w_cls"], 0.1, 5.0, 1.0, 0.1)

# Training settings
st.sidebar.subheader(t["sidebar_training_settings"])
seq_length = st.sidebar.slider(t["sidebar_seq_length"], 10, 60, 20)
hidden_size = st.sidebar.slider(t["sidebar_hidden_size"], 32, 128, 64)
num_layers = st.sidebar.slider(t["sidebar_num_layers"], 1, 4, 2)
epochs = st.sidebar.slider(t["sidebar_epochs"], 10, 200, 50)
batch_size = st.sidebar.slider(t["sidebar_batch_size"], 16, 128, 32)
learning_rate = st.sidebar.select_slider(
    t["sidebar_learning_rate"],
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    value=0.001,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [t["tab_data"], t["tab_train"], t["tab_predictions"],
     t["tab_evaluation"], t["tab_tutorial"]]
)

# ==========================================================================
# Tab 1 — Data
# ==========================================================================
with tab1:
    st.header(t["data_header"])
    with st.expander("ℹ️", expanded=False):
        st.info(t["data_info"])

    if st.button(t["data_load_btn"]):
        with st.spinner(t["data_loading_spinner"]):
            try:
                loader = GLDDataLoader(start_date=start_date, end_date=end_date)
                data = loader.load_data()

                fe = FeatureEngineering()
                data_with_features = fe.add_technical_indicators(data)

                st.session_state.loader = loader
                st.session_state.data = data
                st.session_state.data_with_features = data_with_features
                st.session_state.data_loaded = True

                st.success(
                    t["data_load_success"].format(
                        n=len(data), start=start_date, end=end_date
                    )
                )
            except Exception as e:
                st.error(t["data_load_error"].format(err=str(e)))

    if st.session_state.data_loaded:
        data = st.session_state.data
        data_with_features = st.session_state.data_with_features

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t["data_metric_records"], len(data))
        with col2:
            st.metric(t["data_metric_price"], f"${data['Close'].iloc[-1]:.2f}")
        with col3:
            pct = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            st.metric(t["data_metric_change"], f"{pct:.2f}%")
        with col4:
            st.metric(t["data_metric_features"], len(data_with_features.columns))

        st.subheader(t["data_price_history"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='gold', width=2)
        ))
        fig.update_layout(
            xaxis_title=t["axis_date"],
            yaxis_title=t["axis_price"],
            hovermode="x unified",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(t["data_preview"])
        st.dataframe(data_with_features.tail(10), use_container_width=True)

# ==========================================================================
# Tab 2 — Training  
# ==========================================================================
with tab2:
    st.header(t["train_header"])
    with st.expander("ℹ️", expanded=False):
        st.info(t["train_info"])

    if not st.session_state.data_loaded:
        st.warning(t["train_warn_no_data"])
    else:
        if st.button(t["train_btn"]):
            with st.spinner(t["train_spinner"]):
                try:
                    data_with_features = st.session_state.data_with_features
                    loader = st.session_state.loader

                    fe = FeatureEngineering()
                    features = fe.select_features(data_with_features)
                    features = features.ffill().bfill()

                    # --- Build targets ---
                    if task == "regression":
                        targets = loader.compute_returns(horizon=horizon)
                        y = targets.values
                    elif task == "classification":
                        returns = loader.compute_returns(horizon=horizon)
                        y = (returns > buy_threshold).astype(int).values
                    else:  # multitask
                        returns = loader.compute_returns(horizon=horizon)
                        y_reg = returns.values
                        y_cls = (returns > buy_threshold).astype(int).values
                        y = np.column_stack([y_reg, y_cls])

                    # --- Create sequences ---
                    X, y = fe.create_sequences(features, y, seq_length)

                    # --- Get/create model ---
                    model = _MODEL_MAP[(model_type, task)](
                        input_size=X.shape[2],
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=0.2,
                    )

                    # --- Train ---
                    trainer = ModelTrainer(
                        model,
                        task=task,
                        w_reg=w_reg,
                        w_cls=w_cls,
                    )
                    train_loader, val_loader = trainer.prepare_data(
                        X, y, batch_size=batch_size
                    )

                    status_text = st.empty()
                    for epoch in range(epochs):
                        history = trainer.train_epoch(train_loader, val_loader, learning_rate)
                        status_text.text(f"Epoch {epoch+1}/{epochs}")

                    # --- Save model ---
                    model_dir = "models"
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(
                        model_dir, f"{model_type}_{task}_{horizon}d.pth"
                    )
                    trainer.save_model(model_path)

                    st.session_state.trainer = trainer
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.task = task
                    st.session_state.data = st.session_state.data
                    st.session_state.buy_threshold = buy_threshold
                    st.session_state.model_trained = True

                    # --- Diagnostics ---
                    diag = DiagnosticsAnalyzer.analyze(history)
                    verdict_key = f"diag_verdict_{diag.verdict}"
                    verdict_label = t.get(verdict_key, diag.verdict)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(t["diag_verdict"], verdict_label)
                    with c2:
                        st.metric(t["diag_best_epoch"], diag.best_epoch + 1)
                    with c3:
                        st.metric(t["diag_gen_gap"], f"{diag.generalization_gap:.6f}")

                    st.markdown(f"**{t['diag_explanation']}:** {diag.explanation}")

                    if diag.suggestions:
                        st.markdown(f"**{t['diag_suggestions']}:**")
                        for sug in diag.suggestions:
                            st.markdown(f"- {sug}")

                    st.success(t["train_success"].format(path=model_path))

                except Exception as e:
                    st.error(t["train_error"].format(err=str(e)))
                    import traceback
                    st.code(traceback.format_exc())

# ==========================================================================
# Tab 3 — Predictions
# ==========================================================================
with tab3:
    st.header(t["pred_header"])
    with st.expander("ℹ️", expanded=False):
        st.info(t["pred_info"])

    if not st.session_state.model_trained:
        st.warning(t["pred_warn_no_model"])
    else:
        try:
            trainer = st.session_state.trainer
            X = st.session_state.X
            y = st.session_state.y
            task = st.session_state.task
            data = st.session_state.data

            # Make predictions
            predictions = trainer.predict(X)

            if task == "multitask":
                reg_preds, cls_preds = predictions
                num_predictions = len(reg_preds)
            else:
                num_predictions = len(predictions)

            if task in ("regression", "multitask"):
                st.subheader(t["pred_returns"])
                pred_returns = reg_preds if task == "multitask" else predictions
                actual_returns = y[:, 0] if task == "multitask" else y

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index[-num_predictions:],
                    y=actual_returns, mode='lines',
                    name='Actual',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=data.index[-num_predictions:],
                    y=pred_returns, mode='lines', name='Predicted',
                    line=dict(color='red', width=2, dash='dash')
                ))
                fig.update_layout(
                    xaxis_title=t["axis_date"],
                    yaxis_title=t["axis_return"],
                    hovermode="x unified",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            if task in ("classification", "multitask"):
                st.subheader(t["pred_signals"])
                pred_signals = cls_preds if task == "multitask" else predictions
                actual_signals = y[:, 1] if task == "multitask" else y

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index[-num_predictions:],
                    y=actual_signals, mode='markers',
                    name='Actual',
                    marker=dict(size=8, color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=data.index[-num_predictions:],
                    y=pred_signals, mode='markers', name='Predicted',
                    marker=dict(size=6, color='red', symbol='x')
                ))
                fig.update_layout(
                    xaxis_title=t["axis_date"],
                    yaxis_title=t["axis_signal"],
                    hovermode="x unified",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader(t["pred_recent"])
            if task == "multitask":
                recent_df = pd.DataFrame({
                    "Date": data.index[-20:],
                    "Price": data["Close"].values[-20:],
                    "Predicted Return": reg_preds[-20:],
                    "Predicted Signal": cls_preds[-20:],
                })
            elif task == "regression":
                recent_df = pd.DataFrame({
                    "Date": data.index[-20:],
                    "Price": data["Close"].values[-20:],
                    "Prediction": predictions[-20:],
                })
            else:
                recent_df = pd.DataFrame({
                    "Date": data.index[-20:],
                    "Price": data["Close"].values[-20:],
                    "Prediction": predictions[-20:],
                })
            st.dataframe(recent_df, use_container_width=True)

        except Exception as e:
            st.error(t["pred_error"].format(err=str(e)))

# ==========================================================================
# Tab 4 — Evaluation
# ==========================================================================
with tab4:
    st.header(t["eval_header"])
    with st.expander("ℹ️", expanded=False):
        st.info(t["eval_info"])

    if not st.session_state.model_trained:
        st.warning(t["eval_warn_no_model"])
    else:
        try:
            trainer = st.session_state.trainer
            X = st.session_state.X
            y = st.session_state.y
            task = st.session_state.task
            buy_threshold = st.session_state.buy_threshold

            predictions = trainer.predict(X)

            if task == "regression":
                metrics = ModelEvaluator.evaluate_regression(y, predictions)
            elif task == "classification":
                metrics = ModelEvaluator.evaluate_classification(y, predictions, buy_threshold)
            else:
                metrics = ModelEvaluator.evaluate_multitask(y, predictions, buy_threshold)

            if task in ("regression", "multitask"):
                st.markdown(f"#### {t['eval_regression_metrics']}")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("RMSE", f"{metrics.get('reg_rmse', 0):.6f}")
                with c2:
                    st.metric("MAE", f"{metrics.get('reg_mae', 0):.6f}")
                with c3:
                    st.metric("R²", f"{metrics.get('reg_r2', 0):.6f}")
                with c4:
                    st.metric("MSE", f"{metrics.get('reg_mse', 0):.6f}")

            if task in ("classification", "multitask"):
                st.markdown(f"#### {t['eval_classification_metrics']}")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Accuracy", f"{metrics.get('cls_accuracy', 0):.4f}")
                with c2:
                    st.metric("Precision", f"{metrics.get('cls_precision', 0):.4f}")
                with c3:
                    st.metric("Recall", f"{metrics.get('cls_recall', 0):.4f}")
                with c4:
                    st.metric("F1 Score", f"{metrics.get('cls_f1', 0):.4f}")

                if "cls_confusion_matrix" in metrics:
                    st.subheader(t["eval_confusion_matrix"])
                    cm = np.array(metrics["cls_confusion_matrix"])
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                    ax.figure.colorbar(im, ax=ax)
                    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                           xticklabels=[t["eval_cm_no_buy"], t["eval_cm_buy"]],
                           yticklabels=[t["eval_cm_no_buy"], t["eval_cm_buy"]],
                           title=t["eval_cm_title"], ylabel=t["eval_cm_ylabel"],
                           xlabel=t["eval_cm_xlabel"])
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, format(cm[i, j], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > cm.max() / 2.0 else "black")
                    st.pyplot(fig)

            st.subheader(t["eval_detailed"])
            display_metrics = {k: v for k, v in metrics.items() 
                             if k != "confusion_matrix" and k != "cls_confusion_matrix"}
            metrics_df = pd.DataFrame([display_metrics])
            st.dataframe(metrics_df, use_container_width=True)

        except Exception as e:
            st.error(t["eval_error"].format(err=str(e)))

# ==========================================================================
# Tab 5 — Tutorial / User Guide
# ==========================================================================
with tab5:
    st.header(t["tut_header"])
    st.markdown(t["tut_disclaimer"])

    for i in range(1, 11):
        with st.expander(t[f"tut_s{i}_title"], expanded=(i == 1)):
            st.markdown(t[f"tut_s{i}_body"])

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"### {t['sidebar_about']}")
st.sidebar.info(t["sidebar_about_text"])
