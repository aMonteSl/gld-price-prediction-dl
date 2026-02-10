"""Evaluation tab rendering."""
from __future__ import annotations

import traceback
from typing import Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gldpred.app import state
from gldpred.app.controllers.model_loader import auto_select_model, get_active_model
from gldpred.app.glossary import info_term
from gldpred.app.controllers.analysis_controller import compute_feature_importance_gradient
from gldpred.services.training_service import split_validation


def render(t: Dict[str, str], lang: str) -> None:
    """Render the Evaluation tab."""
    st.header(t["eval_header"])
    st.info(t["eval_info"])

    traj = state.get(state.KEY_TRAJ_METRICS)
    quant = state.get(state.KEY_QUANT_METRICS)

    # Allow tab to work when a model is loaded even without stored metrics
    asset = state.get(state.KEY_ASSET, "GLD")
    model = get_active_model()
    if model is None:
        model = auto_select_model(asset)

    if traj is None and model is None:
        st.warning(t["eval_warn_no_model"])
        return

    if traj is None:
        st.info(
            "📊 " + (
                "Model loaded but no evaluation metrics available. "
                "Train the model or run a forecast first to see metrics."
                if st.session_state.get("lang", "en") == "en"
                else "Modelo cargado pero sin métricas de evaluación. "
                "Entrena el modelo o ejecuta un pronóstico primero."
            )
        )
        # Still show feature importance if model is available
        st.subheader("📊 " + ("Importancia de Features" if lang == "es" else "Feature Importance"))
        _show_feature_importance(t, lang, model)
        return

    try:
        st.subheader(t["eval_trajectory_metrics"])
        c1, c2, c3, c4 = st.columns(4)
        info_term("MSE", "mse", lang)
        c1.metric("MSE", f"{traj.get('mse', 0):.6f}")
        c2.metric("RMSE", f"{traj.get('rmse', 0):.6f}")
        c3.metric("MAE", f"{traj.get('mae', 0):.6f}")
        info_term("Directional Accuracy", "directional_accuracy", lang)
        c4.metric(
            "Dir. Accuracy",
            f"{traj.get('directional_accuracy', 0):.2%}",
        )

        mae_per_step = traj.get("mae_per_step", [])
        if mae_per_step:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, len(mae_per_step) + 1)),
                y=mae_per_step, name="MAE",
            ))
            fig.update_layout(
                xaxis_title=t["axis_day"], yaxis_title="MAE",
                template="plotly_dark", height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        if quant:
            st.subheader(t["eval_quantile_metrics"])
            info_term(
                t["eval_quantile_metrics"], "calibration", lang,
            )
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
                    "Mean Interval Width (P10-P90)",
                    f"{quant['mean_interval_width']:.6f}",
                )

            cal_errs = [
                quant.get(f"q{int(q * 100)}_cal_error", 0)
                for q in (0.1, 0.5, 0.9)
            ]
            avg_cal = float(np.mean(cal_errs))
            st.metric("Avg Calibration Error", f"{avg_cal:.4f}")

        # Feature Importance section (NEW)
        st.subheader("📊 " + ("Importancia de Features" if lang == "es" else "Feature Importance"))
        _show_feature_importance(t, lang, model)

        with st.expander(t["eval_detailed"]):
            st.json({**traj, **(quant or {})})

    except Exception as exc:
        st.error(t["eval_error"].format(err=exc))
        st.code(traceback.format_exc())

def _show_feature_importance(t: Dict[str, str], lang: str, model=None) -> None:
    """Show top feature importance using gradient-based method."""
    if model is None:
        model = get_active_model()
    feature_names = state.get(state.KEY_FEATURE_NAMES, [])
    
    if model is None or not feature_names:
        st.caption("⚠️ " + ("Entrena o carga un modelo primero" if lang == "es" else "Train or load a model first"))
        return
    
    # Get validation data from session state (if available)
    # Otherwise, recompute from raw data
    df = state.get(state.KEY_RAW_DF)
    if df is None:
        st.caption("⚠️ " + ("Datos no disponibles" if lang == "es" else "Data not available"))
        return
    
    with st.spinner("Calculando importancia..." if lang == "es" else "Computing importance..."):
        try:
            from gldpred.features import FeatureEngineering
            
            # Rebuild sequences to get validation data
            eng = FeatureEngineering()
            features_df = eng.select_features(df)
            daily_returns = state.get(state.KEY_DAILY_RETURNS)
            seq_length = st.session_state.get("seq_length", 20)
            forecast_steps = st.session_state.get("forecast_steps", 20)
            
            X, y = eng.create_sequences(
                features_df.values,
                daily_returns.values,
                seq_length=seq_length,
                forecast_steps=forecast_steps,
            )
            
            # Get validation split
            X_val, y_val = split_validation(X, y, test_size=0.2)
            
            # Compute importance
            importances = compute_feature_importance_gradient(
                model, X_val, feature_names, top_k=10
            )
            
            if not importances:
                st.caption("⚠️ " + ("No se pudo calcular importancia" if lang == "es" else "Could not compute importance"))
                return
            
            # Create horizontal bar chart
            features_list = [item["feature"] for item in importances]
            importance_values = [item["importance"] for item in importances]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=features_list,
                x=importance_values,
                orientation='h',
                marker=dict(color='#3498db'),
                text=[f"{v:.6f}" for v in importance_values],
                textposition='outside',
            ))
            
            fig.update_layout(
                xaxis_title="Importancia (gradiente promedio)" if lang == "es" else "Importance (mean gradient)",
                yaxis_title="",
                height=400,
                margin=dict(l=10, r=10, t=20, b=20),
                yaxis=dict(autorange="reversed"),  # Top feature at top
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(
                "ℹ️ " + (
                    "Basado en gradientes: features con mayor impacto en la salida del modelo" 
                    if lang == "es" else 
                    "Gradient-based: features with highest impact on model output"
                )
            )
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())