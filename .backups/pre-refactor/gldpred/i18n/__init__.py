"""
Internationalization for the Streamlit application (EN / ES).

Usage::

    from gldpred.i18n import STRINGS, LANGUAGES
    t = STRINGS["en"]
    st.header(t["data_header"])
"""

LANGUAGES = {"English": "en", "Español": "es"}

STRINGS: dict[str, dict[str, str]] = {
    # ==================================================================
    # ENGLISH
    # ==================================================================
    "en": {
        # -- Chrome -------------------------------------------------------
        "page_title": "Asset Price Prediction",
        "app_title": "📈 Multi-Asset Price Prediction with Deep Learning",
        "app_subtitle": (
            "Multi-step trajectory forecasting with quantile uncertainty "
            "bands and decision support"
        ),

        # -- Sidebar ------------------------------------------------------
        "sidebar_config": "Configuration",
        "sidebar_asset": "Asset / Ticker",
        "sidebar_data_settings": "Data Settings",
        "sidebar_start_date": "Start Date",
        "sidebar_date_range": "Date range: All available history → today (auto)",
        "sidebar_model_settings": "Model Settings",
        "sidebar_model_arch": "Architecture",
        "sidebar_forecast_steps": "Forecast Steps (K days)",
        "sidebar_training_settings": "Training Settings",
        "sidebar_seq_length": "Sequence Length",
        "sidebar_hidden_size": "Hidden Size",
        "sidebar_num_layers": "Number of Layers",
        "sidebar_epochs": "Epochs",
        "sidebar_batch_size": "Batch Size",
        "sidebar_learning_rate": "Learning Rate",
        "sidebar_active_model": "Active Model",
        "sidebar_select_model": "Select model",
        "sidebar_no_models": "No saved models for this asset",
        "sidebar_model_loaded": "✅ Model loaded: {label}",
        "sidebar_model_mismatch": "⚠️ Model asset ({model_asset}) ≠ selected asset ({asset})",
        "sidebar_about": "About",
        "sidebar_about_text": (
            "Multi-step quantile forecasting for GLD, SLV, BTC-USD & PALL "
            "using TCN / GRU / LSTM. Includes trajectory fan charts, model "
            "registry, and educational decision support. Nothing in this app "
            "constitutes financial advice."
        ),

        # -- Tabs ---------------------------------------------------------
        "tab_data": "📊 Data",
        "tab_train": "🔧 Train",
        "tab_models": "🗂️ Models",
        "tab_forecast": "📈 Forecast",
        "tab_recommendation": "🎯 Recommendation",
        "tab_evaluation": "📉 Evaluation",
        "tab_compare": "⚖️ Compare",
        "tab_tutorial": "📚 Tutorial",

        # -- Tab 1: Data --------------------------------------------------
        "data_header": "Data Loading & Exploration",
        "data_refresh_btn": "🔄 Refresh Data",
        "data_auto_loaded": "Data loaded automatically for {asset}.",
        "data_loading_spinner": "Downloading data…",
        "data_load_success": "Loaded {n} records for {asset} ({start} → {end})",
        "data_load_error": "Error loading data: {err}",
        "data_metric_records": "Records",
        "data_metric_price": "Latest Price",
        "data_metric_change": "Price Change",
        "data_metric_features": "Features",
        "data_price_history": "Price History",
        "data_preview": "Data Preview",
        "data_info": (
            "Historical OHLCV data is fetched via yfinance. Technical "
            "indicators (SMA, EMA, RSI, MACD, ATR, volatility, momentum, "
            "lag features) are computed automatically — over 30 features "
            "in total. SMA-200 and ATR% are included for the decision engine."
        ),

        # -- Tab 2: Training ----------------------------------------------
        "train_header": "Model Training",
        "train_warn_no_data": "⚠️ Load data first in the Data tab.",
        "train_mode": "Training Mode",
        "train_mode_new": "Train from scratch",
        "train_mode_finetune": "Load & fine-tune",
        "train_btn": "Train Model",
        "train_finetune_btn": "Fine-tune Model",
        "train_spinner": "Training…",
        "train_success": "Model saved → registry ID: {model_id}",
        "train_error": "Training error: {err}",
        "train_info": (
            "Builds multi-step targets (K future daily returns), creates "
            "input sequences, and trains with pinball (quantile) loss. "
            "The model outputs P10 / P50 / P90 return forecasts for each "
            "future day. Results are saved to the model registry."
        ),
        "train_finetune_epochs": "Additional Epochs",
        "train_select_model": "Select Model to Fine-tune",
        "train_label": "Custom Model Name (optional)",
        "train_label_help": "Give your model a memorable name (max 60 chars). If empty, auto-generated.",
        "train_label_saved_as": "Model saved as: {label}",
        
        # -- Registry Management ------------------------------------------
        "registry_header": "Model Registry",
        "registry_delete_header": "Delete Models",
        "registry_delete_single": "Delete Selected Model",
        "registry_delete_all": "Delete All Models",
        "registry_delete_all_asset": "Delete All {asset} Models",
        "registry_confirm_header": "⚠️ Confirm Deletion",
        "registry_confirm_single": "Type DELETE to confirm deletion of:",
        "registry_confirm_all": "Type DELETE ALL to confirm deletion of {count} models.",
        "registry_confirm_input": "Confirmation",
        "registry_delete_btn": "Confirm Delete",
        "registry_delete_success": "Deleted {count} model(s).",
        "registry_delete_error": "Deletion error: {err}",
        "registry_no_models": "No models in registry.",
        "registry_model_details": "Model Details",

        # -- Diagnostics --------------------------------------------------
        "diag_header": "Training Diagnostics",
        "diag_verdict": "Verdict",
        "diag_verdict_healthy": "✅ Healthy",
        "diag_verdict_overfitting": "⚠️ Overfitting",
        "diag_verdict_underfitting": "⚠️ Underfitting",
        "diag_verdict_noisy": "⚠️ Noisy / Unstable",
        "diag_explanation": "Explanation",
        "diag_suggestions": "Suggestions",
        "diag_best_epoch": "Best Epoch",
        "diag_gen_gap": "Gen. Gap",
        "diag_apply_btn": "✨ Apply Suggestions",
        "diag_applied_success": "Suggestions applied — sidebar settings updated. Retrain to see the effect.",
        "diag_loss_chart": "Loss Curve",

        # -- Fine-tune validation -----------------------------------------
        "train_feature_mismatch": (
            "⚠️ Feature dimension mismatch: saved model expects {expected} "
            "features but current data has {got}. Cannot fine-tune."
        ),

        # -- Tab 3: Forecast ----------------------------------------------
        "forecast_header": "Forecast Trajectory",
        "forecast_warn_no_model": "⚠️ No model loaded. Train a new model or select a saved model from the sidebar.",
        "forecast_fan_chart": "Price Forecast with Uncertainty Bands",
        "forecast_table": "Forecast Table (next K days)",
        "forecast_col_day": "Day",
        "forecast_col_date": "Date",
        "forecast_col_p10": "P10 (Pessimistic)",
        "forecast_col_p50": "P50 (Median)",
        "forecast_col_p90": "P90 (Optimistic)",
        "forecast_col_return": "Median Return",
        "forecast_error": "Forecast error: {err}",
        "forecast_info": (
            "The fan chart shows the median predicted price path (P50) "
            "with P10–P90 uncertainty bands. Wider bands mean higher "
            "uncertainty. The table lists predicted prices and returns "
            "for each future trading day."
        ),

        # -- Tab 4: Recommendation ----------------------------------------
        "reco_header": "Decision Support",
        "reco_warn_no_model": "⚠️ No model loaded. Generate a forecast first (Forecast tab), or select a model from the sidebar.",
        "reco_disclaimer": (
            "> **Disclaimer:** This recommendation is purely educational. "
            "It does NOT constitute financial advice. Past performance does "
            "not guarantee future results. Always consult a qualified "
            "financial advisor before making investment decisions."
        ),
        "reco_action": "Recommendation",
        "reco_confidence": "Confidence",
        "reco_rationale": "Rationale",
        "reco_warnings": "Warnings",
        "reco_buy": "🟢 BUY",
        "reco_hold": "🟡 HOLD",
        "reco_avoid": "🔴 AVOID",
        "reco_decision_window": "Decision Window (days)",
        "reco_error": "Recommendation error: {err}",
        "reco_info": (
            "The recommendation engine combines predicted trajectory "
            "returns, trend filters (SMA50/SMA200), volatility (ATR%), "
            "uncertainty width, and model health diagnostics into a "
            "single BUY / HOLD / AVOID signal with a confidence score."
        ),

        # -- Tab 5: Evaluation --------------------------------------------
        "eval_header": "Model Evaluation",
        "eval_warn_no_model": "⚠️ No model loaded. Train a model or select a saved model from the sidebar.",
        "eval_trajectory_metrics": "Trajectory Metrics (validation set)",
        "eval_quantile_metrics": "Quantile Calibration",
        "eval_detailed": "All Metrics",
        "eval_error": "Evaluation error: {err}",
        "eval_info": (
            "Trajectory metrics measure prediction accuracy on the held-out "
            "validation set. Directional accuracy = fraction of days where "
            "the model correctly predicts the sign of the return. Quantile "
            "calibration checks whether P10/P50/P90 bands contain the "
            "expected fraction of observations."
        ),

        # -- Registry UI --------------------------------------------------
        "registry_header": "Model Registry",
        "registry_no_models": "No saved models found for this asset/architecture.",
        "registry_model_info": "Model Information",
        "registry_created": "Created",
        "registry_architecture": "Architecture",
        "registry_asset": "Asset",
        "registry_epochs": "Epochs",
        "registry_verdict": "Diagnostics",
        "registry_deleted": "Model deleted.",

        # -- Axis labels ---------------------------------------------------
        "axis_date": "Date",
        "axis_price": "Price (USD)",
        "axis_returns": "Returns",
        "axis_day": "Day",

        # -- Risk metrics --------------------------------------------------
        "risk_header": "Risk Metrics",
        "risk_stop_loss": "Stop-Loss",
        "risk_take_profit": "Take-Profit",
        "risk_reward_ratio": "Risk/Reward Ratio",
        "risk_max_drawdown": "Max Drawdown",
        "risk_volatility_regime": "Volatility Regime",
        "risk_regime_low": "🟢 Low",
        "risk_regime_normal": "🟡 Normal",
        "risk_regime_high": "🔴 High",

        # -- Market regime -------------------------------------------------
        "regime_header": "Market Regime",
        "regime_trending_up": "📈 Trending Up",
        "regime_trending_down": "📉 Trending Down",
        "regime_ranging": "↔️ Ranging",
        "regime_high_volatility": "⚡ High Volatility",
        "regime_unknown": "❓ Unknown",

        # -- Asset assignment ----------------------------------------------
        "assign_header": "Primary Model Assignment",
        "assign_btn": "Set as Primary",
        "assign_unassign_btn": "Unassign",
        "assign_current": "Current primary model",
        "assign_none": "No primary model assigned",
        "assign_success": "Primary model for {asset} set to: {label}",
        "assign_removed": "Primary model for {asset} removed.",

        # -- Compare tab ---------------------------------------------------
        "compare_header": "Multi-Asset Comparison",
        "compare_info": (
            "Compare projected outcomes across multiple assets with a "
            "hypothetical investment. Each asset uses its primary model "
            "from the registry. Load data and assign models first."
        ),
        "compare_investment": "Investment Amount ($)",
        "compare_horizon": "Comparison Horizon (days)",
        "compare_btn": "Run Comparison",
        "compare_spinner": "Running forecasts for all assets…",
        "compare_no_models": "No primary models assigned. Go to Train tab and assign models first.",
        "compare_leaderboard": "Leaderboard",
        "compare_rank": "Rank",
        "compare_asset": "Asset",
        "compare_action": "Signal",
        "compare_confidence": "Confidence",
        "compare_pnl_p50": "Median PnL",
        "compare_pnl_pct": "Return %",
        "compare_value_p10": "Value (P10)",
        "compare_value_p50": "Value (P50)",
        "compare_value_p90": "Value (P90)",
        "compare_best_asset": "Best Opportunity",
        "compare_error": "Comparison error: {err}",
        "compare_outcome_header": "{asset} — Projected Outcome",
        "compare_shares": "Shares",
        "compare_current_price": "Current Price",

        # -- Recommendation history ----------------------------------------
        "reco_history_header": "Recommendation History",
        "reco_history_empty": "No recommendations recorded yet.",
        "reco_history_clear": "Clear History",

        # -- Action plan --------------------------------------------------
        "ap_header": "Action Plan",
        "ap_info": (
            "Generate a time-based action plan for your chosen horizon. "
            "Each day is classified as BUY / HOLD / SELL / AVOID using "
            "the quantile forecast, with entry-window detection, optimal "
            "exit selection, scenario analysis, and decision rationale."
        ),
        "ap_generate": "Generate Action Plan",
        "ap_signal_buy": "🟢 BUY",
        "ap_signal_hold": "🟡 HOLD",
        "ap_signal_sell": "🔴 SELL",
        "ap_signal_avoid": "⚫ AVOID",
        "ap_overall_signal": "Overall Signal",
        "ap_confidence": "Confidence",
        "ap_narrative": "Summary",
        "ap_rationale_header": "Decision Rationale",
        "ap_trend": "Trend Confirmation",
        "ap_volatility": "Volatility Regime",
        "ap_quantile_risk": "Risk Assessment",
        "ap_today": "Today's Assessment",
        "ap_scenarios_header": "Scenario Analysis",
        "ap_scenario_optimistic": "Optimistic (P90)",
        "ap_scenario_base": "Base (P50)",
        "ap_scenario_pessimistic": "Pessimistic (P10)",
        "ap_return": "Return",
        "ap_final_price": "Final Price",
        "ap_pnl": "P&L",
        "ap_investment_label": "on {amount}",
        "ap_entry_exit_header": "Entry & Exit Optimization",
        "ap_entry_window": "Best Entry Window",
        "ap_best_exit": "Best Exit Day",
        "ap_no_entry": "No favorable entry window found",
        "ap_timeline_header": "Daily Action Timeline",
        "ap_day_details": "Day {day} — {action}",
        "ap_chart_title": "Price Trajectory & Action Plan",
        "ap_plan_saved": "Plan saved to data/trade_plans/",
        "ap_no_forecast": "Generate a forecast first in the Forecast tab.",
        "ap_col_day": "Day",
        "ap_col_date": "Date",
        "ap_col_action": "Action",
        "ap_col_price": "Price (P50)",
        "ap_col_ret": "Return %",
        "ap_col_risk": "Risk Score",
        "ap_col_reason": "Rationale",
        # Action plan sidebar
        "sidebar_action_plan": "Action Plan Settings",
        "sidebar_tp_horizon": "Plan Horizon (days)",
        "sidebar_tp_take_profit": "Take-Profit (%)",
        "sidebar_tp_stop_loss": "Stop-Loss (%)",
        "sidebar_tp_min_return": "Min Expected Return (%)",
        "sidebar_tp_risk_aversion": "Risk Aversion (λ)",
        "sidebar_tp_investment": "Investment Amount ($)",

        # -- Models tab (new) ----------------------------------------------
        "models_header": "Model Management",
        "models_info": (
            "View, rename, delete, and assign primary models for each asset. "
            "The primary model is used by the Forecast, Recommendation, and "
            "Compare tabs."
        ),
        "models_asset_filter": "Filter by Asset",
        "models_all_assets": "All Assets",
        "models_no_models": "No models found. Train a model first in the Train tab.",
        "models_rename_label": "New label",
        "models_rename_btn": "Rename",
        "models_rename_success": "Model renamed to: {label}",
        "models_rename_error": "Rename error: {err}",
        "models_delete_btn": "🗑️ Delete",
        "models_delete_confirm": "Type DELETE to confirm:",
        "models_delete_success": "Model deleted.",
        "models_delete_error": "Delete error: {err}",
        "models_set_primary_btn": "⭐ Set as Primary",
        "models_unset_primary_btn": "Remove Primary",
        "models_primary_badge": "⭐ PRIMARY",
        "models_primary_set": "Primary model for {asset} set to: {label}",
        "models_primary_removed": "Primary model for {asset} removed.",
        "models_bulk_delete_header": "Bulk Delete",
        "models_bulk_delete_btn": "Delete All Shown Models",
        "models_bulk_confirm": "Type DELETE ALL to confirm deletion of {count} models:",
        "models_col_label": "Label",
        "models_col_asset": "Asset",
        "models_col_arch": "Architecture",
        "models_col_created": "Created",
        "models_col_primary": "Primary",
        "models_col_actions": "Actions",

        # -- Compare tab (updated) ----------------------------------------
        "compare_add_row": "+ Add Asset",
        "compare_remove_row": "✕",
        "compare_select_asset": "Asset",
        "compare_select_model": "Model",
        "compare_no_models_for_asset": "No models for {asset}. Train one first.",
        "compare_base_label": "Base",
        "compare_vs_label": "vs.",

        # -- Tutorial ------------------------------------------------------
        "tut_header": "📚 Tutorial — How This Application Works",
        "tut_disclaimer": (
            "> **Disclaimer:** This application is an educational tool for "
            "exploring deep learning applied to financial time series. "
            "Nothing here constitutes financial advice."
        ),
        "tut_s1_title": "1 — Overview",
        "tut_s1_body": """
This application downloads historical price data for a selected asset
(GLD, SLV, BTC-USD, or PALL), engineers technical features, and trains
a deep-learning model to **forecast a multi-step trajectory** of future
daily returns with **quantile uncertainty bands** (P10 / P50 / P90).

| Tab | Purpose |
|-----|---------|
| **📊 Data** | Download and explore asset data |
| **🔧 Train** | Train or fine-tune a forecasting model |
| **📈 Forecast** | View the predicted price trajectory fan chart |
| **🎯 Recommendation** | Educational BUY / HOLD / AVOID signal |
| **📉 Evaluation** | Trajectory accuracy & quantile calibration |

The default architecture is **TCN** (Temporal Convolutional Network).
GRU and LSTM are also available.
""",
        "tut_s2_title": "2 — Data: Multi-Asset Support",
        "tut_s2_body": """
### Supported assets

| Ticker | Asset | Type |
|--------|-------|------|
| **GLD** | SPDR Gold Shares | Gold ETF |
| **SLV** | iShares Silver Trust | Silver ETF |
| **BTC-USD** | Bitcoin | Cryptocurrency |
| **PALL** | Aberdeen Physical Palladium | Palladium ETF |

Data is fetched via **yfinance**. Over 30 technical features are computed
including SMA (5/10/20/50/200), EMA, RSI-14, MACD, ATR-14, ATR%,
volatility, momentum, volume ratios, and lag features.

SMA-200 and ATR% are specifically used by the recommendation engine
for trend and volatility filters.
""",
        "tut_s3_title": "3 — Model Architectures",
        "tut_s3_body": """
All architectures output **(batch, K, Q)** — a multi-step quantile
forecast for K future days across Q quantile levels.

### TCN (Default)
Stacked causal 1-D convolutions with exponential dilation and residual
connections. Trains fastest due to full parallelism.

### GRU
Gated Recurrent Unit — simpler RNN variant with fewer parameters.

### LSTM
Long Short-Term Memory — better at retaining information across long
sequences but slower and more parameters.

| | TCN | GRU | LSTM |
|-|-----|-----|------|
| Speed | ⚡⚡ Fastest | ⚡ Fast | 🐢 Slower |
| Parameters | Medium | Low | High |
| Long sequences | ✅ | ⚠️ | ✅ |
""",
        "tut_s4_title": "4 — Multi-Step Forecasting & Quantiles",
        "tut_s4_body": """
### What is multi-step forecasting?

Instead of predicting a single value (e.g., "5-day return"), the model
outputs a **trajectory**: predicted daily returns for each of the next
K days (t+1, t+2, …, t+K).

### Quantile uncertainty

For each future day the model outputs three quantiles:

| Quantile | Meaning |
|----------|---------|
| **P10** | 10th percentile — pessimistic scenario |
| **P50** | Median — central forecast |
| **P90** | 90th percentile — optimistic scenario |

The **fan chart** visualises these as bands around the median price path.
Wider bands = more uncertainty.

### Pinball loss

The model is trained with **pinball (quantile) loss**, which penalises
under-prediction and over-prediction asymmetrically for each quantile
level, producing well-calibrated uncertainty estimates.
""",
        "tut_s5_title": "5 — Configurable Parameters",
        "tut_s5_body": """
| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Forecast Steps (K) | 5–60 | 20 | Days into the future |
| Sequence Length | 10–60 | 20 | Lookback window |
| Hidden Size | 32–128 | 64 | Model capacity |
| Num Layers | 1–4 | 2 | Depth |
| Epochs | 10–200 | 50 | Training iterations |
| Batch Size | 16–128 | 32 | Gradient smoothness |
| Learning Rate | 0.0001–0.01 | 0.001 | Update step size |

**Tip:** Start with defaults. If validation loss rises while train loss
falls → reduce epochs/complexity. If both stay high → increase capacity.
""",
        "tut_s6_title": "6 — Training & Fine-Tuning",
        "tut_s6_body": """
### Training from scratch

1. Load data → compute features → create multi-step sequences
2. Temporal 80/20 split (no shuffling — older data trains, newer validates)
3. Train with pinball loss for the configured number of epochs
4. Save model, scaler, and metadata to the **model registry**

### Fine-tuning

Select an existing model from the registry and continue training with
additional epochs. The original scaler is preserved to avoid data
leakage. This is useful when new data becomes available.

### Diagnostics

After training, the loss curves are analysed automatically:
- **Healthy** — both curves decreasing with stable gap
- **Overfitting** — validation rises while training falls
- **Underfitting** — both curves high and flat
- **Noisy** — validation oscillates significantly
""",
        "tut_s7_title": "7 — Forecast Trajectory & Fan Chart",
        "tut_s7_body": """
The **Forecast** tab uses the most recent data to predict the next K
trading days.

### Fan chart

- The solid line is the **median (P50)** predicted price path.
- The shaded band covers **P10 to P90** (80% prediction interval).
- Starting point is the last known close price.

### Price reconstruction

Predicted daily returns are converted to implied prices:

P(t+1) = P(t) × (1 + r(t+1))

This is done for each quantile independently, producing three price
paths (pessimistic, median, optimistic).
""",
        "tut_s8_title": "8 — Decision Support / Recommendation",
        "tut_s8_body": """
> **This is NOT financial advice.**

The recommendation engine combines five signals:

| Signal | What it checks |
|--------|----------------|
| **Expected return** | Median cumulative return over the decision window |
| **Trend filter** | Price > SMA200 AND SMA50 > SMA200 |
| **Volatility filter** | ATR% below asset-specific threshold |
| **Uncertainty width** | P90−P10 band width (penalises wide bands) |
| **Model health gate** | Diagnostics verdict (overfitting/noisy → penalty) |

Output: **BUY / HOLD / AVOID** with a confidence score (0–100) and
a list of rationale strings and warnings.
""",
        "tut_s9_title": "9 — Model Registry",
        "tut_s9_body": """
Every trained model is automatically saved to a local registry with:

- Model weights (.pth)
- Fitted scaler
- Feature schema
- Training configuration
- Training summary (epochs, losses, diagnostics verdict)
- Evaluation metrics

You can load any saved model for fine-tuning or direct inference.
The registry is stored in `data/model_registry/` (git-ignored).
""",
        "tut_s10_title": "10 — Quick-Reference Cheat Sheet",
        "tut_s10_body": """
### Recommended starting config

| Parameter | Value |
|-----------|-------|
| Asset | GLD |
| Architecture | TCN |
| Forecast Steps | 20 |
| Sequence Length | 20 |
| Hidden Size | 64 |
| Layers | 2 |
| Epochs | 50 |
| Batch Size | 32 |
| Learning Rate | 0.001 |

### Common adjustments

| Problem | Try |
|---------|-----|
| Overfitting | ↓ Epochs, ↓ Hidden size, ↓ Layers |
| Underfitting | ↑ Hidden size, ↑ Layers, ↑ Epochs |
| Unstable loss | ↓ Learning rate, ↑ Batch size |
| Wide uncertainty | ↑ Data range, ↑ Epochs |
| Slow training | Use TCN, ↓ Hidden size |
""",
    },

    # ==================================================================
    # SPANISH
    # ==================================================================
    "es": {
        # -- Chrome -------------------------------------------------------
        "page_title": "Predicción de Precios",
        "app_title": "📈 Predicción Multi-Activo con Deep Learning",
        "app_subtitle": (
            "Pronóstico de trayectoria multi-paso con bandas de "
            "incertidumbre cuantílica y soporte de decisión"
        ),

        # -- Sidebar ------------------------------------------------------
        "sidebar_config": "Configuración",
        "sidebar_asset": "Activo / Ticker",
        "sidebar_data_settings": "Datos",
        "sidebar_date_range": "Rango de fechas: todo el historial disponible → hoy (auto)",
        "sidebar_start_date": "Fecha de inicio",
        "sidebar_end_date_auto": "Fecha de fin: hoy (auto)",
        "sidebar_model_settings": "Modelo",
        "sidebar_model_arch": "Arquitectura",
        "sidebar_forecast_steps": "Pasos de pronóstico (K días)",
        "sidebar_training_settings": "Entrenamiento",
        "sidebar_seq_length": "Longitud de secuencia",
        "sidebar_hidden_size": "Tamaño oculto",
        "sidebar_num_layers": "Número de capas",
        "sidebar_epochs": "Épocas",
        "sidebar_batch_size": "Tamaño de lote",
        "sidebar_learning_rate": "Tasa de aprendizaje",
        "sidebar_active_model": "Modelo Activo",
        "sidebar_select_model": "Seleccionar modelo",
        "sidebar_no_models": "Sin modelos guardados para este activo",
        "sidebar_model_loaded": "✅ Modelo cargado: {label}",
        "sidebar_model_mismatch": "⚠️ Activo del modelo ({model_asset}) ≠ activo seleccionado ({asset})",
        "sidebar_about": "Acerca de",
        "sidebar_about_text": (
            "Pronóstico cuantílico multi-paso para GLD, SLV, BTC-USD y PALL "
            "con TCN / GRU / LSTM. Incluye gráficos de abanico, registro de "
            "modelos y soporte de decisión educativo. Nada en esta app "
            "constituye asesoramiento financiero."
        ),

        # -- Tabs ---------------------------------------------------------
        "tab_data": "📊 Datos",
        "tab_train": "🔧 Entrenar",
        "tab_models": "🗂️ Modelos",
        "tab_forecast": "📈 Pronóstico",
        "tab_recommendation": "🎯 Recomendación",
        "tab_evaluation": "📉 Evaluación",
        "tab_compare": "⚖️ Comparar",
        "tab_tutorial": "📚 Tutorial",

        # -- Tab 1: Data --------------------------------------------------
        "data_header": "Carga y exploración de datos",
        "data_refresh_btn": "🔄 Actualizar Datos",
        "data_auto_loaded": "Datos cargados automáticamente para {asset}.",
        "data_loading_spinner": "Descargando datos…",
        "data_load_success": "Cargados {n} registros de {asset} ({start} → {end})",
        "data_load_error": "Error al cargar datos: {err}",
        "data_metric_records": "Registros",
        "data_metric_price": "Último precio",
        "data_metric_change": "Variación",
        "data_metric_features": "Características",
        "data_price_history": "Historia de precios",
        "data_preview": "Vista previa",
        "data_info": (
            "Los datos OHLCV se obtienen de yfinance. Se calculan "
            "automáticamente indicadores técnicos (SMA, EMA, RSI, MACD, "
            "ATR, volatilidad, impulso, rezagos) — más de 30 "
            "características. SMA-200 y ATR% se usan en el motor de decisión."
        ),

        # -- Tab 2: Training ----------------------------------------------
        "train_header": "Entrenamiento del modelo",
        "train_warn_no_data": "⚠️ Primero cargue los datos en la pestaña Datos.",
        "train_mode": "Modo de entrenamiento",
        "train_mode_new": "Entrenar desde cero",
        "train_mode_finetune": "Cargar y ajustar",
        "train_btn": "Entrenar modelo",
        "train_finetune_btn": "Ajustar modelo",
        "train_spinner": "Entrenando…",
        "train_success": "Modelo guardado → ID registro: {model_id}",
        "train_error": "Error de entrenamiento: {err}",
        "train_info": (
            "Construye objetivos multi-paso (K rendimientos diarios futuros), "
            "crea secuencias de entrada y entrena con pérdida pinball "
            "(cuantílica). El modelo produce pronósticos P10/P50/P90 para "
            "cada día futuro. Los resultados se guardan en el registro."
        ),
        "train_finetune_epochs": "Épocas adicionales",
        "train_select_model": "Seleccionar modelo a ajustar",        "train_label": "Nombre personalizado del modelo (opcional)",
        "train_label_help": "Dale a tu modelo un nombre memorable (máx. 60 caracteres). Si está vacío, se genera automáticamente.",
        "train_label_saved_as": "Modelo guardado como: {label}",
        
        # -- Registry Management ------------------------------------------
        "registry_header": "Registro de modelos",
        "registry_delete_header": "Eliminar modelos",
        "registry_delete_single": "Eliminar modelo seleccionado",
        "registry_delete_all": "Eliminar todos los modelos",
        "registry_delete_all_asset": "Eliminar todos los modelos de {asset}",
        "registry_confirm_header": "⚠️ Confirmar eliminación",
        "registry_confirm_single": "Escriba DELETE para confirmar la eliminación de:",
        "registry_confirm_all": "Escriba DELETE ALL para confirmar la eliminación de {count} modelos.",
        "registry_confirm_input": "Confirmación",
        "registry_delete_btn": "Confirmar eliminación",
        "registry_delete_success": "Eliminados {count} modelo(s).",
        "registry_delete_error": "Error de eliminación: {err}",
        "registry_no_models": "No hay modelos en el registro.",
        "registry_model_details": "Detalles del modelo",
        # -- Diagnostics --------------------------------------------------
        "diag_header": "Diagnósticos del entrenamiento",
        "diag_verdict": "Veredicto",
        "diag_verdict_healthy": "✅ Saludable",
        "diag_verdict_overfitting": "⚠️ Sobreajuste",
        "diag_verdict_underfitting": "⚠️ Infraajuste",
        "diag_verdict_noisy": "⚠️ Ruidoso / Inestable",
        "diag_explanation": "Explicación",
        "diag_suggestions": "Sugerencias",
        "diag_best_epoch": "Mejor época",
        "diag_gen_gap": "Brecha gen.",
        "diag_apply_btn": "✨ Aplicar sugerencias",
        "diag_applied_success": "Sugerencias aplicadas — configuración actualizada. Reentrene para ver el efecto.",
        "diag_loss_chart": "Curva de pérdida",

        # -- Fine-tune validation -----------------------------------------
        "train_feature_mismatch": (
            "⚠️ Discrepancia de dimensiones: el modelo guardado espera {expected} "
            "características pero los datos actuales tienen {got}. No se puede ajustar."
        ),

        # -- Tab 3: Forecast ----------------------------------------------
        "forecast_header": "Trayectoria de pronóstico",
        "forecast_warn_no_model": "⚠️ Sin modelo cargado. Entrene un modelo o seleccione uno guardado desde la barra lateral.",
        "forecast_fan_chart": "Pronóstico de precio con bandas de incertidumbre",
        "forecast_table": "Tabla de pronóstico (próximos K días)",
        "forecast_col_day": "Día",
        "forecast_col_date": "Fecha",
        "forecast_col_p10": "P10 (Pesimista)",
        "forecast_col_p50": "P50 (Mediana)",
        "forecast_col_p90": "P90 (Optimista)",
        "forecast_col_return": "Rendimiento mediano",
        "forecast_error": "Error de pronóstico: {err}",
        "forecast_info": (
            "El gráfico de abanico muestra la trayectoria de precio mediana "
            "(P50) con bandas de incertidumbre P10–P90. Bandas más anchas "
            "significan mayor incertidumbre. La tabla lista precios y "
            "rendimientos predichos para cada día futuro."
        ),

        # -- Tab 4: Recommendation ----------------------------------------
        "reco_header": "Soporte de decisión",
        "reco_warn_no_model": "⚠️ Sin modelo cargado. Genere un pronóstico primero (pestaña Pronóstico), o seleccione un modelo desde la barra lateral.",
        "reco_disclaimer": (
            "> **Aviso:** Esta recomendación es puramente educativa. "
            "NO constituye asesoramiento financiero. El rendimiento pasado "
            "no garantiza resultados futuros. Consulte siempre a un "
            "asesor financiero cualificado."
        ),
        "reco_action": "Recomendación",
        "reco_confidence": "Confianza",
        "reco_rationale": "Razonamiento",
        "reco_warnings": "Advertencias",
        "reco_buy": "🟢 COMPRAR",
        "reco_hold": "🟡 MANTENER",
        "reco_avoid": "🔴 EVITAR",
        "reco_decision_window": "Ventana de decisión (días)",
        "reco_error": "Error de recomendación: {err}",
        "reco_info": (
            "El motor combina rendimiento esperado, filtros de tendencia "
            "(SMA50/SMA200), volatilidad (ATR%), amplitud de incertidumbre "
            "y salud del modelo en una señal COMPRAR / MANTENER / EVITAR "
            "con puntuación de confianza."
        ),

        # -- Tab 5: Evaluation --------------------------------------------
        "eval_header": "Evaluación del modelo",
        "eval_warn_no_model": "⚠️ Sin modelo cargado. Entrene un modelo o seleccione uno guardado desde la barra lateral.",
        "eval_trajectory_metrics": "Métricas de trayectoria (validación)",
        "eval_quantile_metrics": "Calibración cuantílica",
        "eval_detailed": "Todas las métricas",
        "eval_error": "Error de evaluación: {err}",
        "eval_info": (
            "Las métricas de trayectoria miden la precisión en el conjunto "
            "de validación. Precisión direccional = fracción de días donde "
            "el modelo predice correctamente el signo del rendimiento. "
            "La calibración verifica si las bandas P10/P50/P90 contienen "
            "la fracción esperada de observaciones."
        ),

        # -- Registry UI --------------------------------------------------
        "registry_header": "Registro de modelos",
        "registry_no_models": "No se encontraron modelos para este activo/arquitectura.",
        "registry_model_info": "Información del modelo",
        "registry_created": "Creado",
        "registry_architecture": "Arquitectura",
        "registry_asset": "Activo",
        "registry_epochs": "Épocas",
        "registry_verdict": "Diagnóstico",
        "registry_deleted": "Modelo eliminado.",

        # -- Axis labels ---------------------------------------------------
        "axis_date": "Fecha",
        "axis_price": "Precio (USD)",
        "axis_returns": "Rendimientos",
        "axis_day": "Día",

        # -- Risk metrics --------------------------------------------------
        "risk_header": "Métricas de Riesgo",
        "risk_stop_loss": "Stop-Loss",
        "risk_take_profit": "Take-Profit",
        "risk_reward_ratio": "Ratio Riesgo/Beneficio",
        "risk_max_drawdown": "Drawdown Máximo",
        "risk_volatility_regime": "Régimen de Volatilidad",
        "risk_regime_low": "🟢 Baja",
        "risk_regime_normal": "🟡 Normal",
        "risk_regime_high": "🔴 Alta",

        # -- Market regime -------------------------------------------------
        "regime_header": "Régimen de Mercado",
        "regime_trending_up": "📈 Tendencia Alcista",
        "regime_trending_down": "📉 Tendencia Bajista",
        "regime_ranging": "↔️ Lateral",
        "regime_high_volatility": "⚡ Alta Volatilidad",
        "regime_unknown": "❓ Desconocido",

        # -- Asset assignment ----------------------------------------------
        "assign_header": "Asignación de Modelo Primario",
        "assign_btn": "Establecer como primario",
        "assign_unassign_btn": "Desasignar",
        "assign_current": "Modelo primario actual",
        "assign_none": "Sin modelo primario asignado",
        "assign_success": "Modelo primario de {asset} establecido: {label}",
        "assign_removed": "Modelo primario de {asset} eliminado.",

        # -- Compare tab ---------------------------------------------------
        "compare_header": "Comparación Multi-Activo",
        "compare_info": (
            "Compare los resultados proyectados en varios activos con una "
            "inversión hipotética. Cada activo usa su modelo primario del "
            "registro. Cargue datos y asigne modelos primero."
        ),
        "compare_investment": "Monto de Inversión ($)",
        "compare_horizon": "Horizonte de Comparación (días)",
        "compare_btn": "Ejecutar Comparación",
        "compare_spinner": "Ejecutando pronósticos para todos los activos…",
        "compare_no_models": "No hay modelos primarios asignados. Vaya a Entrenar y asigne modelos primero.",
        "compare_leaderboard": "Clasificación",
        "compare_rank": "Posición",
        "compare_asset": "Activo",
        "compare_action": "Señal",
        "compare_confidence": "Confianza",
        "compare_pnl_p50": "PnL Mediana",
        "compare_pnl_pct": "Retorno %",
        "compare_value_p10": "Valor (P10)",
        "compare_value_p50": "Valor (P50)",
        "compare_value_p90": "Valor (P90)",
        "compare_best_asset": "Mejor Oportunidad",
        "compare_error": "Error de comparación: {err}",
        "compare_outcome_header": "{asset} — Resultado Proyectado",
        "compare_shares": "Acciones",
        "compare_current_price": "Precio Actual",

        # -- Recommendation history ----------------------------------------
        "reco_history_header": "Historial de Recomendaciones",
        "reco_history_empty": "Sin recomendaciones registradas aún.",
        "reco_history_clear": "Limpiar Historial",

        # -- Action plan --------------------------------------------------
        "ap_header": "Plan de Acción",
        "ap_info": (
            "Genera un plan de acción temporal para tu horizonte elegido. "
            "Cada día se clasifica como COMPRAR / MANTENER / VENDER / EVITAR "
            "usando el pronóstico cuantílico, con detección de ventana de "
            "entrada, selección óptima de salida, análisis de escenarios "
            "y razonamiento de la decisión."
        ),
        "ap_generate": "Generar Plan de Acción",
        "ap_signal_buy": "🟢 COMPRAR",
        "ap_signal_hold": "🟡 MANTENER",
        "ap_signal_sell": "🔴 VENDER",
        "ap_signal_avoid": "⚫ EVITAR",
        "ap_overall_signal": "Señal General",
        "ap_confidence": "Confianza",
        "ap_narrative": "Resumen",
        "ap_rationale_header": "Razonamiento de la Decisión",
        "ap_trend": "Confirmación de Tendencia",
        "ap_volatility": "Régimen de Volatilidad",
        "ap_quantile_risk": "Evaluación de Riesgo",
        "ap_today": "Evaluación de Hoy",
        "ap_scenarios_header": "Análisis de Escenarios",
        "ap_scenario_optimistic": "Optimista (P90)",
        "ap_scenario_base": "Base (P50)",
        "ap_scenario_pessimistic": "Pesimista (P10)",
        "ap_return": "Retorno",
        "ap_final_price": "Precio Final",
        "ap_pnl": "G&P",
        "ap_investment_label": "sobre {amount}",
        "ap_entry_exit_header": "Optimización de Entrada y Salida",
        "ap_entry_window": "Mejor Ventana de Entrada",
        "ap_best_exit": "Mejor Día de Salida",
        "ap_no_entry": "No se encontró ventana de entrada favorable",
        "ap_timeline_header": "Línea de Tiempo de Acciones Diarias",
        "ap_day_details": "Día {day} — {action}",
        "ap_chart_title": "Trayectoria de Precio y Plan de Acción",
        "ap_plan_saved": "Plan guardado en data/trade_plans/",
        "ap_no_forecast": "Genera un pronóstico primero en la pestaña Pronóstico.",
        "ap_col_day": "Día",
        "ap_col_date": "Fecha",
        "ap_col_action": "Acción",
        "ap_col_price": "Precio (P50)",
        "ap_col_ret": "Retorno %",
        "ap_col_risk": "Puntuación Riesgo",
        "ap_col_reason": "Razonamiento",
        # Action plan sidebar
        "sidebar_action_plan": "Config. Plan de Acción",
        "sidebar_tp_horizon": "Horizonte del Plan (días)",
        "sidebar_tp_take_profit": "Take-Profit (%)",
        "sidebar_tp_stop_loss": "Stop-Loss (%)",
        "sidebar_tp_min_return": "Retorno Mín. Esperado (%)",
        "sidebar_tp_risk_aversion": "Aversión al Riesgo (λ)",
        "sidebar_tp_investment": "Monto de Inversión ($)",

        # -- Models tab (new) ----------------------------------------------
        "models_header": "Gestión de Modelos",
        "models_info": (
            "Vea, renombre, elimine y asigne modelos primarios para cada activo. "
            "El modelo primario es usado por las pestañas Pronóstico, Recomendación "
            "y Comparar."
        ),
        "models_asset_filter": "Filtrar por Activo",
        "models_all_assets": "Todos los Activos",
        "models_no_models": "No se encontraron modelos. Entrene un modelo primero en la pestaña Entrenar.",
        "models_rename_label": "Nueva etiqueta",
        "models_rename_btn": "Renombrar",
        "models_rename_success": "Modelo renombrado a: {label}",
        "models_rename_error": "Error al renombrar: {err}",
        "models_delete_btn": "🗑️ Eliminar",
        "models_delete_confirm": "Escriba DELETE para confirmar:",
        "models_delete_success": "Modelo eliminado.",
        "models_delete_error": "Error al eliminar: {err}",
        "models_set_primary_btn": "⭐ Establecer como Primario",
        "models_unset_primary_btn": "Quitar Primario",
        "models_primary_badge": "⭐ PRIMARIO",
        "models_primary_set": "Modelo primario de {asset} establecido: {label}",
        "models_primary_removed": "Modelo primario de {asset} eliminado.",
        "models_bulk_delete_header": "Eliminación Masiva",
        "models_bulk_delete_btn": "Eliminar Todos los Modelos Mostrados",
        "models_bulk_confirm": "Escriba DELETE ALL para confirmar la eliminación de {count} modelos:",
        "models_col_label": "Etiqueta",
        "models_col_asset": "Activo",
        "models_col_arch": "Arquitectura",
        "models_col_created": "Creado",
        "models_col_primary": "Primario",
        "models_col_actions": "Acciones",

        # -- Compare tab (updated) ----------------------------------------
        "compare_add_row": "+ Agregar Activo",
        "compare_remove_row": "✕",
        "compare_select_asset": "Activo",
        "compare_select_model": "Modelo",
        "compare_no_models_for_asset": "Sin modelos para {asset}. Entrene uno primero.",
        "compare_base_label": "Base",
        "compare_vs_label": "vs.",

        # -- Tutorial ------------------------------------------------------
        "tut_header": "📚 Tutorial — Cómo funciona esta aplicación",
        "tut_disclaimer": (
            "> **Aviso legal:** Esta aplicación es una herramienta educativa "
            "para explorar el aprendizaje profundo aplicado a series "
            "temporales financieras. Nada aquí constituye asesoramiento "
            "financiero."
        ),
        "tut_s1_title": "1 — Visión general",
        "tut_s1_body": """
Esta aplicación descarga datos históricos del activo seleccionado
(GLD, SLV, BTC-USD o PALL), calcula características técnicas y entrena
un modelo de aprendizaje profundo para **pronosticar una trayectoria
multi-paso** de rendimientos diarios con **bandas de incertidumbre
cuantílica** (P10 / P50 / P90).

| Pestaña | Propósito |
|---------|-----------|
| **📊 Datos** | Descargar y explorar datos del activo |
| **🔧 Entrenar** | Entrenar o ajustar un modelo |
| **📈 Pronóstico** | Ver la trayectoria con gráfico de abanico |
| **🎯 Recomendación** | Señal educativa COMPRAR / MANTENER / EVITAR |
| **📉 Evaluación** | Precisión y calibración cuantílica |

La arquitectura por defecto es **TCN**. GRU y LSTM también están
disponibles.
""",
        "tut_s2_title": "2 — Datos: Soporte multi-activo",
        "tut_s2_body": """
### Activos soportados

| Ticker | Activo | Tipo |
|--------|--------|------|
| **GLD** | SPDR Gold Shares | ETF de oro |
| **SLV** | iShares Silver Trust | ETF de plata |
| **BTC-USD** | Bitcoin | Criptomoneda |
| **PALL** | Aberdeen Physical Palladium | ETF de paladio |

Los datos se obtienen de **yfinance**. Se calculan más de 30
características técnicas incluyendo SMA (5/10/20/50/200), EMA,
RSI-14, MACD, ATR-14, ATR%, volatilidad, impulso, ratios de
volumen y valores rezagados.
""",
        "tut_s3_title": "3 — Arquitecturas de modelo",
        "tut_s3_body": """
Todas las arquitecturas producen **(lote, K, Q)** — un pronóstico
cuantílico multi-paso para K días futuros y Q niveles cuantílicos.

### TCN (Por defecto)
Convoluciones causales 1-D apiladas con dilatación exponencial y
conexiones residuales. La más rápida por su paralelismo total.

### GRU
Unidad Recurrente con Puertas — variante RNN más simple.

### LSTM
Memoria a Largo-Corto Plazo — mejor retención en secuencias largas
pero más lenta y con más parámetros.

| | TCN | GRU | LSTM |
|-|-----|-----|------|
| Velocidad | ⚡⚡ | ⚡ | 🐢 |
| Parámetros | Medio | Bajo | Alto |
| Secuencias largas | ✅ | ⚠️ | ✅ |
""",
        "tut_s4_title": "4 — Pronóstico multi-paso y cuantiles",
        "tut_s4_body": """
### ¿Qué es el pronóstico multi-paso?

En lugar de predecir un solo valor, el modelo produce una
**trayectoria**: rendimientos diarios predichos para cada uno de los
próximos K días (t+1, t+2, …, t+K).

### Incertidumbre cuantílica

Para cada día futuro el modelo produce tres cuantiles:

| Cuantil | Significado |
|---------|-------------|
| **P10** | Percentil 10 — escenario pesimista |
| **P50** | Mediana — pronóstico central |
| **P90** | Percentil 90 — escenario optimista |

El **gráfico de abanico** visualiza estas bandas alrededor de la
trayectoria de precio mediana. Bandas más anchas = más incertidumbre.

### Pérdida pinball

Se entrena con **pérdida pinball (cuantílica)**, que penaliza la
sub-predicción y sobre-predicción asimétricamente para cada nivel
cuantílico, produciendo estimaciones de incertidumbre bien calibradas.
""",
        "tut_s5_title": "5 — Parámetros configurables",
        "tut_s5_body": """
| Parámetro | Rango | Defecto | Efecto |
|-----------|-------|---------|--------|
| Pasos de pronóstico (K) | 5–60 | 20 | Días hacia el futuro |
| Longitud de secuencia | 10–60 | 20 | Ventana de observación |
| Tamaño oculto | 32–128 | 64 | Capacidad del modelo |
| Capas | 1–4 | 2 | Profundidad |
| Épocas | 10–200 | 50 | Iteraciones de entrenamiento |
| Tamaño de lote | 16–128 | 32 | Suavidad del gradiente |
| Tasa de aprendizaje | 0.0001–0.01 | 0.001 | Tamaño del paso |

**Consejo:** Empiece con los valores por defecto. Si la pérdida de
validación sube mientras la de entrenamiento baja → reduzca
épocas/complejidad.
""",
        "tut_s6_title": "6 — Entrenamiento y ajuste fino",
        "tut_s6_body": """
### Entrenar desde cero

1. Cargar datos → calcular características → crear secuencias
2. División temporal 80/20
3. Entrenar con pérdida pinball
4. Guardar modelo en el **registro de modelos**

### Ajuste fino

Seleccione un modelo del registro y continúe el entrenamiento con
épocas adicionales. El escalador original se preserva.

### Diagnósticos

Las curvas de pérdida se analizan automáticamente:
- **Saludable** — ambas descienden establemente
- **Sobreajuste** — validación sube, entrenamiento baja
- **Infraajuste** — ambas altas y planas
- **Ruidoso** — validación oscila significativamente
""",
        "tut_s7_title": "7 — Trayectoria y gráfico de abanico",
        "tut_s7_body": """
La pestaña **Pronóstico** usa los datos más recientes para predecir
los próximos K días.

### Gráfico de abanico

- La línea sólida es la trayectoria de precio **mediana (P50)**.
- La banda cubre **P10 a P90** (intervalo de predicción del 80%).
- El punto de partida es el último precio de cierre conocido.

### Reconstrucción de precios

Los rendimientos diarios predichos se convierten a precios implícitos:
P(t+1) = P(t) × (1 + r(t+1))
""",
        "tut_s8_title": "8 — Soporte de decisión / Recomendación",
        "tut_s8_body": """
> **Esto NO es asesoramiento financiero.**

El motor de recomendación combina cinco señales:

| Señal | Qué verifica |
|-------|--------------|
| **Rendimiento esperado** | Rendimiento acumulado mediano |
| **Filtro de tendencia** | Precio > SMA200 Y SMA50 > SMA200 |
| **Filtro de volatilidad** | ATR% bajo umbral del activo |
| **Amplitud de incertidumbre** | Ancho de bandas P90−P10 |
| **Salud del modelo** | Veredicto de diagnósticos |

Resultado: **COMPRAR / MANTENER / EVITAR** con puntuación de
confianza (0–100) y lista de razones y advertencias.
""",
        "tut_s9_title": "9 — Registro de modelos",
        "tut_s9_body": """
Cada modelo entrenado se guarda automáticamente con:

- Pesos del modelo (.pth)
- Escalador ajustado
- Esquema de características
- Configuración de entrenamiento
- Resumen de entrenamiento
- Métricas de evaluación

Puede cargar cualquier modelo guardado para ajuste fino o inferencia
directa. El registro se almacena en `data/model_registry/`.
""",
        "tut_s10_title": "10 — Hoja de referencia rápida",
        "tut_s10_body": """
### Configuración inicial recomendada

| Parámetro | Valor |
|-----------|-------|
| Activo | GLD |
| Arquitectura | TCN |
| Pasos de pronóstico | 20 |
| Longitud de secuencia | 20 |
| Tamaño oculto | 64 |
| Capas | 2 |
| Épocas | 50 |
| Tamaño de lote | 32 |
| Tasa de aprendizaje | 0.001 |

### Ajustes comunes

| Problema | Pruebe |
|----------|--------|
| Sobreajuste | ↓ Épocas, ↓ Tamaño oculto, ↓ Capas |
| Infraajuste | ↑ Tamaño oculto, ↑ Capas, ↑ Épocas |
| Pérdida inestable | ↓ Tasa de aprendizaje, ↑ Lote |
| Bandas anchas | ↑ Rango de datos, ↑ Épocas |
| Entrenamiento lento | Usar TCN, ↓ Tamaño oculto |
""",
    },
}
