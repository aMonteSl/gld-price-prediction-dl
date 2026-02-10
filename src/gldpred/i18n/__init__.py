"""
Internationalization for the Streamlit application (EN / ES).

Usage::

    from gldpred.i18n import STRINGS, LANGUAGES
    t = STRINGS["en"]
    st.header(t["data_header"])
"""

LANGUAGES = {"Español": "es", "English": "en"}
DEFAULT_LANGUAGE = "es"

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
        "sidebar_auto_config": "🤖 Auto-Config",
        "sidebar_auto_config_help": "Suggest hyperparameters based on asset and volatility",
        "sidebar_config_applied": "✅ Configuration applied",
        "sidebar_load_model": "📥 Load model",
        "sidebar_model_active": "✅ Active: {label}",

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
        "compare_scatter_title": "Risk vs. Return",
        "compare_scatter_x": "Max Risk (%)",
        "compare_scatter_y": "Expected Return (%)",

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
        "ap_click_generate": "Press **Generate Action Plan** to create an action plan.",
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

        # -- Empty states & Dashboard ------------------------------------
        "empty_no_data": (
            "📊 No market data loaded yet. Go to the **Data** tab "
            "and click **Load Data** to get started."
        ),
        "empty_no_model": (
            "🤖 No model available for this asset. You can:\n"
            "1. **Train** a new model in the Train tab\n"
            "2. **Load** an existing model from the sidebar\n"
            "3. **Assign** a primary model in the Models tab"
        ),
        "empty_no_forecast": (
            "🔮 No forecast generated yet. Go to the **Forecast** tab "
            "to generate predictions, then come back here."
        ),

        # -- Dashboard tab ------------------------------------------------
        "tab_dashboard": "🏠 Dashboard",
        "dash_header": "Investment Decision Board",
        "dash_subtitle": "Quick overview: should you invest today?",
        "dash_investment_label": "Investment Amount ($)",
        "dash_horizon_label": "Horizon (days)",
        "dash_run_analysis": "🔍 Analyze All Assets",
        "dash_leaderboard": "Asset Leaderboard",
        "dash_rank": "#",
        "dash_asset": "Asset",
        "dash_signal": "Signal",
        "dash_confidence": "Confidence",
        "dash_expected_return": "Expected Return",
        "dash_max_risk": "Max Risk",
        "dash_pnl": "P&L ($)",
        "dash_no_models": (
            "No models available. Train or load models first "
            "to see the decision board."
        ),
        "dash_processing": "Analyzing {asset}...",
        "dash_quick_view": "Quick View",
        "dash_entry_window": "Entry Window",
        "dash_best_exit": "Best Exit",
        "dash_view_details": "View Details",
        "dash_analysing_all": "Analyzing all assets...",
        "dash_last_update": "Last updated",
        "dash_click_run": "Press **Analyze All Assets** to get started.",
        "dash_model_label": "Model",

        # -- Portfolio tab ------------------------------------------------
        "tab_portfolio": "💼 Portfolio",
        "portfolio_header": "Portfolio & Trade Log",
        "portfolio_subtitle": "Track your investment decisions and compare predictions vs outcomes.",
        "portfolio_total": "Total Trades",
        "portfolio_open": "Open",
        "portfolio_closed": "Closed",
        "portfolio_win_rate": "Win Rate",
        "portfolio_archive": "Archive Current Plan",
        "portfolio_notes": "Notes (optional)",
        "portfolio_save_trade": "💾 Save to Trade Log",
        "portfolio_saved": "✅ Trade saved to log!",
        "portfolio_no_plan": "Generate an action plan in the Recommendation tab first.",
        "portfolio_log_header": "Trade Log",
        "portfolio_empty": "No trades recorded yet. Generate a recommendation and archive it to start tracking.",
        "portfolio_col_date": "Date",
        "portfolio_col_asset": "Asset",
        "portfolio_col_signal": "Signal",
        "portfolio_col_conf": "Conf.",
        "portfolio_col_expected": "Expected",
        "portfolio_col_actual": "Actual",
        "portfolio_col_status": "Status",
        "portfolio_col_investment": "Investment",
        "portfolio_close_header": "Close a Trade",
        "portfolio_select_trade": "Select trade to close",
        "portfolio_actual_return": "Actual Return (%)",
        "portfolio_actual_price": "Exit Price ($)",
        "portfolio_close_btn": "Close Trade",
        "portfolio_closed_msg": "Trade closed!",

        # -- Health tab ---------------------------------------------------
        "tab_health": "🩺 Health",
        "health_header": "Model Health & Accountability",
        "health_subtitle": "Monitor model freshness, prediction accuracy, and recalibration needs.",
        "health_no_models": "No models found in the registry. Train a model first.",
        "health_total_models": "Total Models",
        "health_assigned": "Assigned (Primary)",
        "health_stale_alert": "Stale / Expired",
        "health_avg_win_rate": "Avg Win Rate",
        "health_asset": "Asset",
        "health_age": "Age",
        "health_days": "days",
        "health_freshness": "Freshness",
        "health_status_fresh": "Fresh",
        "health_status_aging": "Aging",
        "health_status_stale": "Stale",
        "health_status_expired": "Expired",
        "health_training": "Training",
        "health_epochs": "Epochs",
        "health_best_loss": "Best Val Loss",
        "health_accuracy_header": "Prediction Accuracy",
        "health_trades_total": "Trades",
        "health_trades_closed": "Closed",
        "health_win_rate_label": "Win Rate",
        "health_bias": "Pred. Bias",
        "health_avg_predicted": "Avg predicted return",
        "health_avg_actual": "Avg actual",
        "health_no_closed": "No closed trades for this model yet. Archive recommendations in the Portfolio tab and close them with actual outcomes to see accuracy here.",
        "health_recs_header": "Recommendations",

        # -- Backtest tab -------------------------------------------------
        "tab_backtest": "🔬 Backtest",
        "backtest_header": "Walk-Forward Backtest",
        "backtest_subtitle": "Simulate how the model would have performed on historical data.",
        "backtest_params": "Parameters",
        "backtest_num_points": "Backtest points",
        "backtest_investment": "Investment ($)",
        "backtest_model": "Model",
        "backtest_asset": "Asset",
        "backtest_run": "🚀 Run Backtest",
        "backtest_running": "Running walk-forward backtest...",
        "backtest_no_results": "No backtest results produced.",
        "backtest_click_run": "Configure parameters and click Run to start the backtest.",
        "backtest_not_enough_data": "Not enough data for backtesting.",
        "backtest_results_header": "Results",
        "backtest_points": "Points",
        "backtest_dir_accuracy": "Dir. Accuracy",
        "backtest_coverage": "P10–P90 Coverage",
        "backtest_mae": "MAE",
        "backtest_avg_pred": "Avg Predicted",
        "backtest_avg_actual": "Avg Actual",
        "backtest_bias": "Bias",
        "backtest_avg_pnl": "Avg P&L",
        "backtest_band": "P10–P90 Band",
        "backtest_predicted": "Predicted (P50)",
        "backtest_actual": "Actual",
        "backtest_chart_title": "Predicted vs Actual Returns",
        "backtest_chart_x": "Date",
        "backtest_chart_y": "Return (%)",
        "backtest_detail_table": "📋 Detail Table",
        "backtest_col_date": "Date",
        "backtest_col_entry": "Entry",
        "backtest_col_pred": "Pred P50",
        "backtest_col_actual": "Actual",
        "backtest_col_error": "Error",
        "backtest_col_band": "In Band",
        "backtest_col_pnl": "P&L",

        # -- Data Hub tab -------------------------------------------------
        "tab_datahub": "📦 Data Hub",
        "hub_header": "Data Hub — Application Data Center",
        "hub_subtitle": "Inspect, export, and manage all persisted data used by the app.",

        # Market Data
        "hub_market_header": "📊 Market Data",
        "hub_market_asset": "Asset",
        "hub_market_records": "Records",
        "hub_market_range": "Date Range",
        "hub_market_cache_size": "Cache Size",
        "hub_market_refresh": "🔄 Refresh Data",
        "hub_market_export_csv": "📥 Export CSV",
        "hub_market_no_data": "No market data loaded. Go to the Data tab to load asset data.",

        # Models
        "hub_models_header": "🧠 Models",
        "hub_models_empty": "No models in registry. Train a model first.",
        "hub_models_total": "Total Models",
        "hub_models_epochs": "Epochs",
        "hub_models_verdict": "Verdict",
        "hub_models_export_meta": "📥 Export JSON",
        "hub_models_set_primary": "⭐ Set Primary",
        "hub_models_primary_done": "Primary model for {asset} set to: {label}",
        "hub_models_is_primary": "⭐ Primary",
        "hub_models_delete": "🗑️ Delete",
        "hub_models_confirm_delete": "Type DELETE to confirm:",
        "hub_models_deleted": "Model deleted.",

        # Forecasts
        "hub_forecasts_header": "🔮 Forecasts",
        "hub_forecasts_empty": "No forecast cached. Generate one in the Forecast tab.",
        "hub_forecasts_asset": "Asset",
        "hub_forecasts_model": "Model",
        "hub_forecasts_export": "📥 Export Forecast CSV",
        "hub_forecasts_clear": "🗑️ Clear Forecast",
        "hub_forecasts_cleared": "Forecast cleared.",

        # Trade Log
        "hub_trades_header": "💼 Trade Log",
        "hub_trades_empty": "No trades recorded. Archive recommendations in the Portfolio tab.",
        "hub_trades_export": "📥 Export Trade Log CSV",

        # Performance
        "hub_performance_header": "📈 Model Performance History",
        "hub_performance_empty": "No closed trades yet. Close trades with actual outcomes to see performance stats.",
        "hub_perf_model": "Model",
        "hub_perf_trades": "Trades",
        "hub_perf_win_rate": "Win Rate",
        "hub_perf_mae": "MAE",
        "hub_perf_bias": "Bias",
        "hub_perf_avg_pred": "Avg Predicted",
        "hub_perf_avg_actual": "Avg Actual",
        "hub_perf_degradation": "⚠️ Model '{model}' shows degradation (win rate: {wr}%). Consider retraining.",

        # Global actions
        "hub_global_header": "⚙️ Global Actions",
        "hub_global_export_all": "📦 Export All Data (ZIP)",
        "hub_global_download_zip": "📥 Download ZIP",
        "hub_global_nothing_to_export": "No data to export.",
        "hub_global_reset": "🗑️ Reset All Application Data",
        "hub_global_reset_warning": "⚠️ This will permanently delete ALL models, trade logs, forecasts, and cached data. This action cannot be undone.",
        "hub_global_reset_confirm": "Type RESET to confirm:",
        "hub_global_reset_done": "All application data has been reset.",

        # -- Guided Onboarding --------------------------------------------
        "onb_progress": "Step",
        "onb_back": "⬅️ Back",
        "onb_next": "Next ➡️",
        "onb_skip": "⏭️ Skip tutorial",
        "onb_finish": "✅ Get started",
        "onb_restart": "📘 Restart Guided Tutorial",
        "onb_restart_done": "Tutorial restarted! Refresh to see it.",
        "wt_restart": "🚀 Restart Hands-On Walkthrough",

        "onb_step1_title": "Welcome to the App",
        "onb_step1_body": """
**What does this app do?**

Imagine you have **$1,000** and you're wondering: *"Should I invest in gold today?"*

This app uses **deep learning** (artificial intelligence) to analyze years of market
data and give you an informed answer:

- 📈 **Forecasts** the likely price trajectory for the next 20 trading days
- 🎯 **Recommends** whether to BUY, HOLD, or AVOID — with a confidence score
- 💰 **Estimates** how much you could gain or lose on your $1,000
- ⚠️ **Warns** you about risks: stop-loss, max drawdown, volatility

> 🚫 **Important**: This is NOT financial advice. It's an educational tool to explore
> how machine learning can analyze markets.
""",

        "onb_step2_title": "The Dashboard — Your Starting Point",
        "onb_step2_body": """
The **🏠 Dashboard** is the first thing you see. It answers one question:

> *"If I invest today, which asset looks best?"*

It shows:
- A **leaderboard** ranking all assets (Gold, Silver, Bitcoin, Palladium)
- The **top recommendation** with projected gains on your investment
- **Entry timing** (when to buy) and **exit timing** (when to sell)

**Example**: The dashboard might say:
- 🟢 **Gold (GLD) — BUY** with 75% confidence
- Expected return: +2.3% → on $1,000 that's about **+$23**
- Best entry: days 1-3, Best exit: day 15
""",

        "onb_step3_title": "Understanding the Forecast",
        "onb_step3_body": """
The **📈 Forecast** tab shows where the model thinks the price is headed.

Instead of a single prediction, the model gives you **three scenarios**:

| Scenario | What it means |
|----------|---------------|
| **P90** (optimistic) | "Things could go this well" |
| **P50** (median) | "The most likely outcome" |
| **P10** (pessimistic) | "Things could go this badly" |

The **fan chart** draws these as a band. A wider band = more uncertainty.

**Example with $1,000 in gold**:
- P90: $1,040 (+$40)
- P50: $1,015 (+$15)
- P10: $985 (−$15)
""",

        "onb_step4_title": "The Recommendation — Should I Invest?",
        "onb_step4_body": """
The **🎯 Recommendation** tab analyzes the forecast and gives you a clear signal:

- 🟢 **BUY** — The model sees a good opportunity
- 🟡 **HOLD** — Not convinced either way; wait
- 🔴 **AVOID** — Too much risk or downward trend

The recommendation considers **5 factors**:
1. Is the expected return positive enough?
2. Is the market trending up? (SMA50 > SMA200)
3. Is volatility manageable?
4. Is the uncertainty band narrow enough?
5. Is the model healthy and well-trained?

It also generates an **Action Plan**: specific days to buy, hold, or sell.
""",

        "onb_step5_title": "Models — The Heart of the System",
        "onb_step5_body": """
The app uses **neural networks** (deep learning models) to make predictions.

You can:
- **Train** a new model in the 🔧 Train tab
- **Compare** different model types: TCN, GRU, LSTM
- **Save** models to a registry and reuse them later
- **Assign** a primary model per asset

**What's a model?** Think of it as a student that has studied years of market data.
After training, it can make educated guesses about the future.

**Tip**: Start with TCN (the default). It trains the fastest and usually
performs well.
""",

        "onb_step6_title": "Portfolio & Trade Log",
        "onb_step6_body": """
The **💼 Portfolio** tab lets you track your investment decisions over time.

**How it works:**
1. Generate a recommendation in the 🎯 tab
2. Archive it to the trade log
3. Later, close the trade with the actual result
4. Compare what the model predicted vs what actually happened

This answers the critical question: *"Is the model actually right?"*

**Example**: You archived a BUY signal for gold at $185. Two weeks later,
gold is at $188. You close the trade → actual return +1.6% vs predicted +2.1%.
The model was slightly optimistic but directionally correct.
""",

        "onb_step7_title": "Data Hub — Full Transparency",
        "onb_step7_body": """
The **📦 Data Hub** gives you complete visibility into everything the app stores:

- **Market data**: what's loaded, date ranges, cache size
- **Models**: all trained models, their metrics, export/delete
- **Forecasts**: cached predictions, export to CSV
- **Trade log**: all investments, export to CSV
- **Performance**: how accurate each model has been

You can also:
- **Export everything** as a ZIP file
- **Reset all data** if you want a clean start

Nothing is hidden. You're always in control.
""",

        "onb_step8_title": "Important Reminders",
        "onb_step8_body": """
Before you start, keep these in mind:

⚠️ **This is NOT financial advice**
This app is an educational tool. Real investment decisions should always
involve a qualified financial advisor.

📊 **Past performance ≠ future results**
Even the best model can be wrong. Markets are inherently unpredictable.

🔄 **Models need retraining**
Market conditions change. A model trained 30 days ago may be less accurate
than a freshly trained one.

💡 **Start small**
Use the practice mode. Try with a small amount. Understand the signals before
committing real money.

---

🎉 **You're ready to begin!** Head to the Dashboard to see your first analysis.
""",

        # -- Post-onboarding choice screen --------------------------------
        "onb_choice_title": "Would you like a hands-on example?",
        "onb_choice_body": """
You've finished the introduction! Now you can choose:

🚀 **Guided walkthrough** — We'll walk you through the app step by step,
starting with loading data, viewing a forecast, and reading your first
recommendation. Perfect if you want to see the app in action.

🗺️ **Explore on your own** — Jump straight into the app and discover
things at your own pace. You can always restart the tutorial later from
the Tutorial tab.
""",
        "onb_choice_walkthrough": "🚀 Yes, guide me step by step",
        "onb_choice_explore": "🗺️ Explore on my own",

        # -- Guided walkthrough (hands-on) --------------------------------
        "wt_progress": "Hands-on tutorial — step {step} of {total}",
        "wt_done_next": "✅ Done — Next step",
        "wt_finish": "🏁 Finish tutorial",
        "wt_skip": "⏭️ Skip tutorial",
        "wt_go_to_tab": "👉 The tutorial is waiting for you on the **{tab}** tab",
        "wt_complete": "🎉 Tutorial completed!",
        "wt_complete_body": """
Congratulations! You've completed the guided walkthrough.

You now know how to:
- ✅ Load market data for any asset
- ✅ Train or load a prediction model
- ✅ Read fan-chart forecasts with uncertainty bands
- ✅ Interpret recommendations and action plans
- ✅ Use the Dashboard for a quick investment overview

Explore the remaining tabs (**Compare**, **Portfolio**, **Health**,
**Backtest**, **Data Hub**) at your own pace. Happy investing! 📈
""",

        "wt_step1_title": "Step 1: Load Market Data",
        "wt_step1_body": """
📁 You're on the **Data** tab.

This is where the app downloads historical prices for a financial asset.

**What to do:**
1. Look at the **sidebar on the left** — an asset is already selected (GLD by default)
2. The data loads automatically! You should see a price chart and a table of records
3. Notice the date range, number of records, and latest price

💡 **Tip:** You can change the asset in the sidebar (GLD, SLV, BTC-USD, PALL)
and the data refreshes automatically.

When you've seen the data, click **"Done — Next step"** below.
""",

        "wt_step2_title": "Step 2: Get a Prediction Model",
        "wt_step2_body": """
🏋️ You're on the **Train** tab.

To generate forecasts, you need a trained neural network model.

**Two options:**

🔵 **Option A (Quick):** If there are already saved models, go to the
sidebar and click **"Load model from registry"** — pick one and you're set!

🟢 **Option B (Train new):** Configure the parameters here and click
**"Train"**. Use the "Auto-configure" button in the sidebar for
recommended settings. Training takes 1-3 minutes.

💡 **Tip:** TCN (Temporal Convolutional Network) is the default
architecture — it's fast and accurate.

Once you have a model loaded or trained, click **"Done — Next step"**.
""",

        "wt_step3_title": "Step 3: View the Forecast",
        "wt_step3_body": """
📈 You're on the **Forecast** tab.

This shows the **fan chart** — a visual representation of the model's
price prediction over the next 20 trading days.

**What to look for:**
- 🟦 **Blue shaded area** — the uncertainty band (P10 to P90)
- 📏 **Middle line (P50)** — the median prediction (most likely path)
- 📊 **Lower band (P10)** — worst case scenario (10th percentile)
- 📈 **Upper band (P90)** — best case scenario (90th percentile)

💡 **Tip:** The wider the band, the more uncertain the model is.
A narrow band means higher confidence.

If you see the chart, click **"Done — Next step"**.
""",

        "wt_step4_title": "Step 4: Read the Recommendation",
        "wt_step4_body": """
🎯 You're on the **Recommendation** tab.

This is where the app answers the big question: *"Should I invest?"*

**What to do:**
1. Click **"Generate Recommendation"** (or the equivalent button)
2. Read the signal: 🟢 **BUY** / 🟡 **HOLD** / ⚫ **AVOID**
3. Check the **confidence level** (higher = more conviction)
4. Explore the **action plan** — it tells you which days to buy/hold/sell
5. Review the **three scenarios** with money impact (P10/P50/P90)

💡 **Tip:** The recommendation considers trend, volatility, downside risk,
and current timing. It's not just one signal — it's a full analysis.

When you've explored it, click **"Done — Next step"**.
""",

        "wt_step5_title": "Step 5: The Decision Dashboard",
        "wt_step5_body": """
📊 You're on the **Dashboard** tab.

This is the main landing page — designed to answer
*"Where should I invest today?"* in under 30 seconds.

**What to do:**
1. Set your **investment amount** and **horizon** at the top
2. Click **"Run Analysis"**  to analyse all assets
3. See the **leaderboard** — assets ranked by expected return
4. Check each asset's signal, confidence, and risk metrics

💡 **Tip:** The Dashboard requires at least one model assigned
per asset. You can assign models in the **Models** tab.

This is the last step! Click **"Finish tutorial"** when ready.
""",

        # -- Asset classification -----------------------------------------
        "risk_level_low": "Low Risk",
        "risk_level_medium": "Medium Risk",
        "risk_level_high": "High Risk",
        "horizon_short": "Short-term",
        "horizon_medium": "Medium-term",
        "horizon_long": "Long-term",
        "vol_profile_stable": "Stable",
        "vol_profile_moderate": "Moderate",
        "vol_profile_volatile": "Volatile",
        "role_benchmark": "Benchmark",
        "role_diversifier": "Diversifier",
        "role_tactical": "Tactical",
        "role_speculative": "Speculative",
        "category_equity_etf": "Equity ETF",
        "category_bond_etf": "Bond ETF",
        "category_commodity_etf": "Commodity ETF",
        "category_crypto": "Cryptocurrency",
        "category_cash_proxy": "Cash Proxy",
        "sidebar_risk_tier": "Risk Tier",
        "sidebar_all_tiers": "All Tiers",
        "asset_risk_badge": "Risk: {level}",
        "asset_horizon_badge": "Horizon: {horizon}",
        "asset_role_badge": "Role: {role}",
        "asset_benchmark_label": "📊 Benchmark",
        "compare_benchmark_delta": "{ticker} is {delta:.2f}pp {direction} benchmark",
        "compare_above": "above",
        "compare_below": "below",
        "compare_benchmark_ref": "Benchmark: {ticker} P50 = {pnl:+.2f}%",
        "scoring_asset_class": "Asset Risk Profile",
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
        "sidebar_auto_config": "🤖 Configuración Automática",
        "sidebar_auto_config_help": "Sugerir hiperparámetros basados en activo y volatilidad",
        "sidebar_config_applied": "✅ Configuración aplicada",
        "sidebar_load_model": "📥 Cargar modelo",
        "sidebar_model_active": "✅ Activo: {label}",

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
        "compare_scatter_title": "Riesgo vs. Retorno",
        "compare_scatter_x": "Riesgo Máx. (%)",
        "compare_scatter_y": "Retorno Esperado (%)",

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
        "ap_click_generate": "Pulsa **Generar Plan de Acción** para crear un plan.",
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

        # -- Empty states & Dashboard ------------------------------------
        "empty_no_data": (
            "📊 No hay datos de mercado cargados. Ve a la pestaña "
            "**Datos** y haz clic en **Cargar datos** para comenzar."
        ),
        "empty_no_model": (
            "🤖 No hay modelo disponible para este activo. Puedes:\n"
            "1. **Entrenar** un nuevo modelo en la pestaña Entrenar\n"
            "2. **Cargar** un modelo existente desde la barra lateral\n"
            "3. **Asignar** un modelo primario en la pestaña Modelos"
        ),
        "empty_no_forecast": (
            "🔮 Aún no se ha generado pronóstico. Ve a la pestaña "
            "**Pronóstico** para generar predicciones, luego regresa aquí."
        ),

        # -- Dashboard tab ------------------------------------------------
        "tab_dashboard": "🏠 Panel",
        "dash_header": "Panel de Decisión de Inversión",
        "dash_subtitle": "Vista rápida: ¿deberías invertir hoy?",
        "dash_investment_label": "Monto de Inversión ($)",
        "dash_horizon_label": "Horizonte (días)",
        "dash_run_analysis": "🔍 Analizar Todos los Activos",
        "dash_leaderboard": "Ranking de Activos",
        "dash_rank": "#",
        "dash_asset": "Activo",
        "dash_signal": "Señal",
        "dash_confidence": "Confianza",
        "dash_expected_return": "Retorno Esperado",
        "dash_max_risk": "Riesgo Máx.",
        "dash_pnl": "G/P ($)",
        "dash_no_models": (
            "No hay modelos disponibles. Entrena o carga modelos primero "
            "para ver el panel de decisión."
        ),
        "dash_processing": "Analizando {asset}...",
        "dash_quick_view": "Vista Rápida",
        "dash_entry_window": "Ventana de Entrada",
        "dash_best_exit": "Mejor Salida",
        "dash_view_details": "Ver Detalles",
        "dash_analysing_all": "Analizando todos los activos...",
        "dash_last_update": "Última actualización",
        "dash_click_run": "Pulsa **Analizar Todos los Activos** para comenzar.",
        "dash_model_label": "Modelo",

        # -- Portfolio tab ------------------------------------------------
        "tab_portfolio": "💼 Portafolio",
        "portfolio_header": "Portafolio y Registro de Operaciones",
        "portfolio_subtitle": "Rastrea tus decisiones de inversión y compara predicciones vs resultados.",
        "portfolio_total": "Total Operaciones",
        "portfolio_open": "Abiertas",
        "portfolio_closed": "Cerradas",
        "portfolio_win_rate": "Tasa de Éxito",
        "portfolio_archive": "Archivar Plan Actual",
        "portfolio_notes": "Notas (opcional)",
        "portfolio_save_trade": "💾 Guardar en Registro",
        "portfolio_saved": "✅ Operación guardada!",
        "portfolio_no_plan": "Genera un plan de acción en la pestaña Recomendación primero.",
        "portfolio_log_header": "Registro de Operaciones",
        "portfolio_empty": "No hay operaciones registradas. Genera una recomendación y archívala para empezar a rastrear.",
        "portfolio_col_date": "Fecha",
        "portfolio_col_asset": "Activo",
        "portfolio_col_signal": "Señal",
        "portfolio_col_conf": "Conf.",
        "portfolio_col_expected": "Esperado",
        "portfolio_col_actual": "Real",
        "portfolio_col_status": "Estado",
        "portfolio_col_investment": "Inversión",
        "portfolio_close_header": "Cerrar una Operación",
        "portfolio_select_trade": "Seleccionar operación a cerrar",
        "portfolio_actual_return": "Retorno Real (%)",
        "portfolio_actual_price": "Precio de Salida ($)",
        "portfolio_close_btn": "Cerrar Operación",
        "portfolio_closed_msg": "¡Operación cerrada!",

        # -- Health tab ---------------------------------------------------
        "tab_health": "🩺 Salud",
        "health_header": "Salud y Rendición de Cuentas del Modelo",
        "health_subtitle": "Monitorea la frescura del modelo, precisión de predicciones y necesidades de recalibración.",
        "health_no_models": "No se encontraron modelos en el registro. Entrena un modelo primero.",
        "health_total_models": "Total Modelos",
        "health_assigned": "Asignados (Primario)",
        "health_stale_alert": "Caducados / Expirados",
        "health_avg_win_rate": "Tasa Éxito Prom.",
        "health_asset": "Activo",
        "health_age": "Antigüedad",
        "health_days": "días",
        "health_freshness": "Frescura",
        "health_status_fresh": "Fresco",
        "health_status_aging": "Envejeciendo",
        "health_status_stale": "Caducado",
        "health_status_expired": "Expirado",
        "health_training": "Entrenamiento",
        "health_epochs": "Épocas",
        "health_best_loss": "Mejor Pérdida Val",
        "health_accuracy_header": "Precisión de Predicción",
        "health_trades_total": "Operaciones",
        "health_trades_closed": "Cerradas",
        "health_win_rate_label": "Tasa de Éxito",
        "health_bias": "Sesgo Pred.",
        "health_avg_predicted": "Retorno predicho promedio",
        "health_avg_actual": "Real promedio",
        "health_no_closed": "No hay operaciones cerradas para este modelo aún. Archiva recomendaciones en la pestaña Portafolio y ciérralas con resultados reales para ver la precisión aquí.",
        "health_recs_header": "Recomendaciones",

        # -- Backtest tab -------------------------------------------------
        "tab_backtest": "🔬 Backtest",
        "backtest_header": "Backtest Walk-Forward",
        "backtest_subtitle": "Simula cómo habría rendido el modelo con datos históricos.",
        "backtest_params": "Parámetros",
        "backtest_num_points": "Puntos de backtest",
        "backtest_investment": "Inversión ($)",
        "backtest_model": "Modelo",
        "backtest_asset": "Activo",
        "backtest_run": "🚀 Ejecutar Backtest",
        "backtest_running": "Ejecutando backtest walk-forward...",
        "backtest_no_results": "No se produjeron resultados.",
        "backtest_click_run": "Configura los parámetros y haz clic en Ejecutar para iniciar el backtest.",
        "backtest_not_enough_data": "No hay suficientes datos para backtest.",
        "backtest_results_header": "Resultados",
        "backtest_points": "Puntos",
        "backtest_dir_accuracy": "Precisión Dir.",
        "backtest_coverage": "Cobertura P10–P90",
        "backtest_mae": "EAM",
        "backtest_avg_pred": "Predicho Prom.",
        "backtest_avg_actual": "Real Prom.",
        "backtest_bias": "Sesgo",
        "backtest_avg_pnl": "P&L Prom.",
        "backtest_band": "Banda P10–P90",
        "backtest_predicted": "Predicho (P50)",
        "backtest_actual": "Real",
        "backtest_chart_title": "Retornos Predichos vs Reales",
        "backtest_chart_x": "Fecha",
        "backtest_chart_y": "Retorno (%)",
        "backtest_detail_table": "📋 Tabla de Detalle",
        "backtest_col_date": "Fecha",
        "backtest_col_entry": "Entrada",
        "backtest_col_pred": "Pred P50",
        "backtest_col_actual": "Real",
        "backtest_col_error": "Error",
        "backtest_col_band": "En Banda",
        "backtest_col_pnl": "P&L",

        # -- Data Hub tab -------------------------------------------------
        "tab_datahub": "📦 Centro de Datos",
        "hub_header": "Centro de Datos — Panel de Control",
        "hub_subtitle": "Inspecciona, exporta y gestiona todos los datos persistidos de la aplicación.",

        # Market Data
        "hub_market_header": "📊 Datos de Mercado",
        "hub_market_asset": "Activo",
        "hub_market_records": "Registros",
        "hub_market_range": "Rango de Fechas",
        "hub_market_cache_size": "Tamaño Caché",
        "hub_market_refresh": "🔄 Actualizar Datos",
        "hub_market_export_csv": "📥 Exportar CSV",
        "hub_market_no_data": "No hay datos de mercado cargados. Ve a la pestaña Datos para cargar un activo.",

        # Models
        "hub_models_header": "🧠 Modelos",
        "hub_models_empty": "No hay modelos en el registro. Entrena un modelo primero.",
        "hub_models_total": "Total Modelos",
        "hub_models_epochs": "Épocas",
        "hub_models_verdict": "Veredicto",
        "hub_models_export_meta": "📥 Exportar JSON",
        "hub_models_set_primary": "⭐ Primario",
        "hub_models_primary_done": "Modelo primario de {asset} establecido: {label}",
        "hub_models_is_primary": "⭐ Primario",
        "hub_models_delete": "🗑️ Eliminar",
        "hub_models_confirm_delete": "Escribe DELETE para confirmar:",
        "hub_models_deleted": "Modelo eliminado.",

        # Forecasts
        "hub_forecasts_header": "🔮 Pronósticos",
        "hub_forecasts_empty": "No hay pronósticos en caché. Genera uno en la pestaña Pronóstico.",
        "hub_forecasts_asset": "Activo",
        "hub_forecasts_model": "Modelo",
        "hub_forecasts_export": "📥 Exportar Pronóstico CSV",
        "hub_forecasts_clear": "🗑️ Limpiar Pronóstico",
        "hub_forecasts_cleared": "Pronóstico limpiado.",

        # Trade Log
        "hub_trades_header": "💼 Registro de Operaciones",
        "hub_trades_empty": "No hay operaciones registradas. Archiva recomendaciones en la pestaña Portafolio.",
        "hub_trades_export": "📥 Exportar Operaciones CSV",

        # Performance
        "hub_performance_header": "📈 Historial de Rendimiento del Modelo",
        "hub_performance_empty": "No hay operaciones cerradas. Cierra operaciones con resultados reales para ver estadísticas.",
        "hub_perf_model": "Modelo",
        "hub_perf_trades": "Operaciones",
        "hub_perf_win_rate": "Tasa de Éxito",
        "hub_perf_mae": "EAM",
        "hub_perf_bias": "Sesgo",
        "hub_perf_avg_pred": "Predicho Prom.",
        "hub_perf_avg_actual": "Real Prom.",
        "hub_perf_degradation": "⚠️ El modelo '{model}' muestra degradación (tasa de éxito: {wr}%). Considera reentrenar.",

        # Global actions
        "hub_global_header": "⚙️ Acciones Globales",
        "hub_global_export_all": "📦 Exportar Todos los Datos (ZIP)",
        "hub_global_download_zip": "📥 Descargar ZIP",
        "hub_global_nothing_to_export": "No hay datos para exportar.",
        "hub_global_reset": "🗑️ Resetear Todos los Datos",
        "hub_global_reset_warning": "⚠️ Esto eliminará permanentemente TODOS los modelos, registros de operaciones, pronósticos y datos en caché. Esta acción no se puede deshacer.",
        "hub_global_reset_confirm": "Escribe RESET para confirmar:",
        "hub_global_reset_done": "Todos los datos de la aplicación han sido reseteados.",

        # -- Guided Onboarding --------------------------------------------
        "onb_progress": "Paso",
        "onb_back": "⬅️ Atrás",
        "onb_next": "Siguiente ➡️",
        "onb_skip": "⏭️ Saltar tutorial",
        "onb_finish": "✅ Empezar a usar",
        "onb_restart": "📘 Reiniciar Tutorial Guiado",
        "onb_restart_done": "Tutorial reiniciado. Se mostrará al recargar la página.",
        "wt_restart": "🚀 Reiniciar Tutorial Práctico",

        "onb_step1_title": "Bienvenido a la Aplicación",
        "onb_step1_body": """
**¿Qué hace esta aplicación?**

Imagina que tienes **1.000 €** y te preguntas: *"¿Debería invertir en oro hoy?"*

Esta app utiliza **aprendizaje profundo** (inteligencia artificial) para analizar
años de datos de mercado y darte una respuesta informada:

- 📈 **Pronostica** la trayectoria probable del precio para los próximos 20 días
- 🎯 **Recomienda** si COMPRAR, MANTENER o EVITAR — con una puntuación de confianza
- 💰 **Estima** cuánto podrías ganar o perder con tus 1.000 €
- ⚠️ **Te avisa** sobre riesgos: stop-loss, caída máxima, volatilidad

> 🚫 **Importante**: Esto NO es asesoramiento financiero. Es una herramienta
> educativa para explorar cómo la IA puede analizar mercados.
""",

        "onb_step2_title": "El Panel — Tu Punto de Partida",
        "onb_step2_body": """
El **🏠 Panel** es lo primero que ves. Responde a una pregunta:

> *"Si invierto hoy, ¿qué activo se ve mejor?"*

Te muestra:
- Un **ranking** de todos los activos (Oro, Plata, Bitcoin, Paladio)
- La **recomendación principal** con las ganancias proyectadas
- **Cuándo entrar** (comprar) y **cuándo salir** (vender)

**Ejemplo**: El panel podría decir:
- 🟢 **Oro (GLD) — COMPRAR** con 75% de confianza
- Retorno esperado: +2,3% → sobre 1.000 € eso son unos **+23 €**
- Mejor entrada: días 1-3, Mejor salida: día 15
""",

        "onb_step3_title": "Entendiendo el Pronóstico",
        "onb_step3_body": """
La pestaña **📈 Pronóstico** muestra hacia dónde cree el modelo que va el precio.

En lugar de una sola predicción, el modelo te da **tres escenarios**:

| Escenario | Qué significa |
|-----------|---------------|
| **P90** (optimista) | "Las cosas podrían ir así de bien" |
| **P50** (mediana) | "El resultado más probable" |
| **P10** (pesimista) | "Las cosas podrían ir así de mal" |

El **gráfico de abanico** dibuja estos escenarios como una banda.
Una banda más ancha = más incertidumbre.

**Ejemplo con 1.000 € en oro**:
- P90: 1.040 € (+40 €)
- P50: 1.015 € (+15 €)
- P10: 985 € (−15 €)
""",

        "onb_step4_title": "La Recomendación — ¿Debería Invertir?",
        "onb_step4_body": """
La pestaña **🎯 Recomendación** analiza el pronóstico y te da una señal clara:

- 🟢 **COMPRAR** — El modelo ve una buena oportunidad
- 🟡 **MANTENER** — No está convencido; espera
- 🔴 **EVITAR** — Demasiado riesgo o tendencia bajista

La recomendación considera **5 factores**:
1. ¿Es el retorno esperado suficientemente positivo?
2. ¿El mercado está en tendencia alcista? (SMA50 > SMA200)
3. ¿La volatilidad es manejable?
4. ¿La banda de incertidumbre es estrecha?
5. ¿El modelo está bien entrenado?

También genera un **Plan de Acción**: días concretos para comprar, mantener o vender.
""",

        "onb_step5_title": "Los Modelos — El Corazón del Sistema",
        "onb_step5_body": """
La app usa **redes neuronales** (modelos de aprendizaje profundo) para hacer predicciones.

Puedes:
- **Entrenar** un nuevo modelo en la pestaña 🔧 Entrenar
- **Comparar** diferentes tipos: TCN, GRU, LSTM
- **Guardar** modelos en un registro y reutilizarlos
- **Asignar** un modelo primario por activo

**¿Qué es un modelo?** Piensa en él como un estudiante que ha estudiado años
de datos de mercado. Después de entrenar, puede hacer predicciones educadas
sobre el futuro.

**Consejo**: Empieza con TCN (el tipo por defecto). Es el más rápido de entrenar
y normalmente funciona bien.
""",

        "onb_step6_title": "Portafolio y Registro de Operaciones",
        "onb_step6_body": """
La pestaña **💼 Portafolio** te permite seguir tus decisiones de inversión.

**Cómo funciona:**
1. Genera una recomendación en la pestaña 🎯
2. Archívala en el registro de operaciones
3. Más tarde, cierra la operación con el resultado real
4. Compara lo que el modelo predijo vs lo que realmente pasó

Esto responde a la pregunta clave: *"¿Acierta realmente el modelo?"*

**Ejemplo**: Archivaste una señal COMPRAR para oro a 185 $. Dos semanas después,
el oro está a 188 $. Cierras la operación → retorno real +1,6% vs predicho +2,1%.
El modelo fue ligeramente optimista pero acertó la dirección.
""",

        "onb_step7_title": "Centro de Datos — Transparencia Total",
        "onb_step7_body": """
El **📦 Centro de Datos** te da visibilidad completa sobre todo lo que la app almacena:

- **Datos de mercado**: qué hay cargado, rangos de fechas, tamaño de caché
- **Modelos**: todos los modelos entrenados, sus métricas, exportar/eliminar
- **Pronósticos**: predicciones en caché, exportar a CSV
- **Registro de operaciones**: todas las inversiones, exportar a CSV
- **Rendimiento**: cuán preciso ha sido cada modelo

También puedes:
- **Exportar todo** como archivo ZIP
- **Resetear todos los datos** si quieres empezar de cero

Nada está oculto. Siempre tienes el control.
""",

        "onb_step8_title": "Recordatorios Importantes",
        "onb_step8_body": """
Antes de empezar, ten en cuenta:

⚠️ **Esto NO es asesoramiento financiero**
Esta app es una herramienta educativa. Las decisiones de inversión reales
deben involucrar siempre a un asesor financiero cualificado.

📊 **Rendimiento pasado ≠ resultados futuros**
Incluso el mejor modelo puede equivocarse. Los mercados son inherentemente
impredecibles.

🔄 **Los modelos necesitan reentrenamiento**
Las condiciones del mercado cambian. Un modelo entrenado hace 30 días
puede ser menos preciso que uno recién entrenado.

💡 **Empieza poco a poco**
Usa el modo práctico. Prueba con una cantidad pequeña. Entiende las señales
antes de comprometer dinero real.

---

🎉 **¡Estás listo para empezar!** Ve al Panel para ver tu primer análisis.
""",

        # -- Pantalla de elección post-onboarding -------------------------
        "onb_choice_title": "¿Quieres un ejemplo práctico?",
        "onb_choice_body": """
¡Has terminado la introducción! Ahora puedes elegir:

🚀 **Tutorial guiado** — Te guiaremos por la app paso a paso:
cargar datos, ver un pronóstico y leer tu primera recomendación.
Perfecto si quieres ver la app en acción.

🗺️ **Explorar por tu cuenta** — Salta directamente a la app y
descubre las cosas a tu ritmo. Siempre puedes reiniciar el tutorial
desde la pestaña Tutorial.
""",
        "onb_choice_walkthrough": "🚀 Sí, guíame paso a paso",
        "onb_choice_explore": "🗺️ Explorar por mi cuenta",

        # -- Tutorial guiado práctico -------------------------------------
        "wt_progress": "Tutorial práctico — paso {step} de {total}",
        "wt_done_next": "✅ Listo — Siguiente paso",
        "wt_finish": "🏁 Terminar tutorial",
        "wt_skip": "⏭️ Saltar tutorial",
        "wt_go_to_tab": "👉 El tutorial te espera en la pestaña **{tab}**",
        "wt_complete": "🎉 ¡Tutorial completado!",
        "wt_complete_body": """
¡Enhorabuena! Has completado el tutorial guiado.

Ahora sabes cómo:
- ✅ Cargar datos de mercado de cualquier activo
- ✅ Entrenar o cargar un modelo de predicción
- ✅ Leer pronósticos fan-chart con bandas de incertidumbre
- ✅ Interpretar recomendaciones y planes de acción
- ✅ Usar el Panel para una visión rápida de inversión

Explora las pestañas restantes (**Comparar**, **Portafolio**, **Salud**,
**Backtest**, **Centro de Datos**) a tu ritmo. ¡Feliz inversión! 📈
""",

        "wt_step1_title": "Paso 1: Cargar Datos de Mercado",
        "wt_step1_body": """
📁 Estás en la pestaña **Datos**.

Aquí es donde la app descarga los precios históricos de un activo financiero.

**Qué hacer:**
1. Mira la **barra lateral izquierda** — ya hay un activo seleccionado (GLD por defecto)
2. ¡Los datos se cargan automáticamente! Deberías ver un gráfico de precios y una tabla de registros
3. Observa el rango de fechas, número de registros y último precio

💡 **Consejo:** Puedes cambiar el activo en la barra lateral (GLD, SLV, BTC-USD, PALL)
y los datos se actualizan automáticamente.

Cuando hayas visto los datos, haz clic en **"Listo — Siguiente paso"** abajo.
""",

        "wt_step2_title": "Paso 2: Obtener un Modelo de Predicción",
        "wt_step2_body": """
🏋️ Estás en la pestaña **Entrenar**.

Para generar pronósticos, necesitas un modelo de red neuronal entrenado.

**Dos opciones:**

🔵 **Opción A (Rápida):** Si ya hay modelos guardados, ve a la barra
lateral y haz clic en **"Cargar modelo del registro"** — ¡elige uno y listo!

🟢 **Opción B (Entrenar nuevo):** Configura los parámetros aquí y haz clic
en **"Entrenar"**. Usa el botón "Auto-configurar" en la barra lateral para
obtener la configuración recomendada. El entrenamiento toma 1-3 minutos.

💡 **Consejo:** TCN (Red Convolucional Temporal) es la arquitectura por
defecto — es rápida y precisa.

Cuando tengas un modelo cargado o entrenado, haz clic en **"Listo — Siguiente paso"**.
""",

        "wt_step3_title": "Paso 3: Ver el Pronóstico",
        "wt_step3_body": """
📈 Estás en la pestaña **Pronóstico**.

Aquí se muestra el **fan chart** — una representación visual de la
predicción de precios del modelo para los próximos 20 días de trading.

**Qué buscar:**
- 🟦 **Área sombreada azul** — la banda de incertidumbre (P10 a P90)
- 📏 **Línea central (P50)** — la predicción mediana (camino más probable)
- 📊 **Banda inferior (P10)** — peor escenario (percentil 10)
- 📈 **Banda superior (P90)** — mejor escenario (percentil 90)

💡 **Consejo:** Cuanto más ancha sea la banda, más incierto está el modelo.
Una banda estrecha significa mayor confianza.

Si ves el gráfico, haz clic en **"Listo — Siguiente paso"**.
""",

        "wt_step4_title": "Paso 4: Leer la Recomendación",
        "wt_step4_body": """
🎯 Estás en la pestaña **Recomendación**.

Aquí es donde la app responde la gran pregunta: *"¿Debería invertir?"*

**Qué hacer:**
1. Haz clic en **"Generar Recomendación"** (o el botón equivalente)
2. Lee la señal: 🟢 **COMPRAR** / 🟡 **MANTENER** / ⚫ **EVITAR**
3. Revisa el **nivel de confianza** (mayor = más convicción)
4. Explora el **plan de acción** — te dice qué días comprar/mantener/vender
5. Revisa los **tres escenarios** con impacto en dinero (P10/P50/P90)

💡 **Consejo:** La recomendación considera tendencia, volatilidad, riesgo bajista
y timing actual. No es solo una señal — es un análisis completo.

Cuando lo hayas explorado, haz clic en **"Listo — Siguiente paso"**.
""",

        "wt_step5_title": "Paso 5: El Panel de Decisiones",
        "wt_step5_body": """
📊 Estás en la pestaña **Panel**.

Esta es la página principal — diseñada para responder
*"¿Dónde debería invertir hoy?"* en menos de 30 segundos.

**Qué hacer:**
1. Establece tu **monto de inversión** y **horizonte** arriba
2. Haz clic en **"Ejecutar Análisis"** para analizar todos los activos
3. Ve el **leaderboard** — activos ordenados por retorno esperado
4. Revisa la señal, confianza y métricas de riesgo de cada activo

💡 **Consejo:** El Panel requiere al menos un modelo asignado por activo.
Puedes asignar modelos en la pestaña **Modelos**.

¡Este es el último paso! Haz clic en **"Terminar tutorial"** cuando estés listo.
""",

        # -- Asset classification -----------------------------------------
        "risk_level_low": "Riesgo Bajo",
        "risk_level_medium": "Riesgo Medio",
        "risk_level_high": "Riesgo Alto",
        "horizon_short": "Corto plazo",
        "horizon_medium": "Medio plazo",
        "horizon_long": "Largo plazo",
        "vol_profile_stable": "Estable",
        "vol_profile_moderate": "Moderado",
        "vol_profile_volatile": "Volátil",
        "role_benchmark": "Referencia",
        "role_diversifier": "Diversificador",
        "role_tactical": "Táctico",
        "role_speculative": "Especulativo",
        "category_equity_etf": "ETF de Renta Variable",
        "category_bond_etf": "ETF de Bonos",
        "category_commodity_etf": "ETF de Materias Primas",
        "category_crypto": "Criptomoneda",
        "category_cash_proxy": "Cuasi-liquidez",
        "sidebar_risk_tier": "Nivel de Riesgo",
        "sidebar_all_tiers": "Todos los niveles",
        "asset_risk_badge": "Riesgo: {level}",
        "asset_horizon_badge": "Horizonte: {horizon}",
        "asset_role_badge": "Rol: {role}",
        "asset_benchmark_label": "📊 Referencia",
        "compare_benchmark_delta": "{ticker} está {delta:.2f}pp {direction} la referencia",
        "compare_above": "por encima de",
        "compare_below": "por debajo de",
        "compare_benchmark_ref": "Referencia: {ticker} P50 = {pnl:+.2f}%",
        "scoring_asset_class": "Perfil de Riesgo del Activo",
    },
}
