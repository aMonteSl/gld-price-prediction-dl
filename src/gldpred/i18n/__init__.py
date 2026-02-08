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
        "tab_forecast": "📈 Forecast",
        "tab_recommendation": "🎯 Recommendation",
        "tab_evaluation": "📉 Evaluation",
        "tab_tutorial": "📚 Tutorial",

        # -- Tab 1: Data --------------------------------------------------
        "data_header": "Data Loading & Exploration",
        "data_load_btn": "Load Data",
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
        "forecast_warn_no_model": "⚠️ Train a model first.",
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
        "reco_warn_no_model": "⚠️ Train a model first.",
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
        "eval_warn_no_model": "⚠️ Train a model first.",
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
        "tab_forecast": "📈 Pronóstico",
        "tab_recommendation": "🎯 Recomendación",
        "tab_evaluation": "📉 Evaluación",
        "tab_tutorial": "📚 Tutorial",

        # -- Tab 1: Data --------------------------------------------------
        "data_header": "Carga y exploración de datos",
        "data_load_btn": "Cargar datos",
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
        "forecast_warn_no_model": "⚠️ Primero entrene un modelo.",
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
        "reco_warn_no_model": "⚠️ Primero entrene un modelo.",
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
        "eval_warn_no_model": "⚠️ Primero entrene un modelo.",
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
