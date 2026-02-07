"""
Internationalization (i18n) module for the Streamlit application.

All user-facing strings are stored here, keyed by language code.
To add a new language, add a new top-level key (e.g. ``"fr"``) and
provide translations for every entry.

Usage::

    from gldpred.app.i18n import STRINGS
    t = STRINGS["en"]          # or STRINGS["es"]
    st.header(t["data_header"])
"""

LANGUAGES = {"English": "en", "Español": "es"}

# ---------------------------------------------------------------------------
# Master translation dictionary
# ---------------------------------------------------------------------------
STRINGS: dict[str, dict[str, str]] = {
    # ======================================================================
    # ENGLISH
    # ======================================================================
    "en": {
        # -- Page chrome ----------------------------------------------------
        "page_title": "GLD Price Prediction",
        "app_title": "🏅 GLD Price Prediction with Deep Learning",
        "app_subtitle": "Forecast Gold ETF price movements using GRU / LSTM / TCN models",

        # -- Sidebar --------------------------------------------------------
        "sidebar_config": "Configuration",
        "sidebar_language": "Language / Idioma",
        "sidebar_data_settings": "Data Settings",
        "sidebar_start_date": "Start Date",
        "sidebar_end_date": "End Date",
        "sidebar_model_settings": "Model Settings",
        "sidebar_model_arch": "Model Architecture",
        "sidebar_task_type": "Task Type",
        "sidebar_task_regression": "Regression (Returns)",
        "sidebar_task_classification": "Classification (Buy/No-Buy)",
        "sidebar_task_multitask": "Multi-task (Reg + Cls)",
        "sidebar_horizon": "Prediction Horizon (days)",
        "sidebar_training_settings": "Training Settings",
        "sidebar_seq_length": "Sequence Length",
        "sidebar_hidden_size": "Hidden Size",
        "sidebar_num_layers": "Number of Layers",
        "sidebar_epochs": "Epochs",
        "sidebar_batch_size": "Batch Size",
        "sidebar_learning_rate": "Learning Rate",
        "sidebar_buy_threshold": "Buy Threshold",
        "sidebar_w_reg": "Regression Loss Weight",
        "sidebar_w_cls": "Classification Loss Weight",
        "sidebar_about": "About",
        "sidebar_about_text": (
            "This application uses deep learning (GRU / LSTM / TCN) to predict "
            "GLD price movements. It supports regression, classification, and "
            "multi-task learning at multiple time horizons (1, 5, 20 days) "
            "with automatic training diagnostics."
        ),

        # -- Tab names ------------------------------------------------------
        "tab_data": "📊 Data",
        "tab_train": "🔧 Train Model",
        "tab_predictions": "📈 Predictions",
        "tab_evaluation": "📉 Evaluation",
        "tab_tutorial": "📚 Tutorial",

        # -- Tab 1: Data ----------------------------------------------------
        "data_header": "Data Loading and Exploration",
        "data_load_btn": "Load Data",
        "data_loading_spinner": "Loading GLD data...",
        "data_load_success": "✅ Loaded {n} records from {start} to {end}",
        "data_load_error": "❌ Error loading data: {err}",
        "data_metric_records": "Records",
        "data_metric_price": "Latest Price",
        "data_metric_change": "Price Change",
        "data_metric_features": "Features",
        "data_price_history": "Price History",
        "data_preview": "Data Preview",
        "data_info": (
            "GLD historical data is fetched from Yahoo Finance (yfinance). "
            "The table shows OHLCV columns (Open, High, Low, Close, Volume) "
            "plus 28 engineered features such as moving averages, RSI, MACD, "
            "volatility measures, and lag features. These help the model "
            "detect patterns not visible in raw prices."
        ),

        # -- Tab 2: Training ------------------------------------------------
        "train_header": "Model Training",
        "train_warn_no_data": "⚠️ Please load data first in the Data tab",
        "train_btn": "Train Model",
        "train_spinner": "Training model...",
        "train_complete": "Training complete!",
        "train_success": "✅ Model trained successfully! Saved to {path}",
        "train_error": "❌ Error training model: {err}",
        "train_history": "Training History",
        "train_loss_label": "Train Loss",
        "val_loss_label": "Validation Loss",
        "train_xlabel": "Epoch",
        "train_ylabel": "Loss",
        "train_plot_title": "Training and Validation Loss",
        "train_info": (
            "Clicking 'Train Model' runs a full training loop: feature "
            "selection → target computation → sequence creation → 80/20 "
            "train/validation split → gradient-descent optimisation. "
            "Watch the loss plot: both curves should decrease. If validation "
            "loss rises while training loss falls, the model is overfitting."
        ),

        # -- Tab 3: Predictions ---------------------------------------------
        "pred_header": "Model Predictions",
        "pred_warn_no_model": "⚠️ Please train a model first in the Train Model tab",
        "pred_vs_actual": "Predictions vs Actual",
        "pred_returns": "**Predicted Returns:**",
        "pred_implied": "**Implied Price Movements:**",
        "pred_signals": "**Buy/No-Buy Signals:**",
        "pred_actual_returns": "Actual Returns",
        "pred_predicted_returns": "Predicted Returns",
        "pred_actual_price": "Actual Price",
        "pred_implied_price": "Implied Price (from prediction)",
        "pred_actual_signal": "Actual Signal",
        "pred_predicted_signal": "Predicted Signal",
        "pred_recent": "Recent Predictions",
        "pred_error": "❌ Error making predictions: {err}",
        "pred_col_date": "Date",
        "pred_col_price": "Actual Price",
        "pred_col_pred": "Prediction",
        "pred_col_true": "True Value",
        "pred_info": (
            "After training, the model runs a forward pass on every input "
            "sequence. For regression the output is the predicted return; "
            "for classification it is a Buy (>0.5) or No-Buy (≤0.5) "
            "probability. The 'Implied Price' chart multiplies the actual "
            "price by (1 + predicted return). These are historical "
            "predictions, not true future forecasts."
        ),

        # -- Tab 4: Evaluation ----------------------------------------------
        "eval_header": "Model Evaluation",
        "eval_warn_no_model": "⚠️ Please train a model first in the Train Model tab",
        "eval_regression_metrics": "Regression Metrics",
        "eval_classification_metrics": "Classification Metrics",
        "eval_confusion_matrix": "Confusion Matrix",
        "eval_cm_no_buy": "No-Buy",
        "eval_cm_buy": "Buy",
        "eval_cm_title": "Confusion Matrix",
        "eval_cm_ylabel": "True label",
        "eval_cm_xlabel": "Predicted label",
        "eval_detailed": "Detailed Metrics",
        "eval_error": "❌ Error evaluating model: {err}",
        "eval_info": (
            "Regression metrics: MSE, RMSE, MAE measure prediction error "
            "(lower is better); R² measures variance explained (1.0 = "
            "perfect). Classification metrics: Accuracy is the fraction "
            "correct, but Precision, Recall, and F1 are more informative "
            "because a naive 'always Buy' model can still reach ~60% "
            "accuracy. The confusion matrix shows TP/FP/FN/TN counts."
        ),

        # -- Diagnostics panel -----------------------------------------------
        "diag_header": "Training Diagnostics",
        "diag_verdict": "Verdict",
        "diag_verdict_healthy": "✅ Healthy",
        "diag_verdict_overfitting": "⚠️ Overfitting",
        "diag_verdict_underfitting": "⚠️ Underfitting",
        "diag_verdict_noisy": "⚠️ Noisy / Unstable",
        "diag_explanation": "Explanation",
        "diag_suggestions": "Suggestions",
        "diag_best_epoch": "Best Epoch",
        "diag_gen_gap": "Generalization Gap",

        # -- Multi-task prediction labels ----------------------------------------
        "pred_mt_returns": "**Regression Head — Predicted Returns:**",
        "pred_mt_signals": "**Classification Head — Buy/No-Buy Signals:**",
        "pred_mt_col_reg": "Predicted Return",
        "pred_mt_col_cls": "Buy Probability",

        # -- Multi-task evaluation labels ----------------------------------------
        "eval_mt_header": "Multi-task Evaluation",
        "eval_mt_threshold": "Classification Threshold",

        # -- Axis labels (plots) -------------------------------------------
        "axis_date": "Date",
        "axis_price": "Price (USD)",
        "axis_returns": "Returns",
        "axis_signal": "Signal (1=Buy, 0=No-Buy)",

        # -- Tutorial -------------------------------------------------------
        "tut_header": "📚 Tutorial — How This Application Works",
        "tut_disclaimer": (
            "> **Disclaimer:** This application is an educational tool for "
            "exploring deep learning applied to financial time series. "
            "Nothing in this guide or in the application's output constitutes "
            "financial advice."
        ),
        "tut_s1_title": "1 — Overview",
        "tut_s1_body": """
This application downloads historical price data for the **GLD** exchange-traded
fund (Gold ETF), engineers a set of technical features from that data, and then
trains a deep-learning model to **predict future price movements**.

The workflow follows four steps, each represented by a tab in the UI:

| Tab | Purpose |
|-----|---------|
| **📊 Data** | Download and explore GLD historical prices |
| **🔧 Train Model** | Configure and train a neural network |
| **📈 Predictions** | Visualise the model's forecasts |
| **📉 Evaluation** | Measure the model's accuracy with standard metrics |

The sidebar on the left lets you configure every parameter before pressing
*Train Model*.
""",
        "tut_s2_title": "2 — Data: Loading & Exploration",
        "tut_s2_body": """
### How data is loaded

GLD historical data is fetched via **yfinance**, a Python library that retrieves
daily market data from Yahoo Finance. When you press *Load Data*, the app
downloads daily OHLCV (Open, High, Low, Close, Volume) records for the date
range configured in the sidebar.

### What each column represents

| Column | Meaning |
|--------|---------|
| **Open** | The price at market open for the day |
| **High** | The highest price reached during the day |
| **Low** | The lowest price reached during the day |
| **Close** | The price at market close — the most commonly used reference |
| **Volume** | The total number of shares traded that day |
| **Dividends** | Cash dividends paid (usually 0 for GLD) |
| **Stock Splits** | Split events (usually 0 for GLD) |

### Feature engineering

Raw OHLCV data alone is not very informative for a neural network.
The application automatically creates **28 additional features** before
training, including:

- **Moving averages** (SMA, EMA at 5, 10, 20, 50 days) — smoothed trend lines
- **Volatility measures** — rolling standard deviation of returns
- **Momentum indicators** — rate of price change over different windows
- **RSI (Relative Strength Index)** — measures if the asset is overbought or
  oversold (range 0–100)
- **MACD (Moving Average Convergence Divergence)** — trend-following momentum
  indicator
- **Volume ratios** — how today's volume compares to recent averages
- **Lag features** — previous days' prices and returns fed as explicit inputs

These features help the model detect **patterns and regime changes** that are
not visible in raw price data.
""",
        "tut_s3_title": "3 — Model Architectures: GRU vs LSTM vs TCN",
        "tut_s3_body": """
### What are Recurrent Neural Networks (RNNs)?

Standard neural networks treat every input independently. **Recurrent neural
networks** (RNNs) are designed to process *sequences*: they maintain an internal
**hidden state** that is updated at each time step, allowing the model to
remember information from earlier in the sequence.

This makes RNNs naturally suited for **time-series data** such as stock prices,
where the order of observations matters.

### GRU (Gated Recurrent Unit)

The GRU is a modern RNN variant introduced in 2014. It uses two *gates*:

- **Reset gate** — decides how much past information to forget
- **Update gate** — decides how much new information to let in

GRUs are **simpler and faster** to train than LSTMs because they have fewer
parameters.

### LSTM (Long Short-Term Memory)

The LSTM, introduced in 1997, uses three gates:

- **Forget gate** — decides what to discard from the cell state
- **Input gate** — decides which new values to store
- **Output gate** — decides what part of the cell state to output

LSTMs have a separate **cell state** in addition to the hidden state, which
allows them to retain information over **longer sequences** more effectively.

### TCN (Temporal Convolutional Network)

A **TCN** replaces recurrence with stacked 1-D *causal convolutions*.
Key properties:

- **Causal padding** — the model can only see past timesteps, never the
  future, preserving temporal causality.
- **Dilated filters** — each layer doubles the dilation factor, so the
  receptive field grows *exponentially* with depth. This lets the network
  capture long-range dependencies efficiently.
- **Residual connections** — skip connections inside each block prevent
  gradient degradation in deep stacks.

Because convolutions run in parallel across the sequence (no sequential
hidden-state dependency), TCNs **train faster** than RNNs on modern GPUs.

### When to choose which?

| Criterion | GRU | LSTM | TCN |
|-----------|-----|------|-----|
| Speed | ⚡ Fast | 🐢 Slower | ⚡⚡ Fastest |
| Parameters | Fewer | More | Medium |
| Short sequences (≤ 30) | ✅ Sufficient | ✅ Works well | ✅ Good |
| Long sequences (> 60) | ⚠️ May struggle | ✅ Better retention | ✅ Large receptive field |
| Limited data | ✅ Less overfitting | ⚠️ More overfitting | ✅ Weight sharing |
| Parallelism | ❌ Sequential | ❌ Sequential | ✅ Fully parallel |

**Rule of thumb:** Start with GRU. Switch to LSTM for very long sequences
or TCN when training speed matters.

### Task types

This application supports three prediction tasks:

**Regression (Returns)**
- The model outputs a **continuous number** representing the expected
  percentage return over the prediction horizon.
- Example output: `0.012` → the model expects a +1.2 % price increase.

**Classification (Buy / No-Buy)**
- The model outputs a **probability** between 0 and 1.
- If the output is > 0.5, the signal is "**Buy**" (class 1).
- If the output is ≤ 0.5, the signal is "**No-Buy**" (class 0).

**Multi-task (Regression + Classification)**
- A single shared backbone feeds *two* prediction heads simultaneously.
- The regression head predicts returns; the classification head predicts
  buy/no-buy signals.
- Loss: *L = w_reg × MSE + w_cls × BCEWithLogits*, configurable via the
  sidebar sliders.
- Benefit: the shared representation learns richer features because it must
  satisfy both objectives at once.
""",
        "tut_s4_title": "4 — Prediction Horizons: 1, 5 & 20 Days",
        "tut_s4_body": """
The **prediction horizon** is the number of trading days into the future
that the model tries to forecast.

| Horizon | Meaning | Character |
|---------|---------|-----------|
| **1 day** | Predict tomorrow's return / signal | Short-term, noisier |
| **5 days** | Predict the return over the next week | Medium-term balance |
| **20 days** | Predict the return over the next month | Longer-term, smoother |

### Trade-offs

- **Short horizons (1 day)** capture rapid market movements but are dominated
  by **noise** (random daily fluctuations). Models can learn spurious patterns
  and may show lower accuracy.
- **Long horizons (20 days)** smooth out noise, but future events become
  **harder to predict** because more external factors can intervene.
- **Medium horizons (5 days)** often offer a good balance for learning
  meaningful patterns without excessive noise.

### Practical advice

- If you see very erratic prediction plots, try a longer horizon (5 or 20).
- If the model seems overly smoothed and never reacts, try a shorter horizon.
- Compare the evaluation metrics across horizons to find the sweet spot for
  your data range.
""",
        "tut_s5_title": "5 — Configurable Parameters Explained",
        "tut_s5_body": """
Every parameter in the sidebar affects how the model learns. Below is a
guide to each one.

---

#### Sequence Length (Lookback Window)
*Sidebar: 10–60, default 20*

The number of **consecutive days** the model looks at before making a
prediction.

| Value | Effect |
|-------|--------|
| Small (10) | Less context; faster training; may miss longer trends |
| Large (40–60) | More context; slower; risk of overfitting on small datasets |

**Recommended start:** 20.

---

#### Hidden Size
*Sidebar: 32–128, default 64*

The number of **internal neurons** in each recurrent layer — controls
the model's capacity.

| Value | Effect |
|-------|--------|
| Small (32) | Simpler; faster; less overfitting risk; may underfit |
| Large (128) | More expressive; slower; higher overfitting risk |

**Recommended start:** 64.

---

#### Number of Layers
*Sidebar: 1–4, default 2*

Stacked recurrent layers learn **hierarchical patterns**.

| Value | Effect |
|-------|--------|
| 1 | Simple and fast |
| 2 | Good default; multi-scale patterns |
| 3–4 | More powerful; needs more data |

**Recommended start:** 2.

---

#### Epochs
*Sidebar: 10–200, default 50*

One epoch = the model has seen every training sample once.

| Value | Effect |
|-------|--------|
| Low (10–20) | May underfit |
| Medium (30–80) | Good range |
| High (100–200) | Risk of overfitting |

**Tip:** If validation loss rises while training loss falls, reduce epochs.

---

#### Batch Size
*Sidebar: 16–128, default 32*

Samples processed together before a weight update.

| Value | Effect |
|-------|--------|
| Small (16) | Noisier updates; slower wall-time |
| Large (64–128) | Smoother updates; faster per epoch |

**Recommended start:** 32.

---

#### Learning Rate
*Sidebar: 0.0001–0.01, default 0.001*

How much weights change per batch.

| Value | Effect |
|-------|--------|
| Too small (0.0001) | Very slow convergence |
| Good (0.0005–0.001) | Steady learning |
| Too large (0.01) | Unstable; loss may diverge |

**Recommended start:** 0.001.
""",
        "tut_s6_title": "6 — Training: What Happens When You Click 'Train'",
        "tut_s6_body": """
### The training loop

1. **Feature selection** — 28 engineered features; missing values filled.
2. **Target computation** — returns (regression) or binary labels (classification).
3. **Sequence creation** — sliding window of *Sequence Length* days.
4. **Train / Validation split** — 80 / 20 by default.
5. **Gradient-descent loop** — for each epoch the model trains on batches,
   then evaluates on the validation set.

### Understanding the Training History plot

- **Train Loss** (blue): error on training data.
- **Validation Loss** (orange): error on unseen data.

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both decrease steadily | ✅ Good convergence | Continue or stop early |
| Train ↓ val ↑ | ⚠️ Overfitting | ↓ epochs / complexity |
| Both stay high | ⚠️ Underfitting | ↑ capacity / epochs |
| Loss oscillates | ⚠️ Unstable | ↓ learning rate |
| Flat from start | ⚠️ Not learning | ↑ learning rate |

### What is "loss"?

- **Regression:** Mean Squared Error (MSE).
- **Classification:** Binary Cross-Entropy.

Lower loss = better model.
""",
        "tut_s7_title": "7 — Predictions: Interpreting the Output",
        "tut_s7_body": """
### How predictions are generated

After training the model runs a forward pass on each input sequence
(pure inference — no gradient computation).

### Regression output

- Predicted return per date overlaid on actual returns.
- **Implied Price** = actual price × (1 + predicted return).

### Classification output

- Probability > 0.5 → **Buy (1)**; ≤ 0.5 → **No-Buy (0)**.
- Blue dots = actual; red X = predicted.

### Recent Predictions table

| Column | Meaning |
|--------|---------|
| Date | Trading day |
| Actual Price | GLD close |
| Prediction | Raw model output |
| True Value | Actual target |

### Caveats

Predictions are on historical data (train + validation), **not** true
out-of-sample forecasts. Real-world performance may differ due to regime
changes, costs, and slippage.
""",
        "tut_s8_title": "8 — Evaluation: Understanding the Metrics",
        "tut_s8_body": """
### Regression metrics

| Metric | Meaning | Good values |
|--------|---------|-------------|
| **MSE** | Avg squared error | Lower is better |
| **RMSE** | √MSE — same units as target | Lower is better |
| **MAE** | Avg absolute error | Lower is better |
| **R²** | Variance explained | 1.0 = perfect; 0 = mean-level |

In real markets R² of 0.01–0.05 can already be economically meaningful.

### Classification metrics

| Metric | Meaning |
|--------|---------|
| Accuracy | Fraction correct |
| Precision | Of Buy predictions, how many correct? |
| Recall | Of actual Buy days, how many caught? |
| F1 | Harmonic mean of Precision & Recall |

#### Confusion matrix

```
                 Predicted
              No-Buy    Buy
Actual No-Buy   TN       FP
       Buy       FN       TP
```

**Why accuracy alone is not enough:** A model that always says "Buy"
can reach ~60 % accuracy if the market is up 60 % of the time.
Precision, Recall, and F1 reveal true skill.

| Scenario | Metrics | Interpretation |
|----------|---------|----------------|
| Random guessing | R²≈0, Acc≈50% | No skill |
| Slight edge | R²≈0.01–0.05, Acc≈52–55% | Potentially useful |
| Strong (rare) | R²>0.1, F1>0.65 | Verify not overfitting |
| Overfit | R²>0.9 train only | Too good — check val |
""",
        "tut_s9_title": "9 — Practical Examples & Common Scenarios",
        "tut_s9_body": """
> **Note:** These examples are purely educational. They do NOT constitute
> financial advice.

---

**Scenario A — Positive prediction with buy signal**

Both models agree the price is likely to increase. This alignment increases
confidence, but does not guarantee a price rise. Check validation metrics.

---

**Scenario B — Validation loss rising while training loss decreases**

Classic **overfitting**. Reduce epochs, lower complexity, or add more data.

---

**Scenario C — Predictions fluctuate heavily**

May be overfitting to noise or learning rate too high. Try ↓ LR,
↑ sequence length, or longer horizon.

---

**Scenario D — Model always predicts the same value**

Collapsed to the mean. ↑ capacity, ↑ epochs, ↓ LR, or use more data.

---

**Scenario E — Very high accuracy on training data (95 %)**

Likely overfitting. True financial accuracy above 55–60 % is already good.
""",
        "tut_s10_title": "10 — Quick-Reference Cheat Sheet",
        "tut_s10_body": """
### Recommended starting configuration

| Parameter | Value |
|-----------|-------|
| Model | GRU |
| Task | Regression |
| Horizon | 5 days |
| Sequence Length | 20 |
| Hidden Size | 64 |
| Layers | 2 |
| Epochs | 50 |
| Batch Size | 32 |
| Learning Rate | 0.001 |

### Architecture quick comparison

| | GRU | LSTM | TCN |
|-|-----|------|-----|
| Best for | General use | Long sequences | Speed |
| # Parameters | Low | High | Medium |
| Training speed | Fast | Slow | Fastest |

### Common adjustments

| Problem | Try |
|---------|-----|
| Underfitting | ↑ Hidden size, ↑ Layers, ↑ Epochs |
| Overfitting | ↓ Epochs, ↓ Hidden size, ↓ Layers, ↑ Data range |
| Unstable loss | ↓ Learning rate |
| Flat predictions | ↑ Learning rate, ↑ Hidden size |
| Noisy predictions | ↑ Sequence length, ↑ Horizon, ↓ Learning rate |
| Slow training | ↓ Hidden size, ↓ Layers, ↑ Batch size, TCN or GRU |

### Diagnostics verdicts

| Verdict | Meaning | Action |
|---------|---------|--------|
| ✅ Healthy | Both curves decreasing, stable gap | Continue or stop |
| ⚠️ Overfitting | Val ↑ while train ↓ | ↓ epochs / complexity |
| ⚠️ Underfitting | Both curves high and flat | ↑ capacity / epochs |
| ⚠️ Noisy | Validation oscillates | ↓ learning rate, ↑ batch |
""",
    },

    # ======================================================================
    # SPANISH
    # ======================================================================
    "es": {
        # -- Page chrome ----------------------------------------------------
        "page_title": "Predicción del precio de GLD",
        "app_title": "🏅 Predicción del precio de GLD con Deep Learning",
        "app_subtitle": "Pronóstico de movimientos del ETF de oro con modelos GRU / LSTM / TCN",

        # -- Sidebar --------------------------------------------------------
        "sidebar_config": "Configuración",
        "sidebar_language": "Language / Idioma",
        "sidebar_data_settings": "Datos",
        "sidebar_start_date": "Fecha de inicio",
        "sidebar_end_date": "Fecha de fin",
        "sidebar_model_settings": "Modelo",
        "sidebar_model_arch": "Arquitectura del modelo",
        "sidebar_task_type": "Tipo de tarea",
        "sidebar_task_regression": "Regresión (Rendimientos)",
        "sidebar_task_classification": "Clasificación (Compra/No-Compra)",
        "sidebar_task_multitask": "Multi-tarea (Reg + Cls)",
        "sidebar_horizon": "Horizonte de predicción (días)",
        "sidebar_training_settings": "Entrenamiento",
        "sidebar_seq_length": "Longitud de secuencia",
        "sidebar_hidden_size": "Tamaño oculto",
        "sidebar_num_layers": "Número de capas",
        "sidebar_epochs": "Épocas",
        "sidebar_batch_size": "Tamaño de lote",
        "sidebar_learning_rate": "Tasa de aprendizaje",
        "sidebar_buy_threshold": "Umbral de compra",
        "sidebar_w_reg": "Peso de pérdida regresión",
        "sidebar_w_cls": "Peso de pérdida clasificación",
        "sidebar_about": "Acerca de",
        "sidebar_about_text": (
            "Esta aplicación utiliza aprendizaje profundo (GRU / LSTM / TCN) "
            "para predecir movimientos del precio de GLD. Soporta regresión, "
            "clasificación y aprendizaje multi-tarea en múltiples horizontes "
            "temporales (1, 5, 20 días) con diagnósticos automáticos."
        ),

        # -- Tab names ------------------------------------------------------
        "tab_data": "📊 Datos",
        "tab_train": "🔧 Entrenar",
        "tab_predictions": "📈 Predicciones",
        "tab_evaluation": "📉 Evaluación",
        "tab_tutorial": "📚 Tutorial",

        # -- Tab 1: Data ----------------------------------------------------
        "data_header": "Carga y exploración de datos",
        "data_load_btn": "Cargar datos",
        "data_loading_spinner": "Cargando datos de GLD...",
        "data_load_success": "✅ Se cargaron {n} registros desde {start} hasta {end}",
        "data_load_error": "❌ Error al cargar datos: {err}",
        "data_metric_records": "Registros",
        "data_metric_price": "Último precio",
        "data_metric_change": "Variación",
        "data_metric_features": "Características",
        "data_price_history": "Historia de precios",
        "data_preview": "Vista previa de datos",
        "data_info": (
            "Los datos históricos de GLD se obtienen de Yahoo Finance "
            "(yfinance). La tabla muestra columnas OHLCV (Apertura, Máximo, "
            "Mínimo, Cierre, Volumen) más 28 características calculadas como "
            "medias móviles, RSI, MACD, medidas de volatilidad y valores "
            "retardados. Estas ayudan al modelo a detectar patrones no "
            "visibles en los precios brutos."
        ),

        # -- Tab 2: Training ------------------------------------------------
        "train_header": "Entrenamiento del modelo",
        "train_warn_no_data": "⚠️ Primero cargue los datos en la pestaña Datos",
        "train_btn": "Entrenar modelo",
        "train_spinner": "Entrenando modelo...",
        "train_complete": "¡Entrenamiento completo!",
        "train_success": "✅ ¡Modelo entrenado con éxito! Guardado en {path}",
        "train_error": "❌ Error al entrenar el modelo: {err}",
        "train_history": "Historial de entrenamiento",
        "train_loss_label": "Pérdida entren.",
        "val_loss_label": "Pérdida valid.",
        "train_xlabel": "Época",
        "train_ylabel": "Pérdida",
        "train_plot_title": "Pérdida de entrenamiento y validación",
        "train_info": (
            "Al pulsar 'Entrenar modelo' se ejecuta el bucle completo: "
            "selección de características → cálculo de objetivo → creación "
            "de secuencias → división 80/20 entrenamiento/validación → "
            "optimización por descenso de gradiente. Observe la gráfica de "
            "pérdida: ambas curvas deben descender. Si la validación sube "
            "mientras el entrenamiento baja, hay sobreajuste."
        ),

        # -- Tab 3: Predictions ---------------------------------------------
        "pred_header": "Predicciones del modelo",
        "pred_warn_no_model": "⚠️ Primero entrene un modelo en la pestaña Entrenar",
        "pred_vs_actual": "Predicciones vs Real",
        "pred_returns": "**Rendimientos predichos:**",
        "pred_implied": "**Movimientos de precio implícitos:**",
        "pred_signals": "**Señales Compra/No-Compra:**",
        "pred_actual_returns": "Rendimientos reales",
        "pred_predicted_returns": "Rendimientos predichos",
        "pred_actual_price": "Precio real",
        "pred_implied_price": "Precio implícito (según predicción)",
        "pred_actual_signal": "Señal real",
        "pred_predicted_signal": "Señal predicha",
        "pred_recent": "Predicciones recientes",
        "pred_error": "❌ Error al generar predicciones: {err}",
        "pred_col_date": "Fecha",
        "pred_col_price": "Precio real",
        "pred_col_pred": "Predicción",
        "pred_col_true": "Valor real",
        "pred_info": (
            "Tras el entrenamiento, el modelo ejecuta un pase hacia adelante "
            "en cada secuencia de entrada. En regresión, la salida es el "
            "rendimiento predicho; en clasificación es una probabilidad de "
            "Compra (>0.5) o No-Compra (≤0.5). La gráfica 'Precio implícito' "
            "multiplica el precio real por (1 + rendimiento predicho). Estas "
            "son predicciones históricas, no pronósticos futuros reales."
        ),

        # -- Tab 4: Evaluation ----------------------------------------------
        "eval_header": "Evaluación del modelo",
        "eval_warn_no_model": "⚠️ Primero entrene un modelo en la pestaña Entrenar",
        "eval_regression_metrics": "Métricas de regresión",
        "eval_classification_metrics": "Métricas de clasificación",
        "eval_confusion_matrix": "Matriz de confusión",
        "eval_cm_no_buy": "No-Compra",
        "eval_cm_buy": "Compra",
        "eval_cm_title": "Matriz de confusión",
        "eval_cm_ylabel": "Etiqueta real",
        "eval_cm_xlabel": "Etiqueta predicha",
        "eval_detailed": "Métricas detalladas",
        "eval_error": "❌ Error al evaluar el modelo: {err}",
        "eval_info": (
            "Métricas de regresión: MSE, RMSE, MAE miden el error de "
            "predicción (menor es mejor); R² mide la varianza explicada "
            "(1.0 = perfecto). Métricas de clasificación: la Exactitud es "
            "la fracción correcta, pero Precisión, Sensibilidad y F1 son "
            "más informativas porque un modelo ingenuo que siempre diga "
            "'Compra' puede alcanzar ~60% de exactitud. La matriz de "
            "confusión muestra conteos TP/FP/FN/TN."
        ),

        # -- Diagnostics panel -----------------------------------------------
        "diag_header": "Diagnósticos del entrenamiento",
        "diag_verdict": "Veredicto",
        "diag_verdict_healthy": "✅ Saludable",
        "diag_verdict_overfitting": "⚠️ Sobreajuste",
        "diag_verdict_underfitting": "⚠️ Infraajuste",
        "diag_verdict_noisy": "⚠️ Ruidoso / Inestable",
        "diag_explanation": "Explicación",
        "diag_suggestions": "Sugerencias",
        "diag_best_epoch": "Mejor época",
        "diag_gen_gap": "Brecha de generalización",

        # -- Multi-task prediction labels ----------------------------------------
        "pred_mt_returns": "**Cabeza de regresión — Rendimientos predichos:**",
        "pred_mt_signals": "**Cabeza de clasificación — Señales Compra/No-Compra:**",
        "pred_mt_col_reg": "Rendimiento predicho",
        "pred_mt_col_cls": "Probabilidad de compra",

        # -- Multi-task evaluation labels ----------------------------------------
        "eval_mt_header": "Evaluación multi-tarea",
        "eval_mt_threshold": "Umbral de clasificación",

        # -- Axis labels (plots) -------------------------------------------
        "axis_date": "Fecha",
        "axis_price": "Precio (USD)",
        "axis_returns": "Rendimientos",
        "axis_signal": "Señal (1=Compra, 0=No-Compra)",

        # -- Tutorial -------------------------------------------------------
        "tut_header": "📚 Tutorial — Cómo funciona esta aplicación",
        "tut_disclaimer": (
            "> **Aviso legal:** Esta aplicación es una herramienta educativa "
            "para explorar el aprendizaje profundo aplicado a series temporales "
            "financieras. Nada en esta guía ni en la salida de la aplicación "
            "constituye asesoramiento financiero."
        ),
        "tut_s1_title": "1 — Visión general",
        "tut_s1_body": """
Esta aplicación descarga datos históricos de precios del fondo cotizado **GLD**
(ETF de oro), genera un conjunto de características técnicas a partir de esos
datos y entrena un modelo de aprendizaje profundo para **predecir movimientos
futuros del precio**.

El flujo de trabajo consta de cuatro pasos, cada uno representado por una
pestaña en la interfaz:

| Pestaña | Propósito |
|---------|-----------|
| **📊 Datos** | Descargar y explorar precios históricos de GLD |
| **🔧 Entrenar** | Configurar y entrenar una red neuronal |
| **📈 Predicciones** | Visualizar las predicciones del modelo |
| **📉 Evaluación** | Medir la precisión con métricas estándar |

La barra lateral izquierda permite configurar cada parámetro antes de pulsar
*Entrenar modelo*.
""",
        "tut_s2_title": "2 — Datos: Carga y exploración",
        "tut_s2_body": """
### Cómo se cargan los datos

Los datos históricos de GLD se obtienen mediante **yfinance**, una biblioteca
de Python que descarga datos diarios de mercado de Yahoo Finance. Al pulsar
*Cargar datos*, la aplicación descarga registros diarios OHLCV (Apertura,
Máximo, Mínimo, Cierre, Volumen) para el rango de fechas configurado.

### Qué representa cada columna

| Columna | Significado |
|---------|-------------|
| **Open** | Precio de apertura del mercado |
| **High** | Precio más alto del día |
| **Low** | Precio más bajo del día |
| **Close** | Precio de cierre — la referencia más utilizada |
| **Volume** | Número total de acciones negociadas |
| **Dividends** | Dividendos pagados (generalmente 0 para GLD) |
| **Stock Splits** | Eventos de división (generalmente 0 para GLD) |

### Ingeniería de características

Los datos OHLCV brutos por sí solos no son muy informativos para una red
neuronal. La aplicación crea automáticamente **28 características adicionales**,
incluyendo:

- **Medias móviles** (SMA, EMA a 5, 10, 20, 50 días)
- **Medidas de volatilidad** — desviación estándar móvil
- **Indicadores de impulso** — tasa de cambio del precio
- **RSI (Índice de Fuerza Relativa)** — sobrecompra / sobreventa (0–100)
- **MACD** — indicador de impulso de tendencia
- **Ratios de volumen** — volumen respecto a promedios recientes
- **Valores retardados** — precios y rendimientos de días anteriores

Estas ayudan al modelo a detectar **patrones y cambios de régimen** no visibles
en los precios brutos.
""",
        "tut_s3_title": "3 — Arquitecturas: GRU vs LSTM vs TCN",
        "tut_s3_body": """
### ¿Qué son las redes neuronales recurrentes (RNN)?

Las redes neuronales estándar tratan cada entrada de forma independiente.
Las **redes neuronales recurrentes** (RNN) procesan *secuencias*: mantienen
un **estado oculto** interno que se actualiza en cada paso temporal,
permitiendo recordar información anterior.

Esto las hace ideales para **datos de series temporales** como precios
bursátiles, donde el orden importa.

### GRU (Unidad Recurrente con Puertas)

Variante moderna (2014) con dos puertas:

- **Puerta de reinicio** — decide cuánta información pasada olvidar
- **Puerta de actualización** — decide cuánta información nueva admitir

Las GRU son **más simples y rápidas** que las LSTM.

### LSTM (Memoria a Largo-Corto Plazo)

Introducida en 1997, usa tres puertas:

- **Puerta de olvido** — qué descartar del estado de celda
- **Puerta de entrada** — qué valores nuevos almacenar
- **Puerta de salida** — qué parte del estado emitir

Las LSTM tienen un **estado de celda** separado que retiene información en
**secuencias más largas**.

### TCN (Red Convolucional Temporal)

Una **TCN** sustituye la recurrencia por convoluciones causales 1-D apiladas.
Propiedades clave:

- **Relleno causal** — solo ve pasos temporales pasados, nunca el futuro.
- **Filtros dilatados** — cada capa duplica la dilatación, así el campo
  receptivo crece *exponencialmente* con la profundidad.
- **Conexiones residuales** — evitan la degradación del gradiente.

Las convoluciones se ejecutan en paralelo (sin dependencia secuencial),
por lo que las TCN **entrenan más rápido** que las RNN en GPUs modernas.

### ¿Cuándo elegir cuál?

| Criterio | GRU | LSTM | TCN |
|----------|-----|------|-----|
| Velocidad | ⚡ Rápida | 🐢 Más lenta | ⚡⚡ Más rápida |
| Parámetros | Menos | Más | Medio |
| Secuencias cortas (≤30) | ✅ Suficiente | ✅ Funciona bien | ✅ Buena |
| Secuencias largas (>60) | ⚠️ Puede fallar | ✅ Mejor retención | ✅ Gran campo receptivo |
| Datos limitados | ✅ Menos sobreajuste | ⚠️ Más sobreajuste | ✅ Comparte pesos |
| Paralelismo | ❌ Secuencial | ❌ Secuencial | ✅ Totalmente paralela |

**Regla general:** Empiece con GRU. Cambie a LSTM para secuencias largas
o a TCN si la velocidad importa.

### Tipos de tarea

**Regresión (Rendimientos)**
- La salida es un número continuo: el rendimiento esperado.
- Ejemplo: `0.012` → +1.2 % de aumento esperado.

**Clasificación (Compra / No-Compra)**
- La salida es una probabilidad entre 0 y 1.
- > 0.5 → **Compra** (clase 1); ≤ 0.5 → **No-Compra** (clase 0).

**Multi-tarea (Regresión + Clasificación)**
- Un backbone compartido alimenta *dos* cabezas de predicción simultáneamente.
- La cabeza de regresión predice rendimientos; la de clasificación predice
  señales compra/no-compra.
- Pérdida: *L = w_reg × MSE + w_cls × BCEWithLogits*, configurable mediante
  los deslizadores de la barra lateral.
- Ventaja: la representación compartida aprende características más ricas
  al satisfacer ambos objetivos a la vez.
""",
        "tut_s4_title": "4 — Horizontes de predicción: 1, 5 y 20 días",
        "tut_s4_body": """
El **horizonte de predicción** es el número de días de negociación futuros
que el modelo intenta pronosticar.

| Horizonte | Significado | Carácter |
|-----------|-------------|----------|
| **1 día** | Rendimiento/señal de mañana | Corto plazo, más ruido |
| **5 días** | Rendimiento de la próxima semana | Equilibrio |
| **20 días** | Rendimiento del próximo mes | Largo plazo, más suave |

### Compromisos

- **Horizontes cortos (1 día):** capturan movimientos rápidos pero están
  dominados por **ruido** (fluctuaciones diarias aleatorias).
- **Horizontes largos (20 días):** suavizan el ruido, pero son **más difíciles
  de predecir** porque intervienen más factores externos.
- **Horizontes medios (5 días):** ofrecen un buen equilibrio.

### Consejo práctico

- Si las predicciones son erráticas, pruebe un horizonte más largo.
- Si son demasiado planas, pruebe uno más corto.
- Compare métricas entre horizontes para encontrar el punto óptimo.
""",
        "tut_s5_title": "5 — Parámetros configurables",
        "tut_s5_body": """
Cada parámetro de la barra lateral afecta cómo aprende el modelo.

---

#### Longitud de secuencia (ventana de observación)
*Barra lateral: 10–60, por defecto 20*

Número de días consecutivos que el modelo observa antes de predecir.

| Valor | Efecto |
|-------|--------|
| Pequeño (10) | Menos contexto; más rápido; puede perder tendencias |
| Grande (40–60) | Más contexto; más lento; riesgo de sobreajuste |

---

#### Tamaño oculto
*32–128, por defecto 64*

Neuronas internas por capa recurrente — controla la capacidad del modelo.

| Valor | Efecto |
|-------|--------|
| Pequeño (32) | Más simple; menos sobreajuste; puede infraajustar |
| Grande (128) | Más expresivo; más sobreajuste |

---

#### Número de capas
*1–4, por defecto 2*

Capas recurrentes apiladas para patrones jerárquicos.

| Valor | Efecto |
|-------|--------|
| 1 | Simple y rápido |
| 2 | Buen punto medio |
| 3–4 | Más potente; necesita más datos |

---

#### Épocas
*10–200, por defecto 50*

Una época = el modelo ha visto todas las muestras una vez.

| Valor | Efecto |
|-------|--------|
| Bajo (10–20) | Puede infraajustar |
| Medio (30–80) | Buen rango |
| Alto (100–200) | Riesgo de sobreajuste |

---

#### Tamaño de lote
*16–128, por defecto 32*

Muestras procesadas juntas antes de actualizar pesos.

| Valor | Efecto |
|-------|--------|
| Pequeño (16) | Actualizaciones ruidosas; más lento |
| Grande (64–128) | Actualizaciones suaves; más rápido por época |

---

#### Tasa de aprendizaje
*0.0001–0.01, por defecto 0.001*

Cuánto cambian los pesos por lote.

| Valor | Efecto |
|-------|--------|
| Muy baja (0.0001) | Convergencia muy lenta |
| Buena (0.0005–0.001) | Aprendizaje estable |
| Muy alta (0.01) | Inestable; puede divergir |
""",
        "tut_s6_title": "6 — Entrenamiento: Qué ocurre al pulsar 'Entrenar'",
        "tut_s6_body": """
### El bucle de entrenamiento

1. **Selección de características** — 28 características; valores faltantes rellenados.
2. **Cálculo del objetivo** — rendimientos (regresión) o etiquetas binarias (clasificación).
3. **Creación de secuencias** — ventana deslizante de *Longitud de secuencia* días.
4. **División entrenamiento / validación** — 80 / 20 por defecto.
5. **Bucle de descenso de gradiente** — en cada época el modelo entrena por lotes
   y luego evalúa en el conjunto de validación.

### Interpretación de la gráfica de historial

- **Pérdida entren.** (azul): error en datos de entrenamiento.
- **Pérdida valid.** (naranja): error en datos no vistos.

| Patrón | Diagnóstico | Acción |
|--------|-------------|--------|
| Ambas descienden | ✅ Buena convergencia | Continuar o parar |
| Entren. ↓ valid. ↑ | ⚠️ Sobreajuste | ↓ épocas / complejidad |
| Ambas altas | ⚠️ Infraajuste | ↑ capacidad / épocas |
| Oscilaciones | ⚠️ Inestable | ↓ tasa de aprendizaje |
| Plana desde el inicio | ⚠️ No aprende | ↑ tasa de aprendizaje |

### ¿Qué es la "pérdida" (loss)?

- **Regresión:** Error Cuadrático Medio (MSE).
- **Clasificación:** Entropía cruzada binaria.

Menor pérdida = mejor modelo.
""",
        "tut_s7_title": "7 — Predicciones: Interpretación de la salida",
        "tut_s7_body": """
### Cómo se generan las predicciones

El modelo ejecuta un pase hacia adelante en cada secuencia de entrada
(inferencia pura — sin cálculo de gradientes).

### Salida de regresión

- Rendimiento predicho por fecha, superpuesto a los reales.
- **Precio implícito** = precio real × (1 + rendimiento predicho).

### Salida de clasificación

- Probabilidad > 0.5 → **Compra (1)**; ≤ 0.5 → **No-Compra (0)**.
- Puntos azules = real; X rojas = predicho.

### Tabla de predicciones recientes

| Columna | Significado |
|---------|-------------|
| Fecha | Día de negociación |
| Precio real | Cierre de GLD |
| Predicción | Salida bruta del modelo |
| Valor real | Valor objetivo real |

### Advertencias

Las predicciones se hacen sobre datos históricos (entrenamiento +
validación), **no** son pronósticos futuros reales.
""",
        "tut_s8_title": "8 — Evaluación: Entender las métricas",
        "tut_s8_body": """
### Métricas de regresión

| Métrica | Significado | Buenos valores |
|---------|-------------|----------------|
| **MSE** | Error cuadrático medio | Menor es mejor |
| **RMSE** | √MSE — mismas unidades | Menor es mejor |
| **MAE** | Error absoluto medio | Menor es mejor |
| **R²** | Varianza explicada | 1.0 = perfecto; 0 = nivel de la media |

En mercados reales, R² de 0.01–0.05 ya puede ser útil económicamente.

### Métricas de clasificación

| Métrica | Significado |
|---------|-------------|
| Exactitud | Fracción correcta |
| Precisión | De las predicciones Compra, ¿cuántas correctas? |
| Sensibilidad | De los días reales Compra, ¿cuántos detectados? |
| F1 | Media armónica de Precisión y Sensibilidad |

#### Matriz de confusión

```
                    Predicho
              No-Compra  Compra
Real No-Compra   VN        FP
     Compra      FN        VP
```

**¿Por qué la exactitud sola no basta?** Un modelo que siempre diga "Compra"
puede alcanzar ~60 % si el mercado sube el 60 % del tiempo.

| Escenario | Métricas | Interpretación |
|-----------|----------|----------------|
| Azar | R²≈0, Exactitud≈50% | Sin habilidad |
| Ventaja leve | R²≈0.01–0.05, Exactitud≈52–55% | Potencialmente útil |
| Fuerte (raro) | R²>0.1, F1>0.65 | Verificar sobreajuste |
| Sobreajustado | R²>0.9 solo en entren. | Demasiado bueno |
""",
        "tut_s9_title": "9 — Ejemplos prácticos y escenarios comunes",
        "tut_s9_body": """
> **Nota:** Estos ejemplos son puramente educativos. NO constituyen
> asesoramiento financiero.

---

**Escenario A — Predicción positiva con señal de compra**

Ambos modelos coinciden en que el precio subirá. La coincidencia aumenta
la confianza, pero no garantiza el resultado. Verifique las métricas de
validación.

---

**Escenario B — Pérdida de validación sube mientras la de entrenamiento baja**

**Sobreajuste** clásico. Reduzca épocas, complejidad o aumente los datos.

---

**Escenario C — Predicciones muy fluctuantes**

Sobreajuste al ruido o tasa de aprendizaje alta. Pruebe ↓ LR,
↑ longitud de secuencia u horizonte más largo.

---

**Escenario D — El modelo siempre predice el mismo valor**

Colapsó a la media. ↑ capacidad, ↑ épocas, ↓ LR, o use más datos.

---

**Escenario E — Exactitud muy alta en entrenamiento (95 %)**

Probablemente sobreajuste. Una exactitud real superior al 55–60 % ya es buena.
""",
        "tut_s10_title": "10 — Hoja de referencia rápida",
        "tut_s10_body": """
### Configuración inicial recomendada

| Parámetro | Valor |
|-----------|-------|
| Modelo | GRU |
| Tarea | Regresión |
| Horizonte | 5 días |
| Longitud de secuencia | 20 |
| Tamaño oculto | 64 |
| Capas | 2 |
| Épocas | 50 |
| Tamaño de lote | 32 |
| Tasa de aprendizaje | 0.001 |

### Comparación rápida de arquitecturas

| | GRU | LSTM | TCN |
|-|-----|------|-----|
| Ideal para | Uso general | Secuencias largas | Velocidad |
| Parámetros | Bajo | Alto | Medio |
| Velocidad | Rápida | Lenta | Más rápida |

### Ajustes comunes

| Problema | Pruebe |
|----------|--------|
| Infraajuste | ↑ Tamaño oculto, ↑ Capas, ↑ Épocas |
| Sobreajuste | ↓ Épocas, ↓ Tamaño oculto, ↓ Capas, ↑ Rango de datos |
| Pérdida inestable | ↓ Tasa de aprendizaje |
| Predicciones planas | ↑ Tasa de aprendizaje, ↑ Tamaño oculto |
| Predicciones ruidosas | ↑ Secuencia, ↑ Horizonte, ↓ Tasa de aprendizaje |
| Entrenamiento lento | ↓ Tamaño oculto, ↓ Capas, ↑ Lote, TCN o GRU |

### Veredictos del diagnóstico

| Veredicto | Significado | Acción |
|-----------|-------------|--------|
| ✅ Saludable | Ambas curvas descienden, brecha estable | Continuar o parar |
| ⚠️ Sobreajuste | Valid. ↑ mientras entren. ↓ | ↓ épocas / complejidad |
| ⚠️ Infraajuste | Ambas curvas altas y planas | ↑ capacidad / épocas |
| ⚠️ Ruidoso | Validación oscila | ↓ tasa de aprendizaje, ↑ lote |
""",
    },
}
