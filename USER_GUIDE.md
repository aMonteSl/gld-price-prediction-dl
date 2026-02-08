# User Guide — Multi-Asset Price Prediction with Deep Learning

> **Disclaimer:** This application is an educational tool for exploring deep
> learning applied to financial time series. Nothing in this guide or in the
> application's output constitutes financial advice. Always do your own
> research before making any investment decision.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data: Loading & Exploration](#2-data-loading--exploration)
3. [Model Architectures: GRU vs LSTM vs TCN](#3-model-architectures-gru-vs-lstm-vs-tcn)
4. [Quantile Trajectory Forecasting](#4-quantile-trajectory-forecasting)
5. [Configurable Parameters Explained](#5-configurable-parameters-explained)
6. [Training: What Happens When You Click "Train"](#6-training-what-happens-when-you-click-train)
7. [Forecasting: Fan Charts & Uncertainty](#7-forecasting-fan-charts--uncertainty)
8. [Recommendations: BUY / HOLD / AVOID](#8-recommendations-buy--hold--avoid)
9. [Evaluation: Understanding the Metrics](#9-evaluation-understanding-the-metrics)
10. [Model Registry: Saving & Loading Models](#10-model-registry-saving--loading-models)
11. [Practical Examples & Common Scenarios](#11-practical-examples--common-scenarios)
12. [Quick-Reference Cheat Sheet](#12-quick-reference-cheat-sheet)

---

## 1. Overview

This application downloads historical price data for **multiple financial
assets**, engineers a set of technical features, and trains a deep-learning
model to produce **probabilistic quantile trajectory forecasts** — predicting
not just the expected future price path, but also the uncertainty around it.

### Supported Assets

| Ticker | Asset | Type |
|--------|-------|------|
| **GLD** | SPDR Gold Shares | Gold ETF |
| **SLV** | iShares Silver Trust | Silver ETF |
| **BTC-USD** | Bitcoin | Cryptocurrency |
| **PALL** | abrdn Physical Palladium | Palladium ETF |

### Workflow

The app has **six tabs**, each representing a step in the workflow:

| Tab | Purpose |
|-----|---------|
| **📊 Data** | Download and explore historical prices for any supported asset |
| **🔧 Train** | Configure and train a neural network (new or fine-tune) |
| **🔮 Forecast** | View fan-chart trajectories with P10/P50/P90 uncertainty bands |
| **💡 Recommendation** | Get BUY / HOLD / AVOID decisions with confidence scores |
| **📉 Evaluation** | Trajectory metrics + quantile calibration analysis |
| **📚 Tutorial** | Built-in guide (this content is also in the app) |

The **sidebar** on the left lets you select the asset, language, architecture,
forecast horizon, and all training hyperparameters.

---

## 2. Data: Loading & Exploration

### How data is loaded

Historical data is fetched via **yfinance**, a Python library that retrieves
daily market data from Yahoo Finance. When you press *Load Data*, the app
downloads daily OHLCV (Open, High, Low, Close, Volume) records for the
selected asset and date range.

**BTC-USD** trades 7 days a week, while ETFs (GLD, SLV, PALL) only trade on
weekdays. The app handles both calendars automatically.

### What each column represents

| Column | Meaning |
|--------|---------|
| **Open** | The price at market open |
| **High** | The highest price reached during the day |
| **Low** | The lowest price reached during the day |
| **Close** | The price at market close — the primary reference price |
| **Volume** | The total number of shares/units traded |

### Feature engineering

Raw OHLCV data alone is not very informative for a neural network. The
application automatically creates **30+ additional features** before training,
including:

- **Moving averages** (SMA at 5, 10, 20, 50, 200 days; EMA at 12, 26 days)
- **Volatility measures** — rolling standard deviation of returns, ATR
  (Average True Range), ATR as a percentage of close price
- **Momentum indicators** — rate of price change over 10/20 day windows
- **RSI (Relative Strength Index)** — overbought/oversold indicator (0–100)
- **MACD** — trend-following momentum indicator with signal line
- **Bollinger Bands** — upper/lower bands and bandwidth
- **Volume ratios** — how today's volume compares to the 20-day average
- **Price-to-SMA ratios** — relative position to the 50 and 200-day averages
- **Lag features** — previous days' returns fed as explicit inputs

These features help the model detect **patterns, trends, and regime changes**.

---

## 3. Model Architectures: GRU vs LSTM vs TCN

### What are Recurrent Neural Networks (RNNs)?

Standard neural networks treat every input independently. **Recurrent neural
networks** (RNNs) process *sequences*: they maintain an internal **hidden
state** updated at each time step, allowing the model to remember information
from earlier in the sequence.

This makes RNNs naturally suited for **time-series data** such as asset prices,
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

### TCN (Temporal Convolutional Network) — Default

TCN is a **convolutional** approach to sequence modelling. Instead of processing
one time-step at a time, a TCN uses **causal dilated 1-D convolutions**:

- **Causal** — the convolution kernel only looks at present and past values,
  never the future.
- **Dilated** — each successive layer doubles its dilation factor
  (1, 2, 4, 8, …), giving the network an exponentially large **receptive
  field** without needing many layers.
- **Residual connections** — each block adds its input to its output, helping
  gradients flow through deep stacks.

Because convolutions can be computed **fully in parallel** (no sequential
hidden-state dependency), TCNs train significantly faster than RNNs on GPU.

**TCN is the default architecture** in this application because it offers the
best balance of speed and accuracy for financial time series.

### When to choose which?

| Criterion | GRU | LSTM | TCN |
|-----------|-----|------|-----|
| Training speed | ⚡ Fast | 🐢 Slower | 🚀 Fastest (parallel) |
| Number of params | Few | More | Medium |
| Short sequences (≤ 30) | ✅ Good | ✅ Good | ✅ Good |
| Long sequences (> 60) | ⚠️ May struggle | ✅ Better | ✅ Large receptive field |
| Limited data | ✅ Low overfit risk | ⚠️ Higher overfit risk | ✅ Low overfit risk |
| GPU available | Helpful | Helpful | **Big speed-up** |

**Rule of thumb:** Start with **TCN** (the default). Try GRU if you want
simplicity and have a small dataset. Switch to LSTM if you need the strongest
long-range memory and have enough data.

---

## 4. Quantile Trajectory Forecasting

### What is trajectory forecasting?

Unlike single-point predictions (e.g., "tomorrow's return will be +0.5%"),
**trajectory forecasting** predicts an entire path of future values over
multiple steps. If you set `forecast_steps = 20`, the model predicts the
next 20 trading days simultaneously.

### What are quantiles?

Instead of producing a single "best guess" forecast, the model outputs
**multiple quantile levels** that describe the uncertainty around the
prediction:

| Quantile | Name | Meaning |
|----------|------|---------|
| **P10** (0.1) | 10th percentile | There is a 10% chance the true value will be *below* this |
| **P50** (0.5) | Median | The central "best guess" forecast |
| **P90** (0.9) | 90th percentile | There is a 10% chance the true value will be *above* this |

The region between P10 and P90 is called the **80% prediction interval** —
roughly 80% of actual outcomes should fall within this band.

### Why quantiles instead of a single point?

- A single-point forecast gives you no information about **how confident**
  the model is. Is +1% a high-confidence prediction or a wild guess?
- Quantile forecasts directly communicate uncertainty: if the P10–P90 band
  is narrow, the model is confident; if it's wide, there is high uncertainty.
- The decision engine uses this uncertainty to calibrate its recommendations.

### Pinball (Quantile) Loss

The model is trained using **pinball loss** (also called quantile loss):

- For the 0.5 quantile (median), under-predictions and over-predictions are
  penalised equally — this is equivalent to MAE.
- For the 0.1 quantile (P10), the loss penalises **over-prediction** more
  heavily — pushing the P10 estimate downward.
- For the 0.9 quantile (P90), the loss penalises **under-prediction** more
  heavily — pushing the P90 estimate upward.

This asymmetric loss function ensures that each quantile level is properly
calibrated: approximately 10% of actual values should fall below P10, and
approximately 10% should fall above P90.

### Model output

Every model in this application (`GRUForecaster`, `LSTMForecaster`,
`TCNForecaster`) produces an output tensor of shape:

```
(batch_size, forecast_steps, num_quantiles)
```

For the defaults, this is `(batch, 20, 3)` — 20 forecast steps × 3 quantiles
(P10, P50, P90).

---

## 5. Configurable Parameters Explained

Every parameter in the sidebar affects how the model learns. Below is a guide
to each one.

### Asset Selection
*Sidebar: GLD / SLV / BTC-USD / PALL*

Choose the financial asset to analyse and forecast. Each asset has different
volatility characteristics:
- **GLD / SLV / PALL** — relatively low volatility, ETF-based
- **BTC-USD** — high volatility, trades 24/7 including weekends

### Architecture
*Sidebar: TCN (default) / GRU / LSTM*

The neural network backbone. See [Section 3](#3-model-architectures-gru-vs-lstm-vs-tcn)
for detailed comparison.

### Forecast Steps (K)
*Sidebar: 5–30, default 20*

The number of future trading days the model predicts simultaneously. A higher
value produces longer trajectories but requires the model to look further ahead.

| Value | Effect |
|-------|--------|
| 5 | Short trajectory — more accurate but limited look-ahead |
| 20 | Default — good balance between horizon and accuracy |
| 30 | Long trajectory — lower per-step accuracy but broader view |

### Sequence Length (Lookback Window)
*Sidebar: 10–60, default 20*

The number of **consecutive days** the model looks at before making a
prediction. A sequence length of 20 means the model sees the last 20 trading
days of features to predict the next K days.

| Value | Effect |
|-------|--------|
| Small (10) | Less context; faster training; may miss longer trends |
| Large (40–60) | More context; slower training; risk of overfitting on small datasets |

**Recommended start:** 20.

### Hidden Size
*Sidebar: 32–128, default 64*

The number of **internal neurons** in each layer. Controls the model's
**capacity** — how complex the patterns it can learn.

| Value | Effect |
|-------|--------|
| Small (32) | Simpler model; faster; less risk of overfitting; may underfit |
| Large (128) | More expressive; slower; higher risk of overfitting |

### Number of Layers
*Sidebar: 1–4, default 2*

How many layers are **stacked**. Deeper models can learn **hierarchical
patterns** (short-term fluctuations in layer 1, longer trends in layer 2).

| Value | Effect |
|-------|--------|
| 1 | Simple model, fast to train |
| 2 | Good default; captures multi-scale patterns |
| 3–4 | More powerful but slower; needs more data |

### Dropout
*Sidebar: 0.0–0.5, default 0.2*

The probability of randomly "dropping" neurons during training. Acts as
**regularisation** to prevent overfitting.

| Value | Effect |
|-------|--------|
| 0.0 | No dropout — maximum capacity, higher overfit risk |
| 0.2 | Mild regularisation (default) |
| 0.5 | Strong regularisation — may underfit |

### Epochs
*Sidebar: 10–200, default 50*

One epoch = the model has seen every training sample once. More epochs give
the model more chances to learn, but too many cause **overfitting**.

**Tip:** Watch the training history. If validation loss starts **increasing**
while training loss keeps decreasing, you have trained too many epochs.

### Batch Size
*Sidebar: 16–128, default 32*

The number of training samples processed together before updating weights.

| Value | Effect |
|-------|--------|
| Small (16) | Noisier updates; can escape local minima |
| Large (64–128) | Smoother updates; faster per-epoch |

### Learning Rate
*Sidebar: 0.0001–0.01, default 0.001*

Controls how much weights change after each batch.

| Value | Effect |
|-------|--------|
| Too small (0.0001) | Very slow convergence |
| Good range (0.0005–0.001) | Steady learning |
| Too large (0.01) | May diverge or oscillate |

---

## 6. Training: What Happens When You Click "Train"

### New Training vs Fine-Tuning

The Train tab offers two modes:

- **New Training** — starts a fresh model from scratch
- **Fine-Tune** — loads an existing model from the registry and continues
  training on new or additional data. The scaler and weights are preserved.

### The training loop

When you click **Train Model**, the following steps happen:

1. **Feature engineering** — 30+ technical features are computed from OHLCV data.
2. **Feature selection** — Relevant features are selected and missing values
   are filled (forward-fill then back-fill).
3. **Sequence creation** — A sliding window of *Sequence Length* days creates
   input samples; each sample has a **multi-step target** of K future returns.
4. **Train / Validation split** — 80% training, 20% validation (temporal split
   — no data leakage from the future).
5. **Gradient-descent loop** — For each epoch:
   - The model processes all training batches and updates weights via pinball loss.
   - Validation loss is computed without updating weights.
   - Both losses are recorded.

### Understanding the Training History plot

The plot shows two curves over epochs:

- **Train Loss** (blue): pinball loss on training data.
- **Validation Loss** (orange): pinball loss on unseen validation data.

Both should **decrease** over time.

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both curves decrease steadily | ✅ **Healthy** | Continue or stop early |
| Train loss low, val loss high / rising | ⚠️ **Overfitting** | Reduce epochs, capacity, or add dropout |
| Both curves stay high | ⚠️ **Underfitting** | Increase capacity, layers, or epochs |
| Loss oscillates wildly | ⚠️ **Noisy** | Reduce learning rate; increase batch size |

### What is pinball loss?

Unlike MSE (regression) or BCE (classification) used in v2, this application
uses **pinball (quantile) loss** for all models:

- The loss is asymmetric — it penalises under-predictions and over-predictions
  differently depending on the quantile level.
- Lower loss = better-calibrated quantile estimates.
- There is **no separate regression/classification mode** — all models
  produce quantile trajectories.

### Automatic Diagnostics

After training, the app analyses loss curves automatically:

| Verdict | Meaning | Suggested action |
|---------|---------|-----------------|
| ✅ **Healthy** | Both losses decreased steadily | No action needed |
| ⚠️ **Overfitting** | Validation loss diverged upward | ↓ Epochs, ↓ Capacity, ↑ Dropout |
| ⚠️ **Underfitting** | Both losses stayed high | ↑ Capacity, ↑ Epochs |
| ⚠️ **Noisy** | Validation loss oscillated excessively | ↓ Learning rate, ↑ Batch size |

The diagnostics panel also shows the **best epoch**, **generalisation gap**,
and **concrete suggestions**.

### Saving to the Registry

After training completes, the trained model, scaler, and metadata are
automatically saved to the **model registry** (see Section 10).

---

## 7. Forecasting: Fan Charts & Uncertainty

### How forecasts are generated

After training, the **Forecast tab** generates a trajectory forecast from the
most recent data:

1. The last *Sequence Length* days of features are extracted from the dataset.
2. The model produces K-step return forecasts at P10, P50, and P90.
3. Returns are converted to **absolute prices** by compounding from the last
   known closing price.
4. Future dates are generated (business days for ETFs, calendar days for
   BTC-USD).

### Reading the Fan Chart

The Plotly fan chart shows:

- **Dark line (P50)** — the median forecast (best guess).
- **Shaded band (P10–P90)** — the 80% prediction interval.
- **Dashed black line** — the last known actual price.

#### Narrow band → High confidence
If the P10–P90 band is tight around P50, the model is relatively sure about
the direction and magnitude of the price movement.

#### Wide band → High uncertainty
If the band fans out widely, the model is uncertain. This often happens for:
- Highly volatile assets (BTC-USD)
- Longer forecast horizons (further into the future = more uncertainty)
- Under-trained models

### Forecast Table

Below the chart, a table shows the **daily forecast values** — Date, P10, P50,
and P90 prices for each forecast step.

---

## 8. Recommendations: BUY / HOLD / AVOID

### How the Decision Engine works

The **Recommendation tab** converts the forecast trajectory into an actionable
signal: **BUY**, **HOLD**, or **AVOID**.

The decision engine scores multiple factors on a 0–100 scale:

| Factor | What it measures | Impact |
|--------|-----------------|--------|
| **Expected return** | P50 median return over the decision horizon | ±20 points |
| **Uncertainty width** | Distance between P10 and P90 | −5 to −10 points if too wide |
| **Trend filter** | Price relative to SMA50 and SMA200 | ±5 to ±15 points |
| **Volatility filter** | ATR% relative to asset-specific threshold | −5 to −10 points if too high |
| **Diagnostics gate** | Was the model training healthy? | −5 to −10 points if unhealthy |

### Decision Thresholds

| Confidence Score | Signal |
|-----------------|--------|
| ≥ 65 | **BUY** — Strong upward expectation with manageable risk |
| 36–64 | **HOLD** — Mixed signals; insufficient edge |
| ≤ 35 | **AVOID** — Negative expectation or excessive uncertainty |

### Reading the Recommendation Card

The recommendation card shows:

- **Signal** — BUY (green), HOLD (yellow), or AVOID (red)
- **Confidence** — 0–100 score
- **Rationale** — human-readable explanation of the main factors
- **Warnings** — specific risk factors (e.g., "High ATR% volatility",
  "Model diagnostics indicate overfitting")

### Important Disclaimer

The recommendation engine is a **decision-support tool**, not financial advice.
It synthesises information from the model's output but cannot account for:
- Breaking news or geopolitical events
- Transaction costs and slippage
- Your personal risk tolerance and investment horizon
- Market microstructure and liquidity

**Always do your own research** and use the recommendations as one input
among many.

---

## 9. Evaluation: Understanding the Metrics

### Trajectory Metrics

These metrics evaluate the **median (P50) forecast** against actual values:

| Metric | Full Name | Meaning | Good values |
|--------|-----------|---------|-------------|
| **MSE** | Mean Squared Error | Average squared error across all steps | Lower is better |
| **RMSE** | Root MSE | Same units as the target | Lower is better |
| **MAE** | Mean Absolute Error | Average absolute error — less sensitive to outliers | Lower is better |
| **Directional Accuracy** | — | % of steps where predicted direction matched actual | > 50% is better than random |

#### Per-Step Metrics

The Evaluation tab also shows **per-step breakdowns** — how MAE and directional
accuracy change at each forecast step (step 1, step 2, … step K). Typically:
- **Earlier steps** (1–5) are more accurate.
- **Later steps** (15–20) have higher error and lower directional accuracy.

### Quantile Calibration Metrics

These metrics check whether the quantile estimates are **well-calibrated**:

| Metric | Meaning | Ideal value |
|--------|---------|-------------|
| **Q10 Coverage** | % of actual values below P10 | ~10% |
| **Q50 Coverage** | % of actual values below P50 | ~50% |
| **Q90 Coverage** | % of actual values below P90 | ~90% |
| **Q10 Calibration Error** | |Actual coverage − 10%| | 0% |
| **Q90 Calibration Error** | |Actual coverage − 90%| | 0% |
| **Mean Interval Width** | Average P90 − P10 | Narrower is better (but not at the cost of coverage) |

#### Interpreting Calibration

- If **Q90 Coverage = 95%** instead of 90%, the P90 estimates are too
  conservative (too high) — the model is over-estimating uncertainty.
- If **Q10 Coverage = 25%** instead of 10%, the P10 estimates are too
  conservative (too low).
- Good calibration means coverage percentages are **close to their nominal
  quantile levels**.

---

## 10. Model Registry: Saving & Loading Models

### What is the Model Registry?

The model registry is a persistent storage system for trained models. Each
saved model includes:

- **Model weights** (the trained neural network parameters)
- **Scaler** (the `StandardScaler` fitted during training — essential for
  correct inference)
- **Metadata** (architecture, asset, forecast steps, quantiles, training date,
  final validation loss, number of epochs)

### Where models are stored

Models are saved in `data/model_registry/` (git-ignored). Each model has:
- A **model ID** (auto-generated filesystem-safe identifier, e.g., `20261208_143052_abc123ef`)
- A **label** (custom human-readable name, e.g., `GLD_TCN_multistep_K20_v1`)

### Custom Model Names

When training a model, you can provide a **custom label** to make it memorable
and easy to identify later:

**In the Streamlit app:**
1. Before clicking "Train Model", enter a name in the **Custom Model Name** field
2. Examples: `GLD_TCN_v1`, `BTC_high_vol_experiment`, `SLV_LSTM_20_steps`
3. If left empty, the label auto-generates (e.g., `GLD_TCN_20261208_143052`)
4. Maximum 60 characters

**From code:**
```python
model_id = registry.save_model(
    model, scaler, config, feature_names, training_summary,
    label="MyAwesomeModel_v1"  # Optional custom label
)
```

### Using the Registry

**From the Streamlit app:**
- After training, the model is automatically saved to the registry.
- In the Train tab, select **Fine-tune** and choose a saved model from the
  dropdown (models are listed by label, not model ID).
- The **Delete Models** expander at the bottom of the Train tab lets you
  delete individual models or bulk-delete by asset.

**From code:**
```python
from gldpred.registry import ModelRegistry
from gldpred.models import TCNForecaster

registry = ModelRegistry()

# List saved models
models = registry.list_models(asset="GLD", architecture="TCN")
for m in models:
    print(m["label"], m["asset"], m["created_at"], m.get("training_summary", {}).get("final_val_loss"))

# Load a model (use model_id, not label)
model, scaler, metadata = registry.load_model(
    model_id="20261208_143052_abc123ef",
    model_class=TCNForecaster,
    input_size=30
)
```

### Deleting Models

**Why delete models?**
- Free disk space from experiments you no longer need
- Keep the registry clean and organized
- Remove poorly-performing models

**From the Streamlit app:**
1. Go to the Train tab
2. Expand the **Delete Models** section at the bottom
3. **Delete single model:**
   - Select the model from the dropdown
   - Type `DELETE` exactly in the confirmation field
   - Click "Confirm Delete"
4. **Delete all models:**
   - Choose scope (all models or just current asset)
   - Type `DELETE ALL` exactly in the confirmation field
   - Click "Confirm Delete"

**From code:**
```python
# Delete a single model
registry.delete_model(model_id="20261208_143052_abc123ef")

# Delete all GLD models
registry.delete_all_models(asset="GLD", confirmed=True)

# Delete ALL models (use with extreme caution)
registry.delete_all_models(confirmed=True)
```

**Safety notes:**
- Deletion is **permanent** — there is no undo
- Model files (weights, scaler, metadata) are removed from disk
- The confirmation prompts prevent accidental deletion
- Attempting to delete a non-existent model raises `FileNotFoundError`

### Model ID vs Label

| Aspect | Model ID | Label |
|--------|----------|-------|
| **Purpose** | Internal filesystem identifier | Human-readable display name |
| **Format** | Auto-generated timestamp + UUID | Your custom text |
| **Example** | `20261208_143052_abc123ef` | `GLD_TCN_multistep_K20_v1` |
| **Where used** | API calls, file paths | UI dropdowns, lists, success messages |
| **Can change?** | No — fixed at creation | No — set once at training |

When you list or fine-tune models in the app, you see the **label** (e.g.,
`MyAwesomeModel_v1 (TCN, 2026-12-08)`). Internally, the registry uses the
**model ID** to locate files.

---

## 11. Practical Examples & Common Scenarios

> **Note:** These examples are purely educational illustrations. They do NOT
> constitute financial advice.

### Scenario A — Strong BUY signal

*"The model predicts GLD P50 increasing from $240 to $248 over 20 days. The
P10–P90 band is narrow ($236–$252). Confidence: 78."*

**Interpretation:** The model sees a clear uptrend with manageable uncertainty.
The narrow band suggests high conviction.

**What to check:**
- Is the model well-trained? (Healthy diagnostics verdict)
- Have quantile calibration metrics been reviewed?
- Is there a macro event that the model cannot know about?

### Scenario B — Wide uncertainty band

*"The fan chart for BTC-USD shows P10 at $55,000 and P90 at $85,000 after 20
days. P50 is at $70,000."*

**Interpretation:** The model is highly uncertain. A $30,000 range means
almost anything could happen. The recommendation will likely be **HOLD** due
to excessive uncertainty width.

**What to do:**
- This is normal for BTC-USD — it's inherently more volatile.
- Consider reducing `forecast_steps` to 5–10 for tighter shorter-term forecasts.
- If the uncertainty is unusually high, the model may need more training data.

### Scenario C — Validation loss rising while training loss decreases

*"After epoch 30, the training loss keeps falling but the validation loss starts
climbing."*

**Diagnosis:** Classic **overfitting**. The model is memorising training data.

**What to do:**
- Reduce epochs to ~30.
- Reduce model complexity: lower hidden size or number of layers.
- Increase dropout (try 0.3).
- Use a longer date range for more training data.

### Scenario D — AVOID recommendation with low confidence

*"Signal: AVOID. Confidence: 22. Warnings: High volatility (ATR% > threshold),
model diagnostics indicate noisy training, negative expected return."*

**Interpretation:** Multiple negative factors are stacking up. The model's
training was noisy (unreliable weights) and the asset is showing high
volatility. This is a strong signal to stay on the sidelines.

### Scenario E — Model always predicts flat returns

*"The P50 forecast is essentially a straight line at the current price."*

**Diagnosis:** The model has not learned useful patterns — it has collapsed
to predicting near-zero returns.

**What to do:**
- Increase model capacity: raise hidden size or number of layers.
- Increase epochs.
- Lower the learning rate for more stable gradient updates.
- Make sure the data has enough variability (use a longer date range).

### Scenario F — Good trajectory metrics but poor calibration

*"MAE is low and directional accuracy is 58%, but Q10 Coverage is 30% instead
of 10%."*

**Interpretation:** The median forecast is reasonable, but the uncertainty
estimates are too conservative (P10 is too low). The P10–P90 band is wider
than it needs to be.

**What to do:** This sometimes happens with limited training data. The model
will improve its calibration with more data or longer training.

---

## 12. Quick-Reference Cheat Sheet

### Recommended starting configuration

| Parameter | Value |
|-----------|-------|
| Asset | GLD |
| Architecture | TCN |
| Forecast Steps | 20 |
| Sequence Length | 20 |
| Hidden Size | 64 |
| Layers | 2 |
| Dropout | 0.2 |
| Epochs | 50 |
| Batch Size | 32 |
| Learning Rate | 0.001 |

### Architecture comparison

| Property | GRU | LSTM | TCN |
|----------|-----|------|-----|
| Type | Recurrent | Recurrent | Convolutional |
| Speed | Fast | Moderate | **Fastest** |
| Long sequences | Adequate | Best | Very good |
| GPU benefit | Moderate | Moderate | Large |
| **Default?** | No | No | **Yes** |

### Diagnostics verdicts

| Verdict | Meaning | Suggested action |
|---------|---------|-----------------|
| ✅ Healthy | Both losses decrease | Continue or stop early |
| ⚠️ Overfitting | Val loss diverges up | ↓ Epochs, ↓ Capacity, ↑ Dropout |
| ⚠️ Underfitting | Both losses stay high | ↑ Capacity, ↑ Epochs |
| ⚠️ Noisy | Val loss oscillates | ↓ Learning rate, ↑ Batch size |

### Decision signals

| Signal | Confidence | Meaning |
|--------|-----------|---------|
| **BUY** | ≥ 65 | Strong upward expectation, manageable risk |
| **HOLD** | 36–64 | Mixed signals, insufficient edge |
| **AVOID** | ≤ 35 | Negative expectation or excessive uncertainty |

### Common adjustments

| Problem | Try |
|---------|-----|
| Underfitting | ↑ Hidden size, ↑ Layers, ↑ Epochs |
| Overfitting | ↓ Epochs, ↓ Hidden size, ↑ Dropout, ↑ Data range |
| Unstable loss | ↓ Learning rate |
| Flat predictions | ↑ Learning rate, ↑ Hidden size |
| Wide uncertainty band | ↓ Forecast steps, ↑ Data, ↑ Epochs |
| Slow training | Switch to TCN, ↑ Batch size |
| Poor calibration | ↑ Data range, ↑ Epochs |
