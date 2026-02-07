# User Guide — GLD Price Prediction with Deep Learning

> **Disclaimer:** This application is an educational tool for exploring deep
> learning applied to financial time series. Nothing in this guide or in the
> application's output constitutes financial advice.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data: Loading & Exploration](#2-data-loading--exploration)
3. [Model Architectures: GRU vs LSTM vs TCN](#3-model-architectures-gru-vs-lstm-vs-tcn)
4. [Prediction Horizons: 1, 5 & 20 Days](#4-prediction-horizons-1-5--20-days)
5. [Configurable Parameters Explained](#5-configurable-parameters-explained)
6. [Training: What Happens When You Click "Train"](#6-training-what-happens-when-you-click-train)
7. [Predictions: Interpreting the Output](#7-predictions-interpreting-the-output)
8. [Evaluation: Understanding the Metrics](#8-evaluation-understanding-the-metrics)
9. [Practical Examples & Common Scenarios](#9-practical-examples--common-scenarios)
10. [Quick-Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. Overview

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
| **📚 Tutorial** | In-app version of this guide |

The sidebar on the left lets you configure every parameter before pressing
*Train Model*.

---

## 2. Data: Loading & Exploration

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

Raw OHLCV data alone is not very informative for a neural network. The
application automatically creates **28 additional features** before training,
including:

- **Moving averages** (SMA, EMA at 5, 10, 20, 50 days) — smoothed trend lines
- **Volatility measures** — rolling standard deviation of returns
- **Momentum indicators** — rate of price change over different windows
- **RSI (Relative Strength Index)** — measures if the asset is overbought or
  oversold (range 0–100)
- **MACD (Moving Average Convergence Divergence)** — trend-following momentum indicator
- **Volume ratios** — how today's volume compares to recent averages
- **Lag features** — previous days' prices and returns fed as explicit inputs

These features help the model detect **patterns and regime changes** that are
not visible in raw price data.

---

## 3. Model Architectures: GRU vs LSTM vs TCN

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

TCN is a **convolutional** approach to sequence modelling, introduced as an
alternative to RNNs. Instead of processing one time-step at a time, a TCN uses
**causal dilated 1-D convolutions**:

- **Causal** — the convolution kernel only looks at present and past values,
  never the future.
- **Dilated** — each successive layer doubles its dilation factor
  (1, 2, 4, 8, …), giving the network an exponentially large **receptive
  field** without needing many layers.
- **Residual connections** — each block adds its input to its output, helping
  gradients flow through deep stacks.

Because convolutions can be computed **fully in parallel** (no sequential
hidden-state dependency), TCNs train significantly faster than RNNs on GPU.

### When to choose which?

| Criterion | GRU | LSTM | TCN |
|-----------|-----|------|-----|
| Training speed | ⚡ Fast | 🐢 Slower | 🚀 Fastest (parallel) |
| Number of params | Few | More | Medium |
| Short sequences (≤ 30) | ✅ Good | ✅ Good | ✅ Good |
| Long sequences (> 60) | ⚠️ May struggle | ✅ Better | ✅ Large receptive field |
| Limited data | ✅ Low overfit risk | ⚠️ Higher overfit risk | ✅ Low overfit risk |
| GPU available | Helpful | Helpful | **Big speed-up** |

**Rule of thumb:** Start with GRU for simplicity. Try TCN when you want faster
training or longer sequences. Switch to LSTM if you need the strongest
long-range memory and have enough data.

### Task types

This application supports **three** prediction tasks:

**Regression (Returns)**
- The model outputs a **continuous number** representing the expected
  percentage return over the prediction horizon.
- Example output: `0.012` → the model expects a +1.2 % price increase.

**Classification (Buy / No-Buy)**
- The model outputs a **probability** between 0 and 1.
- If the output is > 0.5, the signal is "**Buy**" (class 1).
- If the output is ≤ 0.5, the signal is "**No-Buy**" (class 0).
- The ground truth is derived from whether the actual future return exceeds
  the **buy threshold** (configurable, default 0.3 %).

**Multi-task (Regression + Classification)**
- The model has a **shared backbone** (GRU, LSTM, or TCN) that feeds into
  two separate heads:
  - **Regression head** — predicts the expected return (continuous)
  - **Classification head** — predicts the buy probability (sigmoid)
- The combined loss is: *L = w_reg × MSE + w_cls × BCEWithLogits*
- The loss weights `w_reg` and `w_cls` are adjustable in the sidebar.
- This approach forces the backbone to learn features that are useful for
  **both** tasks, often improving generalisation compared to training two
  separate models.

---

## 4. Prediction Horizons: 1, 5 & 20 Days

The **prediction horizon** is the number of trading days into the future that
the model tries to forecast.

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

---

## 5. Configurable Parameters Explained

Every parameter in the sidebar affects how the model learns. Below is a guide
to each one.

### Sequence Length (Lookback Window)
*Sidebar: 10–60, default 20*

The number of **consecutive days** the model looks at before making a
prediction. A sequence length of 20 means the model sees the last 20 trading
days of features to predict the next movement.

| Value | Effect |
|-------|--------|
| Small (10) | Model sees less context; faster training; may miss longer trends |
| Large (40–60) | Model sees more context; slower training; risk of overfitting on small datasets |

**Recommended start:** 20.

### Hidden Size
*Sidebar: 32–128, default 64*

The number of **internal neurons** in each recurrent layer. This controls the
model's **capacity** — how complex the patterns it can learn.

| Value | Effect |
|-------|--------|
| Small (32) | Simpler model; faster; less risk of overfitting; may underfit |
| Large (128) | More expressive; slower; higher risk of overfitting |

**Recommended start:** 64.

### Number of Layers
*Sidebar: 1–4, default 2*

How many recurrent layers are **stacked** on top of each other. Deeper models
can learn **hierarchical patterns** (e.g., short-term fluctuations in layer 1,
longer trends in layer 2).

| Value | Effect |
|-------|--------|
| 1 | Simple model, fast to train |
| 2 | Good default; captures multi-scale patterns |
| 3–4 | More powerful but significantly slower; needs more data |

**Recommended start:** 2.

### Epochs
*Sidebar: 10–200, default 50*

One epoch means the model has seen **every training sample once**. More epochs
give the model more chances to learn, but too many cause **overfitting**
(memorising the training data instead of generalising).

| Value | Effect |
|-------|--------|
| Low (10–20) | May underfit — the model hasn't learned enough |
| Medium (30–80) | Good range for most experiments |
| High (100–200) | Risk of overfitting unless you have a lot of data |

**Tip:** Watch the training history plot. If validation loss starts
**increasing** while training loss keeps decreasing, you have trained too many
epochs.

### Batch Size
*Sidebar: 16–128, default 32*

The number of training samples processed **together** before updating the
model's weights.

| Value | Effect |
|-------|--------|
| Small (16) | Noisier weight updates; can escape local minima; slower wall-time |
| Large (64–128) | Smoother updates; faster per-epoch; may converge to a flatter minimum |

**Recommended start:** 32.

### Learning Rate
*Sidebar: 0.0001–0.01, default 0.001*

Controls **how much** the model's weights change after each batch.

| Value | Effect |
|-------|--------|
| Too small (0.0001) | Very slow convergence; may get stuck |
| Good range (0.0005–0.001) | Steady learning |
| Too large (0.01) | Training may become unstable; loss may oscillate or diverge |

**Recommended start:** 0.001.

### Buy Threshold (Classification & Multi-task only)
*Sidebar: 0.0–0.02, default 0.003*

The **minimum return** that counts as a "Buy" label. A future return above
this threshold is labelled 1 (Buy); at or below it is labelled 0 (No-Buy).

| Value | Effect |
|-------|--------|
| 0.0 | Any positive return is a Buy — balanced but noisy labels |
| 0.003 (0.3 %) | Requires a meaningful move; fewer but higher-quality Buy labels |
| 0.01 (1 %) | Very selective — only strong moves are labelled Buy |

**Recommended start:** 0.003 (0.3 %).

### Loss Weights — w_reg / w_cls (Multi-task only)
*Sidebar: 0.0–2.0, defaults 1.0 / 1.0*

In multi-task mode the total loss is:

> *L = w_reg × MSE + w_cls × BCEWithLogits*

These sliders let you emphasise one objective over the other.

| Setting | Effect |
|---------|--------|
| w_reg = 1, w_cls = 1 | Equal importance (default) |
| w_reg = 1, w_cls = 0.5 | Prioritise return prediction |
| w_reg = 0.5, w_cls = 1 | Prioritise buy/no-buy signal quality |

**Recommended start:** keep both at 1.0 and adjust only if one head clearly
underperforms.

---

## 6. Training: What Happens When You Click "Train"

### The training loop

When you click **Train Model**, the following steps happen:

1. **Feature selection** — The 28 engineered features are selected and any
   missing values are filled.
2. **Target computation** — Depending on the task:
   - *Regression:* future returns at the chosen horizon are calculated.
   - *Classification:* future returns are converted to binary labels
     (1 = return above buy threshold → *Buy*, 0 = below → *No-Buy*).
   - *Multi-task:* both the continuous return and the binary label are
     computed and passed to the model together.
3. **Sequence creation** — A sliding window of *Sequence Length* days is
   applied to create input samples.
4. **Train / Validation split** — Data is split (by default 80 / 20) so the
   model is evaluated on data it has never seen during training.
5. **Gradient-descent loop** — For each epoch:
   - The model sees all training batches and updates its weights.
   - Then it evaluates on the validation set *without* updating weights.
   - Both *train loss* and *validation loss* are recorded.

### Understanding the Training History plot

The plot shows two curves over epochs:

- **Train Loss** (blue): how wrong the model is on the data it trains on.
- **Validation Loss** (orange): how wrong the model is on unseen data.

Both should **decrease** over time. The key diagnostic patterns are:

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both curves decrease steadily | ✅ **Good convergence** | Continue or stop early |
| Train loss low, val loss high / rising | ⚠️ **Overfitting** | Reduce epochs, hidden size, or layers; add more data |
| Both curves stay high | ⚠️ **Underfitting** | Increase hidden size, layers, or epochs; check data quality |
| Loss oscillates wildly | ⚠️ **Unstable training** | Reduce learning rate |
| Loss is flat from the start | ⚠️ **Not learning** | Increase learning rate or check data quality |

### What is "loss"?

- For **regression**, the loss is **Mean Squared Error (MSE)** — the
  average squared difference between predicted and actual returns.
- For **classification**, the loss is **Binary Cross-Entropy** — measures how
  well the predicted probability matches the true 0/1 label.
- For **multi-task**, the loss is the weighted sum:
  *L = w_reg × MSE + w_cls × BCEWithLogits*.

Lower loss = better model.

### Automatic Diagnostics

After training completes, the app analyses the loss curves and displays a
**diagnostics panel** with four possible verdicts:

| Verdict | Meaning | What the app suggests |
|---------|---------|----------------------|
| ✅ **Healthy** | Both losses decreased steadily | No action needed |
| ⚠️ **Overfitting** | Validation loss increased while training loss decreased | Reduce epochs, hidden size, or layers; add more data |
| ⚠️ **Underfitting** | Both losses remained high; model didn't converge | Increase capacity (hidden size / layers / epochs) |
| ⚠️ **Noisy** | Validation loss oscillated excessively | Reduce learning rate; increase batch size |

The panel also shows:
- **Best epoch** — the epoch with the lowest validation loss
- **Generalisation gap** — difference between final validation and training
  loss; a large gap suggests overfitting
- **Suggestions** — a bullet list of concrete actions you can take

---

## 7. Predictions: Interpreting the Output

### How predictions are generated

After training, the model processes each input sequence (the last *N* days of
features) through the neural network in a single forward pass. No gradient
computation is performed — this is pure inference.

### Regression output

- The model produces a **predicted return** for each date in the dataset.
- The *Predictions vs Actual* plot overlays the true historical returns (blue
  line) with the model's predicted returns (red dashed line).
- A close match means the model has learned the dominant patterns.
- An **Implied Price** chart is also shown — it multiplies the current actual
  price by (1 + predicted return) to show what price the model would "expect".

### Classification output

- The model outputs a probability for each date. Values > 0.5 are interpreted
  as **Buy (1)** and values ≤ 0.5 as **No-Buy (0)**.
- The plot shows actual signals (blue dots) vs predicted signals (red X's).
- Clusters of correct predictions indicate the model is picking up a real
  pattern.

### Multi-task output

In multi-task mode, the model produces **two outputs simultaneously**:

1. **Regression head** — predicted returns, displayed in a line chart just like
   the regression-only task.
2. **Classification head** — buy/no-buy signals, displayed in a scatter plot
   just like the classification-only task.

This lets you see whether both heads agree (e.g., predicted return is positive
*and* the buy signal fires), which increases confidence in the prediction.

### Recent Predictions table

The table at the bottom of the Predictions tab shows the last 20 data points
with:
- **Date** — the trading day
- **Actual Price** — the GLD closing price
- **Prediction** — the model's raw output (return or probability)
- **True Value** — the actual target value for that date

### Important caveats

- Predictions are made on **historical data the model has already seen
  (training set) or held out (validation set)**. They are *not* true
  out-of-sample forecasts into the future.
- Even if metrics look good, real-world performance can differ due to market
  regime changes, transaction costs, and slippage.

---

## 8. Evaluation: Understanding the Metrics

### Regression metrics

| Metric | Full Name | Meaning | Good values |
|--------|-----------|---------|-------------|
| **MSE** | Mean Squared Error | Average of squared prediction errors | Lower is better; scale depends on data |
| **RMSE** | Root Mean Squared Error | Square root of MSE — same units as the target | Lower is better |
| **MAE** | Mean Absolute Error | Average of absolute errors — less sensitive to outliers | Lower is better |
| **R²** | Coefficient of Determination | Proportion of variance explained by the model | 1.0 = perfect; 0.0 = no better than the mean; < 0 = worse than the mean |

**Practical interpretation:**
- An **R² of 0.6** means the model explains 60 % of the variance in future
  returns — this would be exceptionally good for financial data.
- In real markets, **R² values of 0.01–0.05** can already be economically
  meaningful if they are stable out of sample.

### Classification metrics

| Metric | Meaning |
|--------|---------|
| **Accuracy** | Fraction of correct predictions (both Buy and No-Buy) |
| **Precision** | Of all *Buy* predictions, how many were truly positive? |
| **Recall** | Of all truly positive days, how many did the model catch? |
| **F1 Score** | Harmonic mean of Precision and Recall — balances both |

#### The confusion matrix

```
                 Predicted
              No-Buy    Buy
Actual No-Buy   TN       FP
       Buy       FN       TP
```

- **TN (True Negatives):** Correctly predicted No-Buy
- **FP (False Positives):** Predicted Buy but the price actually dropped — *you
  would have entered a losing trade*
- **FN (False Negatives):** Predicted No-Buy but the price actually rose — *you
  missed a profitable opportunity*
- **TP (True Positives):** Correctly predicted Buy

#### Why accuracy alone is not enough

If the market goes up 60 % of the time, a model that **always says "Buy"**
achieves 60 % accuracy — but it has zero skill. This is why **Precision**,
**Recall**, and especially **F1** are more informative: they reveal whether the
model is actually distinguishing between up and down days.

### Multi-task metrics

When running in multi-task mode, the Evaluation tab shows **both** sets of
metrics side by side:

- **Regression metrics** (MSE, RMSE, MAE, R²) from the regression head
- **Classification metrics** (Accuracy, Precision, Recall, F1, confusion
  matrix) from the classification head
- The buy threshold used for label generation is also displayed

All keys in the metrics dictionary are prefixed (`reg_` / `cls_`) so they are
easy to distinguish programmatically.

### What do good vs bad results look like?

| Scenario | Typical metrics | Interpretation |
|----------|----------------|----------------|
| Random guessing | R² ≈ 0, Accuracy ≈ 50 % | Model has learned nothing useful |
| Slight edge | R² ≈ 0.01–0.05, Accuracy ≈ 52–55 % | Potentially useful in finance |
| Strong model (rare) | R² > 0.1, F1 > 0.65 | Unusually good; verify not overfitting |
| Overfit model | R² > 0.9 on training data | Too good to be true — check validation metrics |

---

## 9. Practical Examples & Common Scenarios

> **Note:** These examples are purely educational illustrations. They do NOT
> constitute financial advice.

### Scenario A — Positive prediction with buy signal

*"The model predicts a +1.2 % return over 5 days and the classification model
outputs a Buy signal."*

**Interpretation:** Both models agree that the price is likely to increase.
This **alignment** between regression and classification outputs increases
confidence in the direction, but it does not guarantee that the price will
actually rise.

**What to check:**
- Is the R² or F1 on the validation set reasonable?
- Is the prediction within the normal range of historical returns?

### Scenario B — Validation loss rising while training loss decreases

*"After epoch 30, the training loss keeps falling but the validation loss starts
climbing."*

**Diagnosis:** Classic **overfitting**. The model is memorising the training
data instead of learning generalisable patterns.

**What to do:**
- Stop training earlier (reduce epochs to ~30).
- Reduce model complexity: lower hidden size or number of layers.
- Increase the dataset size (use a longer date range).

### Scenario C — Predictions fluctuate heavily

*"The predicted returns jump between -5 % and +5 % every day."*

**Diagnosis:** The model may be **overfitting to noise** or the learning rate is
too high.

**What to do:**
- Reduce the learning rate (try 0.0005 or 0.0001).
- Increase the sequence length so the model has more context.
- Switch to a longer horizon (5 or 20 days) which is smoother.

### Scenario D — Model always predicts the same value

*"The predicted return is ~0.001 for every single day."*

**Diagnosis:** The model has collapsed to predicting the **mean** of the
training targets — it has not learned any useful patterns.

**What to do:**
- Increase model capacity: raise hidden size or number of layers.
- Increase epochs.
- Lower the learning rate for more stable gradient updates.
- Make sure the data has enough variability (use a longer date range).

### Scenario E — Very high accuracy on training data

*"The classification model shows 95 % accuracy."*

**Diagnosis:** Likely **overfitting** — especially if validation accuracy is
much lower (e.g., 52 %).

**What to check:**
- Compare training and validation metrics side by side.
- If they diverge significantly, reduce model complexity or training time.
- True financial forecasting accuracy above 55–60 % on unseen data is already
  very good.

---

## 10. Quick-Reference Cheat Sheet

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
| Buy Threshold | 0.003 (only for cls / multi-task) |
| w_reg / w_cls | 1.0 / 1.0 (only for multi-task) |

### Architecture comparison

| Property | GRU | LSTM | TCN |
|----------|-----|------|-----|
| Type | Recurrent | Recurrent | Convolutional |
| Speed | Fast | Moderate | Fastest |
| Long sequences | Adequate | Best | Very good |
| GPU benefit | Moderate | Moderate | Large |

### Diagnostics verdicts

| Verdict | Meaning | Suggested action |
|---------|---------|-----------------|
| ✅ Healthy | Both losses decrease | Continue or stop early |
| ⚠️ Overfitting | Val loss diverges up | ↓ Epochs, ↓ Capacity, ↑ Data |
| ⚠️ Underfitting | Both losses stay high | ↑ Capacity, ↑ Epochs |
| ⚠️ Noisy | Val loss oscillates | ↓ Learning rate, ↑ Batch size |

### Common adjustments

| Problem | Try |
|---------|-----|
| Underfitting | ↑ Hidden size, ↑ Layers, ↑ Epochs |
| Overfitting | ↓ Epochs, ↓ Hidden size, ↓ Layers, ↑ Data range |
| Unstable loss | ↓ Learning rate |
| Flat predictions | ↑ Learning rate, ↑ Hidden size |
| Noisy predictions | ↑ Sequence length, ↑ Horizon, ↓ Learning rate |
| Slow training | ↓ Hidden size, ↓ Layers, ↑ Batch size, switch to TCN |
