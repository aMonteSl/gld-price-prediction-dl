# GLD Price Prediction with Deep Learning

Deep-learning application for forecasting **GLD (Gold ETF)** price movements
using historical market data. Built with **PyTorch** and featuring
**GRU, LSTM, and TCN** architectures, the app supports **regression**,
**classification**, and **multi-task learning** at multiple time horizons
(1, 5, 20 days), with **automatic training diagnostics**.

A fully internationalised **Streamlit** GUI (English / Spanish) lets you
explore data, train models, visualise predictions, evaluate performance,
and follow a built-in tutorial — all from the browser.

---

## What's New in v2.0

| Feature | Description |
|---------|-------------|
| **TCN architecture** | Temporal Convolutional Network — causal 1-D CNN with dilated convolutions and residual connections. Trains faster than RNNs. |
| **Multi-task learning** | Shared backbone with a regression + classification head. Loss: *L = w_reg × MSE + w_cls × BCEWithLogits*. |
| **Auto diagnostics** | After training, the app analyses loss curves and reports a verdict (healthy / overfitting / underfitting / noisy) with actionable suggestions. |
| **Buy threshold** | Configurable return threshold for buy-signal labels (default 0.3 %). |
| **pytest suite** | 70+ tests covering models, trainer, evaluator, diagnostics, and feature engineering. |

---

## Features

- **Data Loading** — GLD OHLCV data via yfinance
- **Feature Engineering** — 28 technical indicators (SMA, EMA, RSI, MACD, volatility, lags, …)
- **Deep Learning Models**
  - **GRU** / **LSTM** / **TCN** backbones
  - Task modes: **regression**, **classification**, **multi-task**
  - Fully configurable hyperparameters (hidden size, layers, dropout, …)
- **Training Pipeline** — StandardScaler normalisation, 80/20 split, Adam optimiser, model checkpointing
- **Evaluation** — MSE, RMSE, MAE, R², Accuracy, Precision, Recall, F1, Confusion matrix
- **Diagnostics** — automatic loss-curve analysis with verdict & suggestions
- **Streamlit GUI** — 5 tabs (Data · Train · Predictions · Evaluation · Tutorial), i18n EN/ES

---

## Installation

```bash
git clone https://github.com/aMonteSl/gld-price-prediction-dl.git
cd gld-price-prediction-dl
pip install -r requirements.txt
```

---

## Quick Start

### Streamlit GUI

```bash
streamlit run app.py
```

1. **📊 Data** — Load GLD historical prices for a custom date range
2. **🔧 Train** — Pick architecture (GRU / LSTM / TCN), task (regression / classification / multi-task), horizon (1 / 5 / 20 days), and hyperparameters → train & see diagnostics
3. **📈 Predictions** — Visualise predicted returns, implied prices, or buy/no-buy signals
4. **📉 Evaluation** — Regression & classification metrics with confusion matrix
5. **📚 Tutorial** — Built-in guide covering architectures, parameters, and interpretation

### CLI example

```bash
python scripts/example.py
```

### Programmatic API

```python
from gldpred.data import GLDDataLoader
from gldpred.features import FeatureEngineering
from gldpred.models import TCNRegressor          # or GRUMultiTask, etc.
from gldpred.training import ModelTrainer
from gldpred.evaluation import ModelEvaluator
from gldpred.diagnostics import DiagnosticsAnalyzer

# Load & engineer features
loader = GLDDataLoader(ticker="GLD")
data = loader.load_data()
fe = FeatureEngineering()
features = fe.select_features(fe.add_technical_indicators(data)).ffill().bfill()

# Prepare targets & sequences
targets = loader.compute_returns(horizon=5)
X, y = fe.create_sequences(features, targets, seq_length=20)

# Train
model = TCNRegressor(input_size=X.shape[2], hidden_size=64, num_layers=3)
trainer = ModelTrainer(model, task="regression")
tl, vl = trainer.prepare_data(X, y)
history = trainer.train(tl, vl, epochs=50)

# Diagnostics
diag = DiagnosticsAnalyzer.analyze(history)
print(diag.verdict, diag.explanation)

# Evaluate
preds = trainer.predict(X)
print(ModelEvaluator.evaluate_regression(y, preds))

# Save
trainer.save_model("models/tcn_reg_h5.pth")
```

---

## Project Structure

```
gld-price-prediction-dl/
├── app.py                          # Streamlit entrypoint
├── requirements.txt
├── pytest.ini
├── AGENTS.md                       # AI coding-assistant guide
├── README.md
│
├── src/gldpred/                    # Main Python package
│   ├── __init__.py                 # v2.0.0
│   ├── config.py                   # DataConfig, ModelConfig, TrainingConfig
│   ├── app/
│   │   ├── streamlit_app.py        # 5-tab Streamlit GUI
│   │   └── i18n.py                 # EN/ES translations
│   ├── data/
│   │   └── loader.py               # GLDDataLoader (yfinance)
│   ├── features/
│   │   └── engineering.py          # 28 technical features
│   ├── models/
│   │   └── architectures.py        # GRU/LSTM/TCN × Reg/Cls/MultiTask (9 models)
│   ├── training/
│   │   └── trainer.py              # ModelTrainer (reg / cls / multitask)
│   ├── evaluation/
│   │   └── evaluator.py            # Regression, classification, multitask metrics
│   ├── inference/
│   │   └── predictor.py            # Predictor wrapper
│   └── diagnostics/
│       └── analyzer.py             # DiagnosticsAnalyzer + DiagnosticsResult
│
├── tests/
│   ├── conftest.py                 # Shared fixtures & seeds
│   ├── test_models.py              # 9 model architectures
│   ├── test_trainer.py             # Training / prediction / persistence
│   ├── test_evaluator.py           # Metric calculations
│   ├── test_diagnostics.py         # Loss-curve analysis
│   ├── test_features.py            # Feature engineering & sequences
│   └── test_suite.py               # Legacy test runner
│
├── scripts/
│   └── example.py                  # CLI demo
│
└── models/                         # Saved .pth files (git-ignored)
```

---

## Model Architectures

| Architecture | Type | Key Property |
|-------------|------|--------------|
| **GRU** | Recurrent | Fast, few parameters, good default |
| **LSTM** | Recurrent | Better long-range memory, more parameters |
| **TCN** | Convolutional | Causal dilated CNN, fully parallel, fastest training |

All models share the same constructor signature:
`(input_size, hidden_size=64, num_layers=2, dropout=0.2)`.

### Task Modes

| Mode | Output | Loss |
|------|--------|------|
| Regression | Continuous return | MSE |
| Classification | Buy/No-Buy probability | BCE |
| Multi-task | (return, logits) tuple | w_reg × MSE + w_cls × BCEWithLogits |

---

## Testing

```bash
# Install pytest (if needed)
pip install pytest

# Run all tests
pytest

# Verbose output
pytest -v
```

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- Streamlit ≥ 1.30
- pandas, numpy, scikit-learn, yfinance, matplotlib, plotly

See [requirements.txt](requirements.txt) for the complete list.

---

## License

MIT License

## Contributing

Contributions welcome — please open an issue or submit a Pull Request.
