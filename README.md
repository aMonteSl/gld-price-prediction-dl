# Multi-Asset Price Prediction with Deep Learning

Deep-learning application for forecasting price movements of **multiple
financial assets** using historical market data. Built with **PyTorch** and
featuring **GRU, LSTM, and TCN** architectures, the system produces
**probabilistic quantile trajectory forecasts** and converts them into
actionable **BUY / HOLD / AVOID** recommendations.

Supported assets: **GLD** (Gold ETF), **SLV** (Silver ETF), **BTC-USD**
(Bitcoin), **PALL** (Palladium ETF).

A fully internationalised **Streamlit** GUI (English / Spanish) lets you
explore data, train models, visualise fan-chart forecasts, get recommendations,
evaluate performance, and follow a built-in tutorial — all from the browser.

---

## What's New in v3.0

| Feature | Description |
|---------|-------------|
| **Multi-asset support** | Trade GLD, SLV, BTC-USD, and PALL — dynamically configurable |
| **Quantile trajectory forecasting** | Every model outputs `(batch, K, Q)` — K forecast steps × Q quantiles (P10/P50/P90) |
| **Pinball (quantile) loss** | Unified loss function replacing MSE/BCE — no regression/classification split |
| **Fan-chart visualisation** | Plotly fan charts showing P10–P90 uncertainty bands around the median forecast |
| **Model registry** | Save, load, list, and delete trained models with metadata and scaler persistence |
| **Decision engine** | Converts trajectories into BUY / HOLD / AVOID recommendations with confidence scores |
| **Fine-tuning** | Resume training from a saved model checkpoint |
| **30+ features** | Expanded technical indicators including ATR%, price-to-SMA ratios, and more |
| **6-tab Streamlit GUI** | Data · Train · Forecast · Recommendation · Evaluation · Tutorial |
| **78 pytest tests** | Comprehensive coverage across 8 test modules including integration smoke tests |

---

## Features

- **Data Loading** — Full historical data from asset inception to today for any supported ticker via yfinance
- **Feature Engineering** — 30+ technical indicators (SMA, EMA, RSI, MACD, ATR, Bollinger, lags, …)
- **Deep Learning Models**
  - **GRU** / **LSTM** / **TCN** (default) backbones
  - Unified quantile output: `(batch, forecast_steps, num_quantiles)`
  - Configurable forecast horizon and quantile levels
- **Training Pipeline** — StandardScaler, temporal 80/20 split, Adam, pinball loss, fine-tuning
- **Inference** — Multi-step price trajectories with uncertainty bands
- **Decision Engine** — BUY / HOLD / AVOID recommendations with confidence scoring
- **Evaluation** — Trajectory metrics (MSE, RMSE, MAE, directional accuracy) + quantile calibration
- **Model Registry** — Persistent save/load with scaler, metadata, and architecture info
- **Diagnostics** — Automatic loss-curve analysis with verdict, suggestions, and **Apply Suggestions** button that auto-tunes hyperparameters
- **Loss Chart Markers** — Best-epoch vertical line and overfitting zone shading on training plots
- **Streamlit GUI** — 6 tabs, i18n EN/ES, interactive Plotly charts

---

## Installation

```bash
git clone https://github.com/aMonteSl/gld-price-prediction-dl.git
cd gld-price-prediction-dl
pip install -e .
```

Or without editable mode:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Streamlit GUI

```bash
streamlit run app.py
```

1. **📊 Data** — Load historical prices for GLD, SLV, BTC-USD, or PALL
2. **🔧 Train** — Pick architecture (GRU / LSTM / TCN), forecast steps, quantiles, and hyperparameters → train or fine-tune
3. **🔮 Forecast** — View fan-chart trajectories with P10/P50/P90 uncertainty bands
4. **💡 Recommendation** — Get BUY / HOLD / AVOID decisions with confidence scores
5. **📉 Evaluation** — Trajectory metrics + quantile calibration analysis
6. **📚 Tutorial** — Built-in guide covering architectures, forecasting, and interpretation

### CLI example

```bash
python scripts/example.py
```

### Programmatic API

```python
from gldpred.data import AssetDataLoader
from gldpred.features import FeatureEngineering
from gldpred.models import TCNForecaster
from gldpred.training import ModelTrainer
from gldpred.inference import TrajectoryPredictor
from gldpred.decision import DecisionEngine
from gldpred.evaluation import ModelEvaluator

# Load & engineer features
loader = AssetDataLoader(ticker="GLD")
data = loader.load_data()
fe = FeatureEngineering()
df = fe.add_technical_indicators(data)
feat_df = fe.select_features(df)
feature_names = feat_df.columns.tolist()

# Create sequences (multi-step targets)
returns = loader.daily_returns()
X, y = fe.create_sequences(feat_df.values, returns.values,
                           seq_length=20, forecast_steps=20)

# Train with pinball loss
model = TCNForecaster(input_size=X.shape[2])
trainer = ModelTrainer(model)
train_loader, val_loader = trainer.prepare_data(X, y)
history = trainer.train(train_loader, val_loader, epochs=50)

# Forecast trajectories
predictor = TrajectoryPredictor(model, trainer.scaler, feature_names)
forecast = predictor.predict_trajectory(df, feature_names,
                                         seq_length=20, asset="GLD")
print(forecast.dates, forecast.p50)  # median trajectory

# Get recommendation
engine = DecisionEngine()
rec = engine.recommend(forecast, df)
print(rec.action, rec.confidence, rec.rationale)

# Evaluate
preds = trainer.predict(X)
metrics = ModelEvaluator.evaluate_trajectory(y, preds[:, :, 1])  # median
print(metrics)
```

---

## Project Structure

```
gld-price-prediction-dl/
├── app.py                          # Streamlit entrypoint
├── requirements.txt                # pip dependencies
├── pyproject.toml                  # Build configuration
├── pytest.ini                      # pytest configuration
├── AGENTS.md                       # AI coding-assistant guide
├── README.md                       # ← You are here
├── USER_GUIDE.md                   # Comprehensive user guide
│
├── src/gldpred/                    # Main Python package
│   ├── __init__.py                 # v3.0.0
│   ├── config/
│   │   └── __init__.py             # DataConfig, ModelConfig, TrainingConfig,
│   │                               #   DecisionConfig, AppConfig, SUPPORTED_ASSETS
│   ├── i18n/
│   │   └── __init__.py             # STRINGS, LANGUAGES (EN / ES)
│   ├── data/
│   │   └── loader.py               # AssetDataLoader (yfinance)
│   ├── features/
│   │   └── engineering.py          # 30+ technical features, multi-step sequences
│   ├── models/
│   │   └── architectures.py        # GRUForecaster, LSTMForecaster, TCNForecaster
│   ├── training/
│   │   └── trainer.py              # ModelTrainer, pinball_loss
│   ├── evaluation/
│   │   └── evaluator.py            # Trajectory & quantile metrics
│   ├── diagnostics/
│   │   └── analyzer.py             # DiagnosticsAnalyzer + DiagnosticsResult
│   ├── inference/
│   │   └── predictor.py            # TrajectoryPredictor, TrajectoryForecast
│   ├── registry/
│   │   └── store.py                # ModelRegistry (save / load / list / delete)
│   ├── decision/
│   │   └── engine.py               # DecisionEngine, Recommendation
│   └── app/
│       ├── plots.py                # Fan chart & loss chart helpers
│       └── streamlit_app.py        # 6-tab Streamlit GUI
│
├── tests/
│   ├── conftest.py                 # Shared fixtures & seeds
│   ├── test_models.py              # 3 forecaster architectures (18 tests)
│   ├── test_trainer.py             # Training loop, predict, save/load (11 tests)
│   ├── test_evaluator.py           # Trajectory & quantile metrics (7 tests)
│   ├── test_integration.py         # End-to-end pipeline smoke tests (6 tests)
│   ├── test_diagnostics.py         # Loss-curve analysis (7 tests)
│   ├── test_features.py            # Feature engineering & sequences (7 tests)
│   ├── test_registry.py            # ModelRegistry persistence (5 tests)
│   └── test_decision.py            # DecisionEngine recommendations (9 tests)
│
├── scripts/
│   └── example.py                  # CLI demo
│
└── data/model_registry/            # Saved model artifacts (git-ignored)
```

---

## Model Architectures

| Architecture | Class | Type | Key Property |
|-------------|-------|------|--------------|
| **GRU** | `GRUForecaster` | Recurrent | Fast, few parameters |
| **LSTM** | `LSTMForecaster` | Recurrent | Better long-range memory |
| **TCN** | `TCNForecaster` | Convolutional | Causal dilated CNN, fastest training **(default)** |

All models share the same constructor signature:

```python
Model(input_size, hidden_size=64, num_layers=2, dropout=0.2,
      forecast_steps=20, quantiles=(0.1, 0.5, 0.9))
```

**Unified output:** `forward()` returns a tensor of shape `(batch, K, Q)` where
`K = forecast_steps` and `Q = len(quantiles)`.

**Loss function:** Pinball (quantile) loss — there is no separate regression /
classification / multi-task split.

---

## Decision Engine

The `DecisionEngine` converts forecast trajectories into actionable signals:

| Signal | Meaning |
|--------|---------|
| **BUY** | Strong upward expected return with acceptable uncertainty |
| **HOLD** | Mixed signals or insufficient edge |
| **AVOID** | Negative expected return or excessive volatility |

Each recommendation includes:
- **Confidence score** (0–100)
- **Rationale** — human-readable explanation
- **Warnings** — risk factors to consider

---

## Model Registry

Trained models are saved to the **model registry** (`data/model_registry/`), a persistent storage system that preserves:
- Model weights (PyTorch state dict)
- Scaler (fitted StandardScaler)
- Metadata (architecture, asset, training date, validation loss, feature names, etc.)

### Custom Model Names

When training a model, you can provide a **custom label** (max 60 chars) to make it memorable:

```python
model_id = registry.save_model(
    model, scaler, config, feature_names, training_summary,
    label="GLD_TCN_multistep_K20_v1"
)
```

If omitted, the label auto-generates from asset and architecture (e.g., `GLD_TCN_20261208_143052`).

**In the Streamlit app:** The Train tab has a text input where you can enter a custom name before training.

### Listing & Loading Models

```python
from gldpred.registry import ModelRegistry
from gldpred.models import TCNForecaster

registry = ModelRegistry()

# List all models
models = registry.list_models()
for m in models:
    print(m["label"], m["asset"], m["architecture"], m["created_at"])

# List filtered by asset/architecture
gld_models = registry.list_models(asset="GLD", architecture="TCN")

# Load a model
model, scaler, metadata = registry.load_model(
    model_id="20261208_143052_abc123ef",
    model_class=TCNForecaster,
    input_size=30
)
```

### Deleting Models

**Single model:**
```python
registry.delete_model(model_id="20261208_143052_abc123ef")
```

**Bulk deletion:**
```python
# Delete all models for a specific asset
registry.delete_all_models(asset="GLD", confirmed=True)

# Delete ALL models (use with caution)
registry.delete_all_models(confirmed=True)
```

**In the Streamlit app:** The Train tab has an expandable "Delete Models" section with dropdowns and confirmation inputs. You must type `DELETE` or `DELETE ALL` exactly to confirm.

### Model ID vs Label

- **Model ID** — Auto-generated unique identifier (filesystem-safe, e.g., `20261208_143052_abc123ef`)
- **Label** — Custom human-readable name (e.g., `MyAwesomeModel_v1`)

The label is stored in metadata and displayed throughout the UI, making it easy to identify models. The model ID is used internally for file storage and API calls.

---

## Testing

```bash
# Run all 78 tests
pytest

# Verbose output
pytest -v

# Single module
pytest tests/test_models.py
```

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- Streamlit ≥ 1.30
- pandas, numpy, scikit-learn, yfinance, matplotlib, plotly, joblib

See [requirements.txt](requirements.txt) for the complete list.

---

## License

MIT License

## Contributing

Contributions welcome — please open an issue or submit a Pull Request.
