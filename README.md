# Multi-Asset Price Prediction with Deep Learning

Deep-learning application for forecasting price movements of **multiple
financial assets** using historical market data. Built with **PyTorch** and
featuring **GRU, LSTM, and TCN** architectures, the system produces
**probabilistic quantile trajectory forecasts** and converts them into
actionable **time-based action plans** with per-day BUY / HOLD / SELL / AVOID
classifications, entry-window detection, optimal exit selection,
three-scenario analysis (P10/P50/P90), and multi-factor decision rationale.

Supported assets: **GLD** (Gold ETF), **SLV** (Silver ETF), **BTC-USD**
(Bitcoin), **PALL** (Palladium ETF).

A fully internationalised **Streamlit** GUI (**Spanish-first**, with English
available) lets you explore data, train models, visualise fan-chart forecasts,
get recommendations with risk metrics, compare assets side-by-side for a given
investment amount, evaluate performance, manage all persisted data in the
**Data Hub**, and follow a **guided onboarding tutorial** — all from the browser.

---

## What's New in v3.0

| Feature | Description |
|---------|-------------|
| **Multi-asset support** | Trade GLD, SLV, BTC-USD, and PALL — dynamically configurable |
| **Quantile trajectory forecasting** | Every model outputs `(batch, K, Q)` — K forecast steps × Q quantiles (P10/P50/P90) |
| **Pinball (quantile) loss** | Unified loss function replacing MSE/BCE — no regression/classification split |
| **Fan-chart visualisation** | Plotly fan charts showing P10–P90 uncertainty bands around the median forecast |
| **Model registry** | Save, load, list, and delete trained models with metadata and scaler persistence |
| **Action plan engine** | Converts trajectories into BUY / HOLD / SELL / AVOID action plans with entry-window detection, optimal exit selection, three-scenario analysis, and multi-factor decision rationale |
| **Fine-tuning** | Resume training from a saved model checkpoint |
| **30+ features** | Expanded technical indicators including ATR%, price-to-SMA ratios, and more |
| **13-tab Streamlit GUI** | Dashboard · Data · Train · Models · Forecast · Recommendation · Evaluation · Compare · Portfolio · Health · Backtest · Data Hub · Tutorial |
| **Spanish-first i18n** | App defaults to Spanish with persistent language selection; English fully supported |
| **Decision-first dashboard** | Landing page showing all assets at a glance with recommendations and leaderboard |
| **Data Hub** | Centralised view to inspect, export (CSV/JSON/ZIP), and manage all persisted application data |
| **Guided onboarding** | 8-step interactive tutorial for first-time users with Next/Back/Skip navigation |
| **Portfolio tracking** | Trade log with predicted vs actual outcomes and performance monitoring |
| **Model health monitoring** | Staleness detection, accuracy tracking, and recalibration advice |
| **Walk-forward backtesting** | Out-of-sample validation engine with summary statistics |
| **Asset model assignment** | Assign a primary model to each asset for one-click comparison |
| **Portfolio comparison** | Compare multiple assets side-by-side given an investment amount — ranked leaderboard |
| **Risk metrics** | Stop-loss, take-profit, risk-reward ratio, max drawdown, volatility regime per recommendation |
| **Action plan parameters** | User-configurable horizon, TP%, SL%, min expected return, risk-aversion λ, and investment amount in the sidebar |
| **Asset catalog** | Centralised metadata (type, currency, volatility, descriptions) for every supported ticker |
| **247 pytest tests** | Comprehensive coverage across 18 test modules including integration smoke tests |

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
- **Action Plan Engine** — BUY / HOLD / SELL / AVOID per-day classification with entry-window detection, optimal exit selection, three-scenario analysis (P10/P50/P90 with value impact), and multi-factor decision rationale
- **Portfolio Comparison** — Compare multiple assets for a given investment amount with a ranked leaderboard
- **Risk Metrics** — Stop-loss, take-profit, risk-reward ratio, max drawdown, and volatility regime per recommendation
- **Action Plan Parameters** — User-configurable horizon, TP%, SL%, min expected return, risk-aversion λ, and investment amount
- **Asset Model Assignment** — Assign a primary trained model to each asset for streamlined comparison
- **Evaluation** — Trajectory metrics (MSE, RMSE, MAE, directional accuracy) + quantile calibration
- **Model Registry** — Persistent save/load with scaler, metadata, and architecture info
- **Diagnostics** — Automatic loss-curve analysis with verdict, suggestions, and **Apply Suggestions** button that auto-tunes hyperparameters
- **Loss Chart Markers** — Best-epoch vertical line and overfitting zone shading on training plots
- **Streamlit GUI** — 13 tabs, Spanish-first i18n (EN/ES), guided onboarding, Data Hub, interactive Plotly charts

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

1. **📊 Dashboard** — Decision-first landing page: all assets at a glance, leaderboard, top recommendation
2. **📁 Data** — Load historical prices for GLD, SLV, BTC-USD, or PALL
3. **🏋️ Train** — Pick architecture (GRU / LSTM / TCN), forecast steps, quantiles, and hyperparameters → train or fine-tune
4. **🗂️ Models** — Manage saved models: rename, delete, assign primary per asset
5. **📈 Forecast** — View fan-chart trajectories with P10/P50/P90 uncertainty bands
6. **🎯 Recommendation** — Generate time-based action plans with BUY / HOLD / SELL / AVOID per-day classification, entry-window detection, scenario analysis, and interactive chart
7. **📊 Evaluation** — Trajectory metrics + quantile calibration analysis
8. **⚖️ Compare** — Compare multiple assets side-by-side for a given investment amount
9. **💼 Portfolio** — Trade log with predicted vs actual outcome tracking
10. **🩺 Health** — Model staleness, accuracy monitoring, recalibration advice
11. **🔬 Backtest** — Walk-forward out-of-sample backtesting
12. **🗄️ Data Hub** — Inspect, export (CSV/JSON/ZIP), and manage all persisted data
13. **📚 Tutorial** — Built-in guide + restart guided onboarding

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
from gldpred.decision import DecisionEngine, PortfolioComparator
from gldpred.evaluation import ModelEvaluator
from gldpred.registry import ModelRegistry, ModelAssignments
from gldpred.config import ASSET_CATALOG

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

# Get recommendation with risk metrics
engine = DecisionEngine()
rec = engine.recommend(forecast, df)
print(rec.action, rec.confidence, rec.rationale)
print(rec.risk.stop_loss_pct, rec.risk.take_profit_pct, rec.risk.volatility_regime)

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
│   │   ├── __init__.py             # DataConfig, ModelConfig, TrainingConfig,
│   │   │                           #   DecisionConfig, AppConfig, SUPPORTED_ASSETS
│   │   └── assets.py               # AssetInfo, ASSET_CATALOG (centralised metadata)
│   ├── i18n/
│   │   └── __init__.py             # STRINGS, LANGUAGES (ES / EN), DEFAULT_LANGUAGE="es"
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
│   │   ├── store.py                # ModelRegistry (save / load / list / delete)
│   │   └── assignments.py          # ModelAssignments (primary model per asset)
│   ├── decision/
│   │   ├── engine.py               # DecisionEngine, Recommendation, RiskMetrics
│   │   ├── scenario_analyzer.py    # ScenarioAnalysis, ScenarioOutcome, analyze_scenarios
│   │   ├── action_planner.py       # ActionPlan, DayRecommendation, build_action_plan
│   │   └── portfolio.py            # PortfolioComparator, AssetOutcome, ComparisonResult
│   ├── core/policy/
│   │   └── scoring.py              # DecisionPolicy — transparent scoring wrapper
│   ├── storage/
│   │   └── trade_log.py            # JSONL-based trade log persistence
│   ├── services/
│   │   ├── health_service.py       # Model health monitoring
│   │   └── backtest_engine.py      # Walk-forward backtesting engine
│   └── app/
│       ├── components/
│       │   ├── forecast_cache.py    # In-memory forecast cache with TTL
│       │   ├── empty_states.py     # Guided empty-state UI components
│       │   └── onboarding.py       # Guided 8-step onboarding tutorial
│       ├── controllers/
│       │   └── dashboard_controller.py  # Dashboard analysis engine
│       ├── ui/
│       │   ├── sidebar.py               # Sidebar with language + action plan params
│       │   ├── tabs_dashboard.py        # 📊 Dashboard (landing page)
│       │   ├── tabs_data.py             # 📁 Data loading
│       │   ├── tabs_train.py            # 🏋️ Training
│       │   ├── tabs_models.py           # 🗂️ Model management
│       │   ├── tabs_forecast.py         # 📈 Fan chart forecast
│       │   ├── tabs_recommendation.py   # 🎯 Recommendation + action plan
│       │   ├── tabs_evaluation.py       # 📊 Evaluation metrics
│       │   ├── tabs_compare.py          # ⚖️ Asset comparison
│       │   ├── tabs_portfolio.py        # 💼 Portfolio / trade log
│       │   ├── tabs_health.py           # 🩺 Model health
│       │   ├── tabs_backtest.py         # 🔬 Walk-forward backtesting
│       │   ├── tabs_datahub.py          # 🗄️ Data Hub (inspect/export/manage)
│       │   └── tabs_tutorial.py         # 📚 Tutorial + onboarding restart
│       ├── plots.py                # Fan chart & loss chart helpers
│       └── streamlit_app.py        # 13-tab Streamlit GUI (Spanish-first)
│
├── tests/
│   ├── conftest.py                 # Shared fixtures & seeds
│   ├── test_models.py              # 3 forecaster architectures (18 tests)
│   ├── test_trainer.py             # Training loop, predict, save/load (11 tests)
│   ├── test_evaluator.py           # Trajectory & quantile metrics (7 tests)
│   ├── test_integration.py         # End-to-end pipeline smoke tests (6 tests)
│   ├── test_diagnostics.py         # Loss-curve analysis (7 tests)
│   ├── test_features.py            # Feature engineering & sequences (7 tests)
│   ├── test_registry.py            # ModelRegistry persistence (13 tests)
│   ├── test_decision.py            # DecisionEngine recommendations (9 tests)
│   ├── test_trade_plan.py          # Action plan & scenario analysis (40 tests)
│   ├── test_catalog.py             # Asset catalog & metadata (8 tests)
│   ├── test_assignments.py         # Model assignments persistence (9 tests)
│   ├── test_portfolio.py           # Portfolio comparison & risk metrics (21 tests)
│   ├── test_model_bundle.py        # ModelBundle predict, load_bundle (11 tests)
│   ├── test_forecast_cache.py      # ForecastCache TTL, invalidation (13 tests)
│   ├── test_decision_policy.py     # DecisionPolicy scoring factors (13 tests)
│   ├── test_trade_log.py           # TradeLogStore JSONL persistence (13 tests)
│   ├── test_health_service.py      # HealthService staleness, accuracy (23 tests)
│   └── test_backtest_engine.py     # BacktestEngine walk-forward (17 tests)
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

## Action Plan Engine

The **Recommendation tab** uses `build_action_plan()` to convert a quantile
trajectory forecast into a **time-based action plan** — a per-day action
schedule with entry-window detection, optimal exit selection, scenario analysis,
and multi-factor decision rationale.

### Per-Day Classification

Each day is classified using a **state-machine** approach
(NO_POSITION → BUY → POSITIONED → SELL → CLOSED):

| Action | Meaning |
|--------|---------|
| **BUY** | Favourable entry point — positive outlook with acceptable risk |
| **HOLD** | Maintain position — within risk limits |
| **SELL** | Exit position — take-profit, stop-loss, or declining momentum |
| **AVOID** | Stay away — negative outlook, excessive risk, or position already closed |

### Entry & Exit Optimisation

| Output | Description |
|--------|-------------|
| **Entry window** | Best contiguous range of BUY-classified days (ranked by avg risk-adjusted score) |
| **Best exit day** | `argmax[r_50(t) − λ · max(0, −r_10(t))]` — peak risk-adjusted return |

### Scenario Analysis

Three scenarios are computed from the quantile forecast:

| Scenario | Quantile | Shows |
|----------|----------|-------|
| **Optimistic** | P90 | Best-case return, final price, P&L on investment |
| **Base** | P50 | Median projection |
| **Pessimistic** | P10 | Worst-case drawdown |

### Decision Rationale

Four factors explain the recommendation:

1. **Trend confirmation** — SMA-50 vs SMA-200 (golden/death cross)
2. **Volatility regime** — ATR% classification (low / normal / high)
3. **Quantile risk** — P10 drawdown vs stop-loss threshold
4. **Today's assessment** — Whether today falls within the optimal entry window

### Sidebar Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Horizon** | 10 days | Number of forecast days in the plan (1–60) |
| **Take-Profit %** | 5.0 | Cumulative P50 return that triggers SELL |
| **Stop-Loss %** | 3.0 | P10 drawdown that triggers SELL |
| **Min Expected Return %** | 1.0 | Minimum median return for favourable outlook |
| **Risk Aversion (λ)** | 0.5 | Penalty weight on downside risk in scoring |
| **Investment ($)** | 10 000 | Hypothetical amount for scenario value-impact |

### Persistence

Every generated plan is saved as a JSON file in `data/trade_plans/` with
the format `plan_{ASSET}_{timestamp}.json`, enabling post-session review.

### Portfolio Comparison

The `PortfolioComparator` enables side-by-side comparison of multiple assets:

1. Assign a primary model to each asset in the Train tab
2. Set an investment amount and horizon in the Compare tab
3. The engine forecasts each asset, ranks them by median PnL, and presents a leaderboard

Each `AssetOutcome` includes projected values at P10/P50/P90, PnL, and a full recommendation.
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
# Run all 247 tests
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
