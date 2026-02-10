# AGENTS.md — AI-Assisted Development Guide

> This file helps GitHub Copilot, ChatGPT, Claude, and other AI coding
> assistants understand the project structure, conventions, and constraints
> so they can generate accurate, consistent code.

---

## 1. Project Overview

**Multi-Asset Price Prediction** is a deep-learning application that
downloads historical price data for multiple financial assets, engineers
technical features, and trains **GRU / LSTM / TCN** models (PyTorch) to
produce **probabilistic quantile trajectory forecasts**. Supported assets
(17 tickers across 3 risk tiers) are defined in the `SUPPORTED_ASSETS`
constant and the `ASSET_CATALOG` metadata registry.  **SPY** (S&P 500
ETF) serves as the mandatory benchmark baseline.

The system uses a **unified quantile forecasting** approach — there is no
separate regression, classification, or multi-task split. Every model
outputs a tensor of shape `(batch, K, Q)` where `K` is the number of
forecast steps (default 20) and `Q` is the number of quantiles (default
`(0.1, 0.5, 0.9)`). Training is driven by **pinball (quantile) loss**.

A **decision engine** converts forecast trajectories into actionable
**BUY / HOLD / AVOID** recommendations with confidence scores, **risk
metrics** (stop-loss, take-profit, risk-reward ratio, max drawdown),
**market regime detection**, and **asset-class-aware risk modifiers**
that adjust scoring based on each asset's risk level, volatility
profile, and role.

A **portfolio comparison engine** ranks multiple assets side-by-side for a
given investment amount, producing a leaderboard of expected outcomes
with the **S&P 500 (SPY) as benchmark** reference.

A **Streamlit** GUI provides interactive data exploration (auto-loaded),
training, **model management** (rename, delete, assign primary),
forecasting, recommendation, evaluation, asset comparison, **portfolio
tracking** (trade log with predicted vs actual outcomes), **model health
monitoring** (staleness, accuracy, recalibration advice), **walk-forward
backtesting**, a centralised **Data Hub** for inspecting, exporting, and
managing all persisted data, and a built-in tutorial with **guided
onboarding** for first-time users — all fully internationalised in
**Spanish (default) and English**. An **educational glossary** with 25
bilingual terms provides context-sensitive help via popover components
throughout the interface.

A **decision-first dashboard** serves as the landing page, showing all
assets at a glance with recommendations, leaderboard, and entry/exit
timing — answering *"Should I invest today?"* in under 30 seconds.

---

## 2. Repository Layout

```
gld-price-prediction-dl/
├── app.py                        # Thin entrypoint — run `streamlit run app.py`
├── requirements.txt              # pip dependencies
├── pyproject.toml                # Build configuration
├── pytest.ini                    # pytest configuration
├── README.md                     # User-facing README
├── USER_GUIDE.md                 # Comprehensive tutorial / user guide
├── AGENTS.md                     # ← You are here
├── MEJORAS.md                    # UX strategy & 7-phase engineering plan
│
├── src/gldpred/                  # Main Python package
│   ├── __init__.py               # Package root (version, public API) — v3.0.0
│   │
│   ├── config/
│   │   ├── __init__.py           # DataConfig, ModelConfig, TrainingConfig,
│   │   │                         #   DecisionConfig, AppConfig, SUPPORTED_ASSETS
│   │   └── assets.py             # AssetInfo, ASSET_CATALOG (centralised metadata)
│   │
│   ├── i18n/
│   │   └── __init__.py           # STRINGS, LANGUAGES (EN / ES) — 500+ keys
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py             # AssetDataLoader (yfinance, any supported ticker)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py        # FeatureEngineering (30+ features, multi-step seqs)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── architectures.py      # GRUForecaster, LSTMForecaster, TCNForecaster
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py            # ModelTrainer, pinball_loss
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py          # ModelEvaluator (trajectory + quantile metrics)
│   │
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   └── analyzer.py           # DiagnosticsAnalyzer (loss-curve analysis)
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py          # TrajectoryPredictor, TrajectoryForecast
│   │
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── store.py              # ModelRegistry, ModelBundle (save / load / list / delete / rename / load_bundle)
│   │   └── assignments.py        # ModelAssignments (primary model per asset)
│   │
│   ├── decision/
│   │   ├── __init__.py
│   │   ├── engine.py             # DecisionEngine, Recommendation, RiskMetrics,
│   │   │                         #   RecommendationHistory (BUY/HOLD/AVOID)
│   │   ├── scenario_analyzer.py  # ScenarioAnalysis, ScenarioOutcome, analyze_scenarios
│   │   ├── action_planner.py     # ActionPlan, DayRecommendation, EntryWindow,
│   │   │                         #   ExitPoint, DecisionRationale, build_action_plan
│   │   └── portfolio.py          # PortfolioComparator, AssetOutcome, ComparisonResult
│   │
│   ├── core/
│   │   └── policy/
│   │       ├── __init__.py       # Exports DecisionPolicy, PolicyResult, ScoreFactor
│   │       └── scoring.py        # DecisionPolicy — transparent scoring wrapper
│   │
│   ├── storage/
│   │   ├── __init__.py           # Exports TradeLogEntry, TradeLogStore
│   │   └── trade_log.py          # JSONL-based trade log persistence
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── health_service.py     # HealthService, ModelHealthReport, staleness_verdict
│   │   └── backtest_engine.py    # BacktestEngine, BacktestResult, BacktestSummary
│   │
│   └── app/
│       ├── __init__.py
│       ├── state.py              # Centralised session-state keys & helpers
│       ├── data_controller.py    # Cached data loading (@st.cache_data)
│       ├── glossary.py           # Educational glossary + info_term() popover
│       ├── compare_controller.py # Compare-tab orchestration
│       ├── plots.py              # Fan chart & loss chart plot helpers
│       ├── streamlit_app.py      # 13-tab Streamlit GUI (Spanish-first)
│       ├── components/
│       │   ├── __init__.py       # ForecastCache, empty_states
│       │   ├── forecast_cache.py # In-memory forecast cache with TTL + data-hash
│       │   ├── empty_states.py   # Guided empty-state UI components
│       │   └── onboarding.py     # Guided 8-step onboarding tutorial
│       ├── controllers/
│       │   └── dashboard_controller.py  # Dashboard analysis engine
│       └── ui/
│           ├── tabs_dashboard.py     # 📊 Dashboard (landing page)
│           ├── tabs_data.py          # 📁 Data loading
│           ├── tabs_train.py         # 🏋️ Training
│           ├── tabs_models.py        # 🗂️ Model management
│           ├── tabs_forecast.py      # 📈 Fan chart forecast
│           ├── tabs_recommendation.py # 🎯 Recommendation + action plan
│           ├── tabs_evaluation.py    # 📊 Evaluation metrics
│           ├── tabs_compare.py       # ⚖️ Asset comparison + scatter
│           ├── tabs_portfolio.py     # 💼 Portfolio / trade log
│           ├── tabs_health.py        # 🩺 Model health monitoring
│           ├── tabs_backtest.py      # 🔬 Walk-forward backtesting
│           ├── tabs_datahub.py       # 🗄️ Data Hub (inspect/export/manage)
│           └── tabs_tutorial.py      # 📚 Tutorial + onboarding restart
│
├── scripts/
│   └── example.py                # CLI example script
│
├── tests/
│   ├── conftest.py               # Shared fixtures & seeds
│   ├── test_models.py            # 3 forecaster architectures
│   ├── test_trainer.py           # Training loop, predict, save/load
│   ├── test_evaluator.py         # Trajectory & quantile metrics
│   ├── test_diagnostics.py       # Loss-curve analysis
│   ├── test_features.py         # Feature engineering & sequences
│   ├── test_registry.py          # ModelRegistry persistence
│   ├── test_decision.py          # DecisionEngine & Recommendation
│   ├── test_catalog.py           # Asset catalog & metadata
│   ├── test_assignments.py       # Model assignments persistence
│   ├── test_portfolio.py         # Portfolio comparison & risk metrics
│   ├── test_trade_plan.py        # Action plan & scenario analysis (40 tests)
│   ├── test_model_bundle.py      # ModelBundle predict, load_bundle roundtrip
│   ├── test_integration.py       # End-to-end pipeline smoke tests
│   ├── test_forecast_cache.py    # ForecastCache TTL, invalidation, hash
│   ├── test_decision_policy.py   # DecisionPolicy scoring factors
│   ├── test_trade_log.py         # TradeLogStore JSONL persistence
│   ├── test_health_service.py    # HealthService staleness, accuracy, recommendations
│   └── test_backtest_engine.py   # BacktestEngine walk-forward, summary stats
│
└── data/model_registry/          # Saved model artifacts (git-ignored)
```

---

## 3. Key Modules & Public API

| Module | Key Export | Purpose |
|--------|-----------|---------|
| `gldpred.config` | `DataConfig`, `ModelConfig`, `TrainingConfig`, `DecisionConfig`, `AppConfig`, `SUPPORTED_ASSETS`, `ASSET_CATALOG`, `AssetInfo`, `BENCHMARK_ASSET`, `RISK_LEVELS`, `INVESTMENT_HORIZONS`, `VOLATILITY_PROFILES`, `ASSET_ROLES`, `ASSET_CATEGORIES`, `assets_by_risk`, `assets_by_category`, `assets_by_role`, `get_benchmark` | Typed configuration via `@dataclass` + asset metadata catalog with 3-tier risk classification |
| `gldpred.data` | `AssetDataLoader` | Download & cache OHLCV data from asset inception to today for any supported ticker via yfinance |
| `gldpred.features` | `FeatureEngineering` | Compute 30+ technical features (SMA, EMA, RSI, MACD, …); build multi-step sequences |
| `gldpred.models` | `GRUForecaster`, `LSTMForecaster`, `TCNForecaster` | PyTorch `nn.Module` subclasses for quantile trajectory forecasting |
| `gldpred.training` | `ModelTrainer`, `pinball_loss` | Train/val loop with pinball loss, StandardScaler, temporal split |
| `gldpred.evaluation` | `ModelEvaluator` | Trajectory metrics (`evaluate_trajectory`) and quantile calibration (`evaluate_quantiles`) |
| `gldpred.inference` | `TrajectoryPredictor`, `TrajectoryForecast` | Price-path reconstruction from return forecasts + signal generation |
| `gldpred.diagnostics` | `DiagnosticsAnalyzer` | Heuristic loss-curve analysis (verdict + suggestions) |
| `gldpred.registry` | `ModelRegistry`, `ModelBundle`, `ModelAssignments` | Persist, load, list, delete, and rename trained model artifacts; `load_bundle()` for registry-backed inference; assign primary model per asset |
| `gldpred.decision` | `DecisionEngine`, `Recommendation`, `RiskMetrics`, `RecommendationHistory`, `PortfolioComparator`, `ActionPlan`, `DayRecommendation`, `EntryWindow`, `ExitPoint`, `DecisionRationale`, `ScenarioAnalysis`, `ScenarioOutcome`, `build_action_plan`, `summarize_action_plan`, `analyze_scenarios` | Convert forecast trajectories into BUY / HOLD / AVOID with confidence, risk metrics, regime detection, **asset-class-aware risk modifiers**; portfolio comparison with **SPY benchmark**; time-based action plans with BUY / HOLD / SELL / AVOID per-day classification, entry-window detection, scenario analysis |
| `gldpred.core.policy` | `DecisionPolicy`, `PolicyResult`, `ScoreFactor` | Transparent scoring wrapper around DecisionEngine — decomposes recommendation into labelled, bilingual factors with sentiments |
| `gldpred.storage` | `TradeLogEntry`, `TradeLogStore` | JSONL-based trade log persistence — append, load, close trades, summary stats |
| `gldpred.services` | `HealthService`, `ModelHealthReport`, `BacktestEngine`, `BacktestResult`, `BacktestSummary` | Model health monitoring (staleness, accuracy, recommendations); walk-forward backtesting engine |
| `gldpred.i18n` | `STRINGS`, `LANGUAGES`, `DEFAULT_LANGUAGE` | Dictionary-based i18n (Spanish default / English) — 1900+ keys |
| `gldpred.app.state` | `init_state`, `get`, `put`, `clear_training_state`, `clear_data_state`, `KEY_*` | Centralised session-state keys, defaults, and helpers |
| `gldpred.app.data_controller` | `LoadedData`, `fetch_asset_data`, `invalidate_cache` | Cached data loading via `@st.cache_data` (1-hour TTL) |
| `gldpred.app.glossary` | `GlossaryEntry`, `GLOSSARY`, `info_term` | Educational glossary with 25 bilingual terms + popover component |
| `gldpred.app.compare_controller` | `CompareRow`, `run_comparison`, `available_models_for_asset` | Compare-tab orchestration: per-row asset+model selection, comparison pipeline |
| `gldpred.app.components` | `ForecastCache`, `show_empty_no_data`, `show_empty_no_model`, `show_empty_no_forecast`, `should_show_onboarding`, `show_onboarding`, `restart_onboarding` | In-memory forecast cache + guided empty-state UI components + onboarding |
| `gldpred.app.controllers` | `DashboardAssetResult`, `DashboardResult`, `run_dashboard_analysis` | Dashboard analysis engine — iterates assets, loads models, runs forecasts |
| `gldpred.app.streamlit_app` | *(script)* | Streamlit application with 13 tabs (Spanish-first) |
| `gldpred.app.plots` | `create_loss_chart`, `create_fan_chart` | Plotly chart helpers (loss chart with best-epoch markers, fan chart) |

### Model classes (all in `gldpred.models`)

| Architecture | Class | Default |
|-------------|-------|---------|
| GRU | `GRUForecaster` | |
| LSTM | `LSTMForecaster` | |
| TCN | `TCNForecaster` | ✔ (default) |

All three share the same constructor signature and output contract:

- **Constructor:** `(input_size, hidden_size=64, num_layers=2, dropout=0.2, forecast_steps=20, quantiles=(0.1, 0.5, 0.9))`
- **`forward()` → `Tensor` of shape `(batch, K, Q)`** where `K = forecast_steps` and `Q = len(quantiles)`.

---

## 4. Technology Stack

| Technology | Version | Role |
|------------|---------|------|
| Python | 3.10+ | Language |
| PyTorch | ≥ 2.0 | Deep learning framework |
| Streamlit | ≥ 1.30 | Web GUI |
| yfinance | ≥ 0.2 | Market data download |
| scikit-learn | ≥ 1.3 | Preprocessing (StandardScaler), metrics |
| pandas / numpy | ≥ 2.0 / ≥ 1.24 | Data manipulation |
| matplotlib / plotly | ≥ 3.7 / ≥ 5.18 | Training plots / interactive charts |
| joblib | ≥ 1.3 | Scaler serialisation in model artifacts |

---

## 5. Coding Conventions

### 5.1 Style
- **PEP 8** with 88-char line length (Black-compatible).
- Docstrings: Google style (`"""One-liner."""` or multi-line with Args/Returns).
- Type hints encouraged, especially for public method signatures.
- Imports: stdlib → third-party → local, separated by blank lines.

### 5.2 Package structure
- Each domain has its own sub-package under `src/gldpred/`.
- `__init__.py` re-exports the public API so users can write
  `from gldpred.data import AssetDataLoader`.
- `config` and `i18n` are standalone packages (not nested under `app`).
- Private helpers start with `_`.

### 5.3 PyTorch models
- All models subclass `nn.Module`.
- Constructor signature:
  `(input_size, hidden_size=64, num_layers=2, dropout=0.2, forecast_steps=20, quantiles=(0.1, 0.5, 0.9))`.
- **Unified output:** `forward()` returns a tensor of shape
  `(batch, K, Q)` where `K = forecast_steps` and `Q = len(quantiles)`.
  There is no separate regression / classification / multi-task split.
- **Loss function:** pinball (quantile) loss, implemented as
  `pinball_loss()` in `training/trainer.py`.
- TCN models use causal dilated 1-D convolutions with exponential dilation
  and residual connections (`_CausalConv1dBlock`, `_TCNBackbone`).
- TCN is the **default architecture** in the Streamlit app.

### 5.4 Features & sequences
- `FeatureEngineering.select_features()` returns a **DataFrame** of
  selected feature columns.
- `FeatureEngineering.create_sequences()` returns targets of shape
  `(N, K)` — a multi-step forecast target — not `(N,)`.

### 5.5 i18n
- Every user-facing string lives in `src/gldpred/i18n/__init__.py` under
  `STRINGS["en"]` and `STRINGS["es"]`.
- In the Streamlit app, call `t = _t()` then use `t["key_name"]`.
- Key naming convention: `<section>_<purpose>`, e.g. `train_header`,
  `forecast_fan_chart`, `rec_confidence`, `tut_s3_body`.
- When adding new UI text, add the key to **both** `"en"` and `"es"`.

---

## 6. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Run tests (pytest — 269 tests across 18 files)
pytest
pytest -v                       # verbose
pytest tests/test_models.py     # single module

# Run the CLI example
python scripts/example.py
```

---

## 7. Critical Constraints

> **DO NOT** change ML logic, training behaviour, or model outputs.

These are strictly off-limits unless explicitly requested:

- Pinball loss function and optimiser configuration in `ModelTrainer`.
- Feature engineering formulas in `FeatureEngineering`.
- Model architectures (layer structure, activation functions, output shape)
  in `architectures.py`.
- Evaluation metric calculations in `ModelEvaluator`.
- Sequence creation and temporal train/val splitting logic.
- Decision engine thresholds and recommendation logic in `DecisionEngine`.

**Safe to change:**
- UI layout, styling, and new Streamlit widgets in `streamlit_app.py`.
- i18n strings in `i18n/__init__.py` (add/fix translations).
- Documentation files (`README.md`, `USER_GUIDE.md`, `AGENTS.md`).
- Test coverage in `tests/`.
- Configuration defaults in `config/__init__.py`.
- `SUPPORTED_ASSETS` list (to add new tickers).

---

## 8. Extension Guidelines

### Adding a new model architecture
1. Create the `nn.Module` sub-class in `models/architectures.py`.
   It must accept `forecast_steps` and `quantiles` in its constructor
   and return `(batch, K, Q)` from `forward()`.
2. Export it from `models/__init__.py`.
3. Add the architecture option to the sidebar selectbox in
   `streamlit_app.py`.
4. Add the corresponding i18n keys in `i18n/__init__.py` (both EN and ES).
5. Add parametric test entries in `tests/test_models.py`.

### Adding a new supported asset
1. Add the ticker string to the `SUPPORTED_ASSETS` tuple in
   `config/__init__.py`.
2. Add an `AssetInfo` entry to `ASSET_CATALOG` in `config/assets.py`
   (with descriptions in EN and ES).
3. No further code changes needed — `AssetDataLoader` and the Streamlit
   sidebar dynamically read from `SUPPORTED_ASSETS`.

### Adding a new language
1. Choose an ISO 639-1 code (e.g. `"fr"`).
2. Add `"Français": "fr"` to the `LANGUAGES` dict in `i18n/__init__.py`.
3. Add `"fr": { ... }` to `STRINGS` with all the same keys as `"en"`.
4. No changes needed in `streamlit_app.py` — the language selector picks
   up new entries automatically.

### Adding a new evaluation metric
1. Implement it in `evaluation/evaluator.py` inside `evaluate_trajectory`
   or `evaluate_quantiles`.
2. Add the metric name to the returned dict.
3. Display it in the Evaluation tab of `streamlit_app.py` (use i18n keys).

### Adding a new feature
1. Add the computation in `features/engineering.py`.
2. The feature is automatically included because `ModelTrainer` selects
   all numeric columns from the engineered DataFrame.

### Adding a new decision rule
1. Extend or modify `DecisionEngine` in `decision/engine.py`.
2. If new recommendation types are needed, update the `Recommendation`
   dataclass.
3. Add corresponding i18n keys and update the Recommendation tab in
   `streamlit_app.py`.

---

## 9. Testing

Run the test suite with:

```bash
# pytest (269 tests across 18 files)
pytest
pytest -v                       # verbose
pytest tests/test_models.py     # single module
```

### Test layout

| File | Covers |
|------|--------|
| `tests/conftest.py` | Shared fixtures: seeds, synthetic data, OHLCV |
| `tests/test_models.py` | 3 forecaster architectures (output shape, gradient flow) |
| `tests/test_trainer.py` | Data preparation, training loop, predict, save/load |
| `tests/test_evaluator.py` | Trajectory & quantile calibration metrics |
| `tests/test_diagnostics.py` | Loss-curve verdicts (healthy, overfit, underfit, noisy) |
| `tests/test_features.py` | Feature engineering, multi-step sequences |
| `tests/test_registry.py` | ModelRegistry save, load, list, delete |
| `tests/test_decision.py` | DecisionEngine recommendations, confidence, asset-class modifier |
| `tests/test_catalog.py` | Asset catalog metadata, AssetInfo, ASSET_CATALOG, risk tiers, benchmark, classification helpers |
| `tests/test_assignments.py` | ModelAssignments persistence, assign/unassign |
| `tests/test_portfolio.py` | RiskMetrics, regime detection, PortfolioComparator, benchmark tagging |
| `tests/test_trade_plan.py` | Action plan & scenario analysis: day classification, entry window, exit, scenarios, narrative, edge cases |
| `tests/test_model_bundle.py` | ModelBundle predict, load_bundle roundtrip, weight preservation |
| `tests/test_integration.py` | End-to-end pipeline smoke tests (all architectures) |
| `tests/test_forecast_cache.py` | ForecastCache TTL, invalidation, hash |
| `tests/test_decision_policy.py` | DecisionPolicy scoring factors |
| `tests/test_trade_log.py` | TradeLogStore JSONL persistence |
| `tests/test_health_service.py` | HealthService staleness, accuracy, recommendations |
| `tests/test_backtest_engine.py` | BacktestEngine walk-forward, summary stats |

When adding new functionality, add corresponding parametric tests.
Tests should be self-contained and not require network access (mock yfinance
calls when testing the data loader). Use `conftest.py` fixtures for
reproducible seeds and synthetic data.

---

## 10. Common Pitfalls

| Pitfall | Explanation |
|---------|-------------|
| Forgetting both languages | Every new string must be added to `STRINGS["en"]` **and** `STRINGS["es"]` in `i18n/__init__.py` |
| Comparing translated sidebar values | Asset / model selection uses translated keys — don't compare against hardcoded English strings |
| Editing `app.py` instead of `streamlit_app.py` | `app.py` is just a bootstrap; all GUI logic is in `src/gldpred/app/streamlit_app.py` |
| Large model files in git | Model artifacts in `data/model_registry/` should be git-ignored |
| Breaking the `_t()` pattern | Always call `t = _t()` at the top of each tab/section; `t` must be resolved fresh for each render |
| Wrong output shape | All models return `(batch, K, Q)` — never `(batch,)` or a tuple. Verify shape in tests |
| Using MSE/BCE loss | The project uses **pinball (quantile) loss** exclusively. Do not introduce MSE or BCE |
| Forgetting `forecast_steps` / `quantiles` args | Model constructors require these; omitting them will use defaults `(20, (0.1, 0.5, 0.9))` |
| Treating targets as `(N,)` | `create_sequences` returns multi-step targets of shape `(N, K)` |
| Old i18n path | i18n is now at `src/gldpred/i18n/__init__.py`, **not** `src/gldpred/app/i18n.py` |
| Old data loader class | Use `AssetDataLoader`, not the removed `GLDDataLoader` |
| Diagnostics on short runs | `DiagnosticsAnalyzer` needs ≥ 4 epochs; fewer returns a "not enough data" verdict |
| Hardcoding asset ticker | Always use `SUPPORTED_ASSETS` from `config`; never hardcode `"GLD"` |
| Missing `AssetInfo` for new ticker | Every asset in `SUPPORTED_ASSETS` must have a matching entry in `ASSET_CATALOG` (`config/assets.py`) |
| Ignoring `RiskMetrics` in tests | New decision-engine tests should verify `rec.risk` fields (stop_loss_pct, take_profit_pct, etc.) |
| Skipping model assignment | The Compare tab requires at least one asset with an assigned primary model via `ModelAssignments` |
| Using `KEY_TRAINER` for inference | Forecast / Recommendation / Evaluation tabs use `ModelBundle` via `get_active_model()` — never read `KEY_TRAINER` outside the training tab |
