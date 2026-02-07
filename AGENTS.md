# AGENTS.md — AI-Assisted Development Guide

> This file helps GitHub Copilot, ChatGPT, Claude, and other AI coding
> assistants understand the project structure, conventions, and constraints
> so they can generate accurate, consistent code.

---

## 1. Project Overview

**GLD Price Prediction** is a deep-learning application that downloads
historical **GLD ETF** (Gold) data, engineers technical features, and trains
**GRU / LSTM / TCN** models (PyTorch) to predict future price movements. It
supports **regression** (predicted returns), **classification** (buy /
no-buy signals), and **multi-task learning** (both simultaneously) at
1-, 5-, and 20-day horizons, with **automatic training diagnostics**.

A **Streamlit** GUI provides interactive data exploration, training,
prediction visualisation, evaluation, and a built-in tutorial — all fully
internationalised in **English and Spanish**.

---

## 2. Repository Layout

```
gld-price-prediction-dl/
├── app.py                        # Thin entrypoint — run `streamlit run app.py`
├── requirements.txt              # pip dependencies
├── pytest.ini                    # pytest configuration
├── README.md                     # User-facing README
├── USER_GUIDE.md                 # Comprehensive tutorial / user guide
├── AGENTS.md                     # ← You are here
│
├── src/gldpred/                  # Main Python package
│   ├── __init__.py               # Package root (version, public API) — v2.0.0
│   ├── config.py                 # DataConfig, ModelConfig, TrainingConfig
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py             # GLDDataLoader (yfinance wrapper)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py        # FeatureEngineering (28 technical indicators)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── architectures.py      # 9 models: GRU/LSTM/TCN × Reg/Cls/MultiTask
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py            # ModelTrainer (reg/cls/multitask, save/load)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py          # Regression, classification & multitask metrics
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py          # Predictor (wraps trainer for signal generation)
│   │
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   └── analyzer.py           # DiagnosticsAnalyzer (loss-curve analysis)
│   │
│   └── app/
│       ├── __init__.py
│       ├── streamlit_app.py      # Streamlit GUI (5 tabs + sidebar)
│       └── i18n.py               # All UI strings in EN and ES
│
├── scripts/
│   └── example.py                # CLI example script
│
├── tests/
│   ├── conftest.py               # Shared fixtures & seeds
│   ├── test_models.py            # 9 model architectures
│   ├── test_trainer.py           # Training / prediction / persistence
│   ├── test_evaluator.py         # Metric calculations
│   ├── test_diagnostics.py       # Loss-curve analysis
│   ├── test_features.py          # Feature engineering & sequences
│   └── test_suite.py             # Legacy test runner
│
└── models/                       # Saved .pth model weights (git-ignored)
```

---

## 3. Key Modules & Public API

| Module | Key Export | Purpose |
|--------|-----------|---------|
| `gldpred.config` | `DataConfig`, `ModelConfig`, `TrainingConfig` | Typed configuration via `@dataclass` |
| `gldpred.data` | `GLDDataLoader` | Download & cache GLD OHLCV data via yfinance |
| `gldpred.features` | `FeatureEngineering` | Compute 28 technical features (SMA, EMA, RSI, MACD, …) |
| `gldpred.models` | 9 model classes (see below) | PyTorch `nn.Module` subclasses |
| `gldpred.training` | `ModelTrainer` | Train/val loop, StandardScaler, reg/cls/multitask, save/load |
| `gldpred.evaluation` | `ModelEvaluator` | Regression, classification & multitask metrics |
| `gldpred.inference` | `Predictor` | Convenience wrapper for batch prediction + signal generation |
| `gldpred.diagnostics` | `DiagnosticsAnalyzer` | Heuristic loss-curve analysis (verdict + suggestions) |
| `gldpred.app.i18n` | `STRINGS`, `LANGUAGES` | Dictionary-based i18n (English / Spanish) |
| `gldpred.app.streamlit_app` | *(script)* | Streamlit application with 5 tabs |

### Model classes (all in `gldpred.models`)

| Architecture | Regression | Classification | Multi-task |
|-------------|-----------|---------------|------------|
| GRU | `GRURegressor` | `GRUClassifier` | `GRUMultiTask` |
| LSTM | `LSTMRegressor` | `LSTMClassifier` | `LSTMMultiTask` |
| TCN | `TCNRegressor` | `TCNClassifier` | `TCNMultiTask` |

---

## 4. Technology Stack

| Technology | Version | Role |
|------------|---------|------|
| Python | 3.10+ | Language |
| PyTorch | ≥ 2.0 | Deep learning framework |
| Streamlit | ≥ 1.30 | Web GUI |
| yfinance | ≥ 0.2 | Market data download |
| scikit-learn | ≥ 1.3 | Preprocessing (StandardScaler, train_test_split), metrics |
| pandas / numpy | ≥ 2.0 / ≥ 1.24 | Data manipulation |
| matplotlib / plotly | ≥ 3.7 / ≥ 5.18 | Training plots / interactive charts |

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
  `from gldpred.data import GLDDataLoader`.
- Private helpers start with `_`.

### 5.3 PyTorch models
- All models subclass `nn.Module`.
- Constructor signature: `(input_size, hidden_size=64, num_layers=2, dropout=0.2)`.
- **Single-output models:** `forward()` returns a tensor of shape `(batch,)`.
  Regression outputs raw values; classification applies sigmoid.
- **Multi-task models:** `forward()` returns a tuple `(reg_out, cls_logits)`
  where `cls_logits` are **raw logits** (no sigmoid — the loss uses
  `BCEWithLogitsLoss` for numerical stability).
- TCN models use causal dilated 1-D convolutions with exponential dilation
  and residual connections (`_CausalConv1dBlock`, `_TCNBackbone`).
- Multi-task models use `_MultiTaskWrapper` which attaches `reg_head` and
  `cls_head` on top of any backbone.

### 5.4 i18n
- Every user-facing string lives in `src/gldpred/app/i18n.py` under
  `STRINGS["en"]` and `STRINGS["es"]`.
- In the Streamlit app, call `t = _t()` then use `t["key_name"]`.
- Key naming convention: `<section>_<purpose>`, e.g. `train_header`,
  `pred_actual_returns`, `eval_cm_ylabel`, `tut_s3_body`.
- When adding new UI text, add the key to **both** `"en"` and `"es"`.

---

## 6. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Run tests (pytest)
pytest

# Run legacy tests
python tests/test_suite.py

# Run the CLI example
python scripts/example.py
```

---

## 7. Critical Constraints

> **DO NOT** change ML logic, training behaviour, or model outputs.

These are strictly off-limits unless explicitly requested:

- Loss functions and optimiser configuration in `ModelTrainer`.
- Feature engineering formulas in `FeatureEngineering`.
- Model architectures (layer structure, activation functions) in `architectures.py`.
- Evaluation metric calculations in `ModelEvaluator`.
- Sequence creation and train/val splitting logic.

**Safe to change:**
- UI layout, styling, and new Streamlit widgets in `streamlit_app.py`.
- i18n strings in `i18n.py` (add/fix translations).
- Documentation files (`README.md`, `USER_GUIDE.md`).
- Test coverage in `tests/test_suite.py`.
- Configuration defaults in `config.py`.

---

## 8. Extension Guidelines

### Adding a new model architecture
1. Create the `nn.Module` sub-class in `models/architectures.py`.
2. Export it from `models/__init__.py`.
3. Add entries to `_MODEL_MAP` in `streamlit_app.py` for each task
   (regression, classification, multitask).
4. Add a sidebar option in `streamlit_app.py` (update the `model_type`
   selectbox).
5. Add the corresponding i18n keys in `i18n.py` (both EN and ES).
6. Add parametric test entries in `tests/test_models.py`.

### Adding a new language
1. Choose an ISO 639-1 code (e.g. `"fr"`).
2. Add `"Français": "fr"` to the `LANGUAGES` dict in `i18n.py`.
3. Add `"fr": { ... }` to `STRINGS` with all the same keys as `"en"`.
4. No changes needed in `streamlit_app.py` — the language selector picks
   up new entries automatically.

### Adding a new evaluation metric
1. Implement it in `evaluation/evaluator.py` inside the appropriate method.
2. Add the metric name to the returned dict.
3. Display it in Tab 4 of `streamlit_app.py` (use i18n keys).

### Adding a new feature
1. Add the computation in `features/engineering.py`.
2. The feature is automatically included because `ModelTrainer` selects
   all numeric columns from the engineered DataFrame.

---

## 9. Testing

Run the test suite with:

```bash
# Preferred — pytest (74+ tests)
pytest
pytest -v            # verbose
pytest tests/test_models.py  # single module

# Legacy runner
python tests/test_suite.py
```

### Test layout

| File | Covers |
|------|--------|
| `tests/conftest.py` | Shared fixtures: seeds, synthetic data, OHLCV |
| `tests/test_models.py` | All 9 model architectures (shape, gradient, sigmoid range) |
| `tests/test_trainer.py` | Data preparation, training loop, predict, save/load |
| `tests/test_evaluator.py` | Regression, classification, multitask metrics |
| `tests/test_diagnostics.py` | Loss-curve verdicts (healthy, overfit, underfit, noisy) |
| `tests/test_features.py` | Feature engineering, sequences, multitask targets |
| `tests/test_suite.py` | Legacy test runner (kept for backwards compatibility) |

When adding new functionality, add corresponding parametric tests.
Tests should be self-contained and not require network access (mock yfinance
calls when testing the data loader). Use `conftest.py` fixtures for
reproducible seeds and synthetic data.

---

## 10. Common Pitfalls

| Pitfall | Explanation |
|---------|-------------|
| Forgetting both languages | Every new string must be added to `STRINGS["en"]` **and** `STRINGS["es"]` |
| Comparing translated sidebar values | Task type detection uses `t["sidebar_task_regression"]` — don't compare against hardcoded English strings |
| Editing `app.py` instead of `streamlit_app.py` | `app.py` is just a bootstrap; all GUI logic is in `src/gldpred/app/streamlit_app.py` |
| Large model files in git | `.pth` files should be in `.gitignore`; they are generated by training |
| Breaking the `_t()` pattern | Always call `t = _t()` at the top of each tab/section; `t` must be resolved fresh for each render |
| Multi-task cls_logits vs probabilities | Multi-task `forward()` returns raw logits; apply `sigmoid()` only at inference, not during loss computation |
| Forgetting `_MODEL_MAP` | When adding a model, update the `_MODEL_MAP` dict in `streamlit_app.py` or the model won't appear in the UI |
| Diagnostics on short runs | `DiagnosticsAnalyzer` needs ≥ 4 epochs; fewer returns a "not enough data" verdict |
