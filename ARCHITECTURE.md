# GLD Price Prediction - Application Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT GUI (app.py)                      │
│  ┌───────────┬────────────────┬──────────────┬────────────────┐ │
│  │ Data Tab  │ Train Model Tab│ Predictions  │  Evaluation    │ │
│  │           │                │     Tab      │      Tab       │ │
│  └───────────┴────────────────┴──────────────┴────────────────┘ │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CORE MODULES                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐      ┌──────────────────┐                │
│  │  data_loader.py  │─────▶│ feature_eng.py   │                │
│  │                  │      │                  │                │
│  │ - yfinance API   │      │ - Technical      │                │
│  │ - GLD data       │      │   indicators     │                │
│  │ - Returns calc   │      │ - 28 features    │                │
│  │ - Signal gen     │      │ - Sequences      │                │
│  └──────────────────┘      └────────┬─────────┘                │
│                                     │                           │
│                                     ▼                           │
│                          ┌──────────────────┐                   │
│                          │   models.py      │                   │
│                          │                  │                   │
│                          │ - GRURegressor   │                   │
│                          │ - LSTMRegressor  │                   │
│                          │ - GRUClassifier  │                   │
│                          │ - LSTMClassifier │                   │
│                          └────────┬─────────┘                   │
│                                   │                             │
│                                   ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  evaluator.py    │◀───│   trainer.py     │                  │
│  │                  │    │                  │                  │
│  │ - Regression     │    │ - Training loop  │                  │
│  │   metrics        │    │ - Data prep      │                  │
│  │ - Classification │    │ - Save/Load      │                  │
│  │   metrics        │    │ - Predictions    │                  │
│  └──────────────────┘    └──────────────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load GLD data (yfinance)                                    │
│          ▼                                                       │
│  2. Create technical indicators (28 features)                   │
│          ▼                                                       │
│  3. Generate targets (returns or signals)                       │
│          ▼                                                       │
│  4. Create sequences (sliding window)                           │
│          ▼                                                       │
│  5. Normalize and split data                                    │
│          ▼                                                       │
│  6. Train PyTorch model (GRU/LSTM)                              │
│          ▼                                                       │
│  7. Evaluate performance                                        │
│          ▼                                                       │
│  8. Make predictions                                            │
│          ▼                                                       │
│  9. Visualize results                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      MODEL ARCHITECTURES                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT SEQUENCE (batch, seq_len, features)                      │
│          │                                                       │
│          ▼                                                       │
│  ┌─────────────┐         ┌─────────────┐                       │
│  │  GRU/LSTM   │         │  GRU/LSTM   │                       │
│  │   Layer 1   │         │   Layer 1   │                       │
│  └──────┬──────┘         └──────┬──────┘                       │
│         │                       │                               │
│         ▼                       ▼                               │
│  ┌─────────────┐         ┌─────────────┐                       │
│  │  GRU/LSTM   │         │  GRU/LSTM   │                       │
│  │   Layer 2   │         │   Layer 2   │                       │
│  └──────┬──────┘         └──────┬──────┘                       │
│         │                       │                               │
│         ▼                       ▼                               │
│  ┌─────────────┐         ┌─────────────┐                       │
│  │   Linear    │         │   Linear    │                       │
│  │   + ReLU    │         │   + ReLU    │                       │
│  │  + Dropout  │         │  + Dropout  │                       │
│  └──────┬──────┘         └──────┬──────┘                       │
│         │                       │                               │
│         ▼                       ▼                               │
│  ┌─────────────┐         ┌─────────────┐                       │
│  │   Linear    │         │   Linear    │                       │
│  └──────┬──────┘         │  + Sigmoid  │                       │
│         │                └──────┬──────┘                       │
│         ▼                       ▼                               │
│    REGRESSION              CLASSIFICATION                        │
│   (returns)                (buy/no-buy)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION HORIZONS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1-Day Horizon:   Next day price movement                       │
│  ├─ Use case: Day trading, short-term decisions                │
│  └─ Target: (Price[t+1] - Price[t]) / Price[t]                 │
│                                                                  │
│  5-Day Horizon:   One week ahead prediction                     │
│  ├─ Use case: Swing trading, weekly rebalancing                │
│  └─ Target: (Price[t+5] - Price[t]) / Price[t]                 │
│                                                                  │
│  20-Day Horizon:  One month ahead prediction                    │
│  ├─ Use case: Position trading, monthly decisions              │
│  └─ Target: (Price[t+20] - Price[t]) / Price[t]                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   TECHNICAL INDICATORS (28)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Price-based:                                                   │
│  - Returns (1 day)                                              │
│  - SMA (5, 10, 20, 50 days)                                     │
│  - EMA (5, 10, 20, 50 days)                                     │
│  - Price to SMA ratios (20, 50 days)                            │
│  - Momentum (5, 10 days)                                        │
│                                                                  │
│  Volatility:                                                    │
│  - Rolling volatility (5, 20 days)                              │
│                                                                  │
│  Technical Indicators:                                          │
│  - RSI (14 days)                                                │
│  - MACD and MACD Signal                                         │
│                                                                  │
│  Volume:                                                        │
│  - Volume SMA (20 days)                                         │
│  - Volume ratio                                                 │
│                                                                  │
│  Lag Features:                                                  │
│  - Lagged prices (1, 2, 3, 5 days)                              │
│  - Lagged returns (1, 2, 3, 5 days)                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Data Loading
- Automated download from Yahoo Finance
- Configurable date ranges
- Error handling and validation

### 2. Feature Engineering
- 28 technical indicators
- Automatic normalization
- Sequence creation for time series

### 3. Model Training
- 4 model types (GRU/LSTM × Regression/Classification)
- Configurable hyperparameters
- Training history tracking
- Model checkpointing

### 4. Evaluation
- Comprehensive metrics
- Visualization of results
- Confusion matrices for classification

### 5. Streamlit GUI
- Interactive configuration
- Real-time training progress
- Interactive charts with Plotly
- Easy model comparison

## Technologies Used

- **PyTorch**: Deep learning framework
- **yfinance**: Financial data API
- **pandas/numpy**: Data manipulation
- **scikit-learn**: Preprocessing and metrics
- **Streamlit**: Web interface
- **Plotly/Matplotlib**: Visualization
