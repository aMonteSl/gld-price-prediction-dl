# GLD Price Prediction with Deep Learning

Deep learning application for forecasting GLD (Gold ETF) price movements using historical market data. The project uses PyTorch with GRU and LSTM architectures to predict both regression targets (returns) and classification targets (buy/no-buy signals) at multiple time horizons (1, 5, and 20 days).

## Features

- **Data Loading**: Automated fetching of GLD historical data using yfinance
- **Feature Engineering**: Technical indicators including:
  - Moving averages (SMA, EMA)
  - Volatility measures
  - Momentum indicators
  - RSI, MACD
  - Volume indicators
  - Lag features
- **Deep Learning Models**:
  - GRU and LSTM architectures
  - Both regression (returns prediction) and classification (buy/no-buy signals)
  - Customizable hyperparameters
- **Training Pipeline**: 
  - Automated data preprocessing and normalization
  - Train/validation split
  - Model checkpointing
- **Evaluation Metrics**:
  - Regression: MSE, RMSE, MAE, R²
  - Classification: Accuracy, Precision, Recall, F1, Confusion Matrix
- **Streamlit GUI**: Interactive web interface for:
  - Data exploration and visualization
  - Model configuration and training
  - Real-time predictions
  - Performance evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/aMonteSl/gld-price-prediction-dl.git
cd gld-price-prediction-dl

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Streamlit GUI

```bash
streamlit run app.py
```

The GUI provides an intuitive interface to:
1. **Load Data**: Download GLD historical data for a specified date range
2. **Configure Models**: Select model type (GRU/LSTM), task (regression/classification), horizon (1/5/20 days), and hyperparameters
3. **Train Models**: Train models with real-time progress tracking
4. **View Predictions**: Visualize predictions vs actual prices
5. **Evaluate Performance**: See detailed metrics and confusion matrices

### Running the Example Script

```bash
python example.py
```

This script demonstrates:
- Loading GLD data
- Feature engineering
- Training both regression and classification models
- Model evaluation
- Saving trained models

### Using the API Programmatically

```python
from data_loader import GLDDataLoader
from feature_engineering import FeatureEngineering
from models import GRURegressor, LSTMClassifier
from trainer import ModelTrainer
from evaluator import ModelEvaluator

# Load data
loader = GLDDataLoader(ticker='GLD')
data = loader.load_data()

# Feature engineering
fe = FeatureEngineering()
data_with_features = fe.add_technical_indicators(data)
features = fe.select_features(data_with_features)

# Prepare targets
targets = loader.compute_returns(horizon=5)  # 5-day returns
X, y = fe.create_sequences(features, targets, seq_length=20)

# Train model
model = GRURegressor(input_size=X.shape[2])
trainer = ModelTrainer(model, task='regression')
train_loader, val_loader = trainer.prepare_data(X, y)
history = trainer.train(train_loader, val_loader, epochs=50)

# Evaluate
predictions = trainer.predict(X)
metrics = ModelEvaluator.evaluate_regression(y, predictions)
print(metrics)

# Save model
trainer.save_model('models/gru_regression_h5.pth')
```

## Project Structure

```
gld-price-prediction-dl/
├── app.py                    # Streamlit GUI application
├── data_loader.py           # Data loading with yfinance
├── feature_engineering.py   # Feature creation and preprocessing
├── models.py                # PyTorch model architectures
├── trainer.py               # Training pipeline
├── evaluator.py             # Evaluation metrics
├── example.py               # Usage example script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Models

### GRU/LSTM Regressor
- Predicts future returns at specified horizon
- Output: Continuous value representing expected return

### GRU/LSTM Classifier
- Predicts buy/no-buy signals
- Output: Binary signal (1 = buy, 0 = no-buy)

## Prediction Horizons

The application supports three prediction horizons:
- **1-day**: Short-term predictions
- **5-day**: Medium-term predictions  
- **20-day**: Long-term predictions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- pandas, numpy
- yfinance
- scikit-learn
- streamlit
- matplotlib, plotly

See `requirements.txt` for complete list.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
