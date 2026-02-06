# GLD Price Prediction - Quick Reference Guide

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run Streamlit GUI (Recommended)

```bash
streamlit run app.py
# or use the convenience script
./run.sh
```

### Option 2: Run Example Script

```bash
python example.py
```

### Option 3: Run Test Suite

```bash
python test_suite.py
```

## Using the Streamlit GUI

### 1. Data Tab
- Set start and end dates for historical data
- Click "Load Data" to download GLD price data
- View price charts and data statistics

### 2. Train Model Tab
Configure model settings:
- **Model Architecture**: Choose GRU or LSTM
- **Task Type**: 
  - Regression: Predicts future returns
  - Classification: Predicts buy/no-buy signals
- **Horizon**: Choose 1, 5, or 20 days
- **Sequence Length**: Number of historical days to use (10-60)
- **Hidden Size**: Model complexity (32-128)
- **Layers**: Number of recurrent layers (1-4)
- **Epochs**: Training iterations (10-200)
- **Batch Size**: Training batch size (16-128)
- **Learning Rate**: Optimizer learning rate

Click "Train Model" to start training.

### 3. Predictions Tab
- View predicted vs actual returns/prices
- See recent predictions in table format
- Interactive charts with Plotly

### 4. Evaluation Tab
- View performance metrics
- Regression: MSE, RMSE, MAE, R²
- Classification: Accuracy, Precision, Recall, F1, Confusion Matrix

## API Usage Examples

### Load and Prepare Data

```python
from data_loader import GLDDataLoader
from feature_engineering import FeatureEngineering

# Load GLD data
loader = GLDDataLoader(ticker='GLD')
data = loader.load_data()

# Create features
fe = FeatureEngineering()
data_with_features = fe.add_technical_indicators(data)
features = fe.select_features(data_with_features).ffill().bfill()

# Create targets for 5-day horizon
returns = loader.compute_returns(horizon=5)
signals = loader.compute_signals(horizon=5)

# Create sequences
X, y = fe.create_sequences(features, returns, seq_length=20)
```

### Train a Regression Model

```python
from models import GRURegressor
from trainer import ModelTrainer

# Initialize model
model = GRURegressor(input_size=X.shape[2], hidden_size=64, num_layers=2)

# Train
trainer = ModelTrainer(model, task='regression')
train_loader, val_loader = trainer.prepare_data(X, y, batch_size=32)
history = trainer.train(train_loader, val_loader, epochs=50)

# Save
trainer.save_model('models/gru_regression_h5.pth')
```

### Train a Classification Model

```python
from models import LSTMClassifier
from trainer import ModelTrainer

# Initialize model
model = LSTMClassifier(input_size=X.shape[2], hidden_size=64, num_layers=2)

# Train
trainer = ModelTrainer(model, task='classification')
train_loader, val_loader = trainer.prepare_data(X, y, batch_size=32)
history = trainer.train(train_loader, val_loader, epochs=50)

# Save
trainer.save_model('models/lstm_classifier_h5.pth')
```

### Make Predictions

```python
# Make predictions on new data
predictions = trainer.predict(X_new)

# For classification, convert to binary
buy_signals = (predictions > 0.5).astype(int)
```

### Evaluate Model

```python
from evaluator import ModelEvaluator

evaluator = ModelEvaluator()

# Regression metrics
metrics = evaluator.evaluate_regression(y_true, y_pred)
evaluator.print_metrics(metrics, task='regression')

# Classification metrics
metrics = evaluator.evaluate_classification(y_true, y_pred)
evaluator.print_metrics(metrics, task='classification')
```

### Load Pre-trained Model

```python
from models import GRURegressor
from trainer import ModelTrainer

# Create model with same architecture
model = GRURegressor(input_size=28, hidden_size=64, num_layers=2)
trainer = ModelTrainer(model, task='regression')

# Load weights
trainer.load_model('models/gru_regression_h5.pth')

# Make predictions
predictions = trainer.predict(X)
```

## Model Architectures

### GRU (Gated Recurrent Unit)
- Faster training than LSTM
- Good for shorter sequences
- Fewer parameters

### LSTM (Long Short-Term Memory)
- Better for long-term dependencies
- More parameters than GRU
- Better at learning complex patterns

## Prediction Horizons

- **1-day**: Next day predictions (short-term trading)
- **5-day**: One week ahead (swing trading)
- **20-day**: One month ahead (position trading)

## Tips for Best Results

1. **Data**: Use at least 2-3 years of historical data
2. **Sequence Length**: 20-30 days typically works well
3. **Hidden Size**: Start with 64, increase if underfitting
4. **Layers**: 2 layers is a good balance
5. **Epochs**: Train until validation loss stops improving
6. **Horizons**: Shorter horizons are generally easier to predict

## Troubleshooting

### "No data found" error
- Check internet connection
- Verify ticker symbol is correct ('GLD')
- Try different date ranges

### Poor model performance
- Increase training epochs
- Try different model architectures (GRU vs LSTM)
- Adjust sequence length
- Check for data quality issues

### Out of memory
- Reduce batch size
- Reduce hidden size or number of layers
- Use fewer training samples

## File Structure

```
├── app.py                    # Streamlit GUI
├── data_loader.py           # Data loading with yfinance
├── feature_engineering.py   # Feature creation
├── models.py                # PyTorch models (GRU/LSTM)
├── trainer.py               # Training pipeline
├── evaluator.py             # Evaluation metrics
├── example.py               # Usage examples
├── test_suite.py            # Test suite
├── run.sh                   # Quick start script
├── requirements.txt         # Dependencies
└── models/                  # Saved models (created automatically)
```

## Technical Indicators Used

- **Moving Averages**: SMA, EMA (5, 10, 20, 50 days)
- **Volatility**: Rolling standard deviation (5, 20 days)
- **Momentum**: Price differences (5, 10 days)
- **RSI**: Relative Strength Index (14 days)
- **MACD**: Moving Average Convergence Divergence
- **Volume**: Volume ratios and averages
- **Lag Features**: Previous prices and returns (1-5 days)

Total: 28 engineered features
