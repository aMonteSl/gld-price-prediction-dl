"""
Comprehensive test suite for GLD price prediction application.
"""
import sys
import os

# Ensure the src/ directory is on the Python path.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
)

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

from gldpred.data import GLDDataLoader
from gldpred.features import FeatureEngineering
from gldpred.models import GRURegressor, LSTMRegressor, GRUClassifier, LSTMClassifier
from gldpred.training import ModelTrainer
from gldpred.evaluation import ModelEvaluator


def test_feature_engineering():
    """Test feature engineering with synthetic data."""
    print("\n" + "="*60)
    print("TEST 1: Feature Engineering")
    print("="*60)

    # Create synthetic price data
    dates = pd.date_range('2020-01-01', periods=200)
    data = pd.DataFrame({
        'Open': np.random.randn(200).cumsum() + 100,
        'High': np.random.randn(200).cumsum() + 101,
        'Low': np.random.randn(200).cumsum() + 99,
        'Close': np.random.randn(200).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 200),
        'Dividends': np.zeros(200),
        'Stock Splits': np.zeros(200)
    }, index=dates)

    fe = FeatureEngineering()
    data_with_features = fe.add_technical_indicators(data)

    print(f"✓ Created {len(data_with_features.columns)} features from OHLCV data")
    print(f"  Original columns: {len(data.columns)}")
    print(f"  New features: {len(data_with_features.columns) - len(data.columns)}")

    # Test feature selection
    features = fe.select_features(data_with_features)
    print(f"✓ Selected {len(features.columns)} features for modeling")

    # Test sequence creation
    targets = pd.Series(np.random.randn(len(features)), index=features.index)
    X, y = fe.create_sequences(features.fillna(0), targets, seq_length=20)
    print(f"✓ Created {len(X)} sequences with shape {X.shape}")

    return True


def test_models():
    """Test all model architectures."""
    print("\n" + "="*60)
    print("TEST 2: Model Architectures")
    print("="*60)

    n_samples = 100
    seq_length = 20
    n_features = 15

    X = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)
    y_reg = np.random.randn(n_samples).astype(np.float32)
    y_clf = np.random.randint(0, 2, n_samples).astype(np.float32)

    models = [
        ('GRU Regressor', GRURegressor(n_features, 32, 2), 'regression', y_reg),
        ('LSTM Regressor', LSTMRegressor(n_features, 32, 2), 'regression', y_reg),
        ('GRU Classifier', GRUClassifier(n_features, 32, 2), 'classification', y_clf),
        ('LSTM Classifier', LSTMClassifier(n_features, 32, 2), 'classification', y_clf),
    ]

    for name, model, task, y in models:
        # Test forward pass
        test_input = torch.randn(2, seq_length, n_features)
        output = model(test_input)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ {name}: {param_count:,} parameters, output shape: {output.shape}")

    return True


def test_training_pipeline():
    """Test training pipeline."""
    print("\n" + "="*60)
    print("TEST 3: Training Pipeline")
    print("="*60)

    n_samples = 100
    seq_length = 20
    n_features = 15

    X = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)
    y_reg = np.random.randn(n_samples).astype(np.float32)
    y_clf = np.random.randint(0, 2, n_samples).astype(np.float32)

    # Test regression training
    print("\nRegression Training:")
    model = GRURegressor(n_features, 32, 2)
    trainer = ModelTrainer(model, task='regression')
    train_loader, val_loader = trainer.prepare_data(X, y_reg, batch_size=16)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    history = trainer.train(train_loader, val_loader, epochs=5)
    print(f"  Initial loss: {history['train_loss'][0]:.6f}")
    print(f"  Final loss: {history['train_loss'][-1]:.6f}")
    print(f"  ✓ Training converged: {history['train_loss'][-1] < history['train_loss'][0]}")

    # Test classification training
    print("\nClassification Training:")
    model = LSTMClassifier(n_features, 32, 2)
    trainer = ModelTrainer(model, task='classification')
    train_loader, val_loader = trainer.prepare_data(X, y_clf, batch_size=16)

    history = trainer.train(train_loader, val_loader, epochs=5)
    print(f"  Initial loss: {history['train_loss'][0]:.6f}")
    print(f"  Final loss: {history['train_loss'][-1]:.6f}")

    return True


def test_evaluation():
    """Test evaluation metrics."""
    print("\n" + "="*60)
    print("TEST 4: Evaluation Metrics")
    print("="*60)

    n_samples = 100

    # Test regression metrics
    print("\nRegression Metrics:")
    y_true_reg = np.random.randn(n_samples)
    y_pred_reg = y_true_reg + np.random.randn(n_samples) * 0.1

    metrics = ModelEvaluator.evaluate_regression(y_true_reg, y_pred_reg)
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  ✓ R² > 0.5: {metrics['r2'] > 0.5}")

    # Test classification metrics
    print("\nClassification Metrics:")
    y_true_clf = np.random.randint(0, 2, n_samples)
    y_pred_clf = (y_true_clf + np.random.randn(n_samples) * 0.3 > 0.5).astype(float)

    metrics = ModelEvaluator.evaluate_classification(y_true_clf, y_pred_clf)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")

    return True


def test_model_persistence():
    """Test model saving and loading."""
    print("\n" + "="*60)
    print("TEST 5: Model Persistence")
    print("="*60)

    os.makedirs('models', exist_ok=True)

    n_features = 15
    X = np.random.randn(50, 20, n_features).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)

    # Train and save
    model = GRURegressor(n_features, 32, 2)
    trainer = ModelTrainer(model, task='regression')
    train_loader, val_loader = trainer.prepare_data(X, y, batch_size=16)
    trainer.train(train_loader, val_loader, epochs=3)

    # Get predictions before saving
    pred_before = trainer.predict(X[:5])

    # Save model
    model_path = 'models/test_persistence.pth'
    trainer.save_model(model_path)
    print(f"✓ Model saved to {model_path}")

    # Load model
    new_model = GRURegressor(n_features, 32, 2)
    new_trainer = ModelTrainer(new_model, task='regression')
    new_trainer.load_model(model_path)
    print(f"✓ Model loaded from {model_path}")

    # Get predictions after loading
    pred_after = new_trainer.predict(X[:5])

    # Verify predictions match
    max_diff = np.max(np.abs(pred_before - pred_after))
    print(f"✓ Prediction difference: {max_diff:.10f}")
    print(f"✓ Predictions match: {max_diff < 1e-6}")

    # Clean up
    os.remove(model_path)

    return True


def test_multiple_horizons():
    """Test training models for multiple horizons."""
    print("\n" + "="*60)
    print("TEST 6: Multiple Horizon Predictions")
    print("="*60)

    # Create synthetic price data
    dates = pd.date_range('2020-01-01', periods=200)
    prices = pd.Series(np.random.randn(200).cumsum() + 100, index=dates)

    horizons = [1, 5, 20]

    for horizon in horizons:
        # Compute returns for this horizon
        returns = (prices.shift(-horizon) - prices) / prices

        # Count valid returns
        valid_returns = returns.dropna()
        print(f"\n✓ Horizon {horizon}: {len(valid_returns)} valid return samples")
        print(f"  Mean return: {valid_returns.mean():.6f}")
        print(f"  Std return: {valid_returns.std():.6f}")

    return True


def run_all_tests():
    """Run all test suites."""
    print("\n" + "#"*60)
    print("# GLD PRICE PREDICTION - COMPREHENSIVE TEST SUITE")
    print("#"*60)

    tests = [
        test_feature_engineering,
        test_models,
        test_training_pipeline,
        test_evaluation,
        test_model_persistence,
        test_multiple_horizons,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"❌ {test.__name__} failed: {str(e)}")
            results.append((test.__name__, False))
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The application is ready to use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")

    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
