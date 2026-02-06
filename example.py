"""
Example script demonstrating how to use the GLD price prediction models.
"""
from data_loader import GLDDataLoader
from feature_engineering import FeatureEngineering
from models import GRURegressor, LSTMClassifier
from trainer import ModelTrainer
from evaluator import ModelEvaluator
import numpy as np


def main():
    print("GLD Price Prediction Example")
    print("=" * 50)
    
    # 1. Load data
    print("\n1. Loading GLD data...")
    loader = GLDDataLoader(ticker='GLD')
    data = loader.load_data()
    print(f"Loaded {len(data)} records")
    
    # 2. Feature engineering
    print("\n2. Creating features...")
    fe = FeatureEngineering()
    data_with_features = fe.add_technical_indicators(data)
    features = fe.select_features(data_with_features)
    features = features.ffill().bfill()
    print(f"Created {len(features.columns)} features")
    
    # 3. Prepare targets for different horizons
    horizons = [1, 5, 20]
    
    for horizon in horizons:
        print(f"\n{'='*50}")
        print(f"Training models for {horizon}-day horizon")
        print(f"{'='*50}")
        
        # Regression task
        print(f"\n3. Preparing regression targets (returns)...")
        targets_reg = loader.compute_returns(horizon=horizon)
        X_reg, y_reg = fe.create_sequences(features, targets_reg, seq_length=20)
        print(f"Created {len(X_reg)} sequences for regression")
        
        # Classification task
        print(f"\n4. Preparing classification targets (signals)...")
        targets_clf = loader.compute_signals(horizon=horizon)
        X_clf, y_clf = fe.create_sequences(features, targets_clf, seq_length=20)
        print(f"Created {len(X_clf)} sequences for classification")
        
        # 5. Train regression model (GRU)
        print(f"\n5. Training GRU regression model...")
        input_size = X_reg.shape[2]
        model_reg = GRURegressor(input_size=input_size, hidden_size=64, num_layers=2)
        trainer_reg = ModelTrainer(model_reg, task='regression')
        train_loader_reg, val_loader_reg = trainer_reg.prepare_data(X_reg, y_reg, batch_size=32)
        history_reg = trainer_reg.train(train_loader_reg, val_loader_reg, epochs=30)
        
        # 6. Evaluate regression model
        print(f"\n6. Evaluating regression model...")
        predictions_reg = trainer_reg.predict(X_reg)
        evaluator = ModelEvaluator()
        metrics_reg = evaluator.evaluate_regression(y_reg, predictions_reg)
        evaluator.print_metrics(metrics_reg, task='regression')
        
        # 7. Train classification model (LSTM)
        print(f"\n7. Training LSTM classification model...")
        model_clf = LSTMClassifier(input_size=input_size, hidden_size=64, num_layers=2)
        trainer_clf = ModelTrainer(model_clf, task='classification')
        train_loader_clf, val_loader_clf = trainer_clf.prepare_data(X_clf, y_clf, batch_size=32)
        history_clf = trainer_clf.train(train_loader_clf, val_loader_clf, epochs=30)
        
        # 8. Evaluate classification model
        print(f"\n8. Evaluating classification model...")
        predictions_clf = trainer_clf.predict(X_clf)
        metrics_clf = evaluator.evaluate_classification(y_clf, predictions_clf)
        evaluator.print_metrics(metrics_clf, task='classification')
        
        # 9. Save models
        print(f"\n9. Saving models...")
        trainer_reg.save_model(f'models/gru_regression_h{horizon}.pth')
        trainer_clf.save_model(f'models/lstm_classification_h{horizon}.pth')
    
    print("\n" + "="*50)
    print("Example completed successfully!")
    print("="*50)


if __name__ == '__main__':
    main()
