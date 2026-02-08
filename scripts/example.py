"""
Example script — v3.0 multi-step quantile trajectory forecasting.
"""
import sys
import os

# Ensure the src/ directory is on the Python path.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
)

from gldpred.data import AssetDataLoader
from gldpred.features import FeatureEngineering
from gldpred.models import TCNForecaster
from gldpred.training import ModelTrainer
from gldpred.evaluation import ModelEvaluator
from gldpred.inference import TrajectoryPredictor
from gldpred.diagnostics import DiagnosticsAnalyzer
from gldpred.decision import DecisionEngine
import numpy as np


def main():
    print("Multi-Asset Price Prediction — v3.0 Example")
    print("=" * 55)

    # 1. Load data
    asset = "GLD"
    print(f"\n1. Loading {asset} data…")
    loader = AssetDataLoader(ticker=asset)
    data = loader.load_data()
    daily_ret = loader.daily_returns()
    print(f"   Loaded {len(data)} records")

    # 2. Feature engineering
    print("\n2. Computing technical indicators…")
    eng = FeatureEngineering()
    data = eng.add_technical_indicators(data)
    feature_names = eng.select_features(data)
    print(f"   {len(feature_names)} features selected")

    # 3. Multi-step sequences
    forecast_steps = 20
    seq_length = 20
    quantiles = (0.1, 0.5, 0.9)

    X, y = eng.create_sequences(
        data[feature_names].values,
        daily_ret.values,
        seq_length=seq_length,
        forecast_steps=forecast_steps,
    )
    print(f"   Sequences: X={X.shape}, y={y.shape}")

    # 4. Train TCN model
    print("\n3. Training TCN quantile forecaster…")
    input_size = X.shape[2]
    model = TCNForecaster(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        forecast_steps=forecast_steps,
        quantiles=quantiles,
    )
    trainer = ModelTrainer(model, quantiles=quantiles, device="cpu")
    train_loader, val_loader = trainer.prepare_data(X, y, test_size=0.2, batch_size=32)
    history = trainer.train(train_loader, val_loader, epochs=30, learning_rate=0.001)
    print(f"   Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"   Final val   loss: {history['val_loss'][-1]:.6f}")

    # 5. Diagnostics
    print("\n4. Training diagnostics…")
    diag = DiagnosticsAnalyzer.analyze(history)
    print(f"   Verdict: {diag.verdict}")
    print(f"   {diag.explanation}")

    # 6. Evaluate
    print("\n5. Evaluation on validation set…")
    split = int(len(X) * 0.8)
    y_val = y[split:]
    pred_val = trainer.predict(X[split:])
    evaluator = ModelEvaluator()
    traj_metrics = evaluator.evaluate_trajectory(y_val, pred_val[:, :, 1])
    quant_metrics = evaluator.evaluate_quantiles(y_val, pred_val, list(quantiles))
    evaluator.print_metrics(traj_metrics)
    evaluator.print_metrics(quant_metrics)

    # 7. Forecast trajectory
    print("\n6. Generating forecast trajectory…")
    predictor = TrajectoryPredictor(trainer)
    forecast = predictor.predict_trajectory(data, feature_names, seq_length, asset)
    print(f"   Forecast dates: {forecast.dates[0]} → {forecast.dates[-1]}")
    print(f"   P50 price path: {forecast.price_paths[:, 1].round(2)}")

    # 8. Decision recommendation
    print("\n7. Decision support…")
    engine = DecisionEngine(horizon_days=5)
    reco = engine.recommend(
        forecast.returns_quantiles,
        data,
        quantiles=forecast.quantiles,
        diagnostics_verdict=diag.verdict,
    )
    print(f"   Action: {reco.action}")
    print(f"   Confidence: {reco.confidence:.0f}/100")
    for r in reco.rationale:
        print(f"   • {r}")
    if reco.warnings:
        for w in reco.warnings:
            print(f"   ⚠ {w}")

    print("\n" + "=" * 55)
    print("Example completed successfully!")
    print("=" * 55)


if __name__ == "__main__":
    main()
