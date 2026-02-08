"""End-to-end integration smoke tests.

Runs the full pipeline without network access, using synthetic OHLCV data:
  features → sequences → train → diagnostics → predict → evaluate → recommend
"""
from __future__ import annotations

import numpy as np
import pytest

from gldpred.features import FeatureEngineering
from gldpred.models import GRUForecaster, LSTMForecaster, TCNForecaster
from gldpred.training import ModelTrainer
from gldpred.evaluation import ModelEvaluator
from gldpred.diagnostics import DiagnosticsAnalyzer
from gldpred.inference import TrajectoryPredictor
from gldpred.decision import DecisionEngine
from conftest import FORECAST_STEPS, QUANTILES, SEQ_LENGTH


_ARCHS = [
    ("GRU", GRUForecaster),
    ("LSTM", LSTMForecaster),
    ("TCN", TCNForecaster),
]


class TestFullPipeline:
    """Smoke-test the full pipeline for each architecture."""

    @pytest.mark.parametrize("name,cls", _ARCHS, ids=[a[0] for a in _ARCHS])
    def test_pipeline(self, synthetic_ohlcv, name, cls):
        df = synthetic_ohlcv.copy()

        # 1. Feature engineering
        fe = FeatureEngineering
        featured_df = fe.add_technical_indicators(df)
        selected = fe.select_features(featured_df)
        assert not selected.empty

        # Drop NaN rows (created by rolling indicators) — align returns too
        clean = featured_df.dropna()
        clean_features = selected.loc[clean.index]
        daily_returns = clean["returns"]

        X, y = fe.create_sequences(
            clean_features, daily_returns,
            seq_length=SEQ_LENGTH, forecast_steps=FORECAST_STEPS,
        )
        assert X.ndim == 3  # (N, seq_len, features)
        assert y.ndim == 2  # (N, K)
        assert X.shape[1] == SEQ_LENGTH
        assert y.shape[1] == FORECAST_STEPS

        input_size = X.shape[2]

        # 2. Model + trainer
        model = cls(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            forecast_steps=FORECAST_STEPS,
            quantiles=QUANTILES,
        )
        trainer = ModelTrainer(model, quantiles=QUANTILES, device="cpu")
        train_dl, val_dl = trainer.prepare_data(X, y, test_size=0.2, batch_size=8)
        history = trainer.train(train_dl, val_dl, epochs=4, learning_rate=0.001)

        assert len(history["train_loss"]) == 4
        assert len(history["val_loss"]) == 4
        assert all(l > 0 for l in history["train_loss"])

        # 3. Diagnostics (needs ≥ 4 epochs)
        result = DiagnosticsAnalyzer.analyze(history)
        assert result.verdict in ("healthy", "overfitting", "underfitting", "noisy")
        assert isinstance(result.suggestions, list)

        # 4. Predict
        preds = trainer.predict(X[-5:])
        assert preds.shape == (5, FORECAST_STEPS, len(QUANTILES))

        # 5. Evaluate trajectory (pass median slice, not full quantile tensor)
        evaluator = ModelEvaluator()
        median_idx = list(QUANTILES).index(0.5)
        preds_median = preds[:, :, median_idx]  # (N, K)
        traj_metrics = evaluator.evaluate_trajectory(y[-5:], preds_median)
        assert "mae" in traj_metrics
        assert "rmse" in traj_metrics

        # 6. Evaluate quantiles
        quant_metrics = evaluator.evaluate_quantiles(
            y[-5:], preds, quantiles=list(QUANTILES)
        )
        for q in QUANTILES:
            key = f"q{int(q * 100)}_cal_error"
            assert key in quant_metrics

        # 7. Inference — trajectory forecast
        feature_names = list(clean_features.columns)
        predictor = TrajectoryPredictor(trainer)
        forecast = predictor.predict_trajectory(
            clean, feature_names, SEQ_LENGTH, "GLD"
        )
        assert forecast.price_paths.shape[0] == FORECAST_STEPS + 1  # K+1
        assert forecast.price_paths.shape[1] == len(QUANTILES)
        assert len(forecast.dates) == FORECAST_STEPS

        # 8. Decision engine
        engine = DecisionEngine(horizon_days=FORECAST_STEPS)
        reco = engine.recommend(
            forecast.returns_quantiles, clean, quantiles=QUANTILES
        )
        assert reco.action in ("BUY", "HOLD", "AVOID")
        assert 0 <= reco.confidence <= 100


class TestDataLoaderConfig:
    """Verify data-loading configuration (no network calls)."""

    def test_end_date_is_today(self):
        from gldpred.data import AssetDataLoader
        from datetime import datetime

        loader = AssetDataLoader(ticker="GLD")
        # end_date should be today's date
        assert loader.end_date.date() == datetime.now().date()

    def test_no_end_date_param(self):
        """AssetDataLoader constructor does not accept end_date."""
        from gldpred.data import AssetDataLoader
        import inspect

        sig = inspect.signature(AssetDataLoader.__init__)
        assert "end_date" not in sig.parameters

    def test_default_start_date_approx_5_years(self):
        from gldpred.data import AssetDataLoader
        from datetime import datetime, timedelta

        loader = AssetDataLoader(ticker="GLD")
        expected = datetime.now() - timedelta(days=365 * 5)
        # Within 2 days tolerance
        assert abs((loader.start_date - expected).days) <= 2
