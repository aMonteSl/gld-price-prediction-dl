"""Tests for the model registry (save, load, list, delete)."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from gldpred.models import TCNForecaster
from gldpred.registry import ModelRegistry
from conftest import FORECAST_STEPS, INPUT_SIZE, QUANTILES


def _make_model():
    return TCNForecaster(
        input_size=INPUT_SIZE,
        hidden_size=32,
        num_layers=1,
        forecast_steps=FORECAST_STEPS,
        quantiles=QUANTILES,
    )


class TestModelRegistry:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model = _make_model()
            scaler = StandardScaler()
            scaler.fit(np.random.randn(20, INPUT_SIZE))

            model_id = reg.save_model(
                model, scaler,
                config={"asset": "GLD", "architecture": "TCN"},
                feature_names=[f"f{i}" for i in range(INPUT_SIZE)],
                training_summary={"epochs": 10, "final_val_loss": 0.01},
            )
            assert os.path.isdir(os.path.join(tmpdir, model_id))

            loaded_model, loaded_scaler, meta = reg.load_model(
                model_id, TCNForecaster,
                input_size=INPUT_SIZE,
                hidden_size=32,
                num_layers=1,
                forecast_steps=FORECAST_STEPS,
                quantiles=QUANTILES,
            )
            assert meta["model_id"] == model_id
            assert meta["asset"] == "GLD"
            assert meta["architecture"] == "TCN"
            assert loaded_scaler.mean_ is not None

    def test_list_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model = _make_model()
            scaler = StandardScaler()
            scaler.fit(np.random.randn(20, INPUT_SIZE))

            reg.save_model(model, scaler, {"asset": "GLD", "architecture": "TCN"}, [], {})
            reg.save_model(model, scaler, {"asset": "SLV", "architecture": "GRU"}, [], {})

            all_models = reg.list_models()
            assert len(all_models) == 2

            gld_models = reg.list_models(asset="GLD")
            assert len(gld_models) == 1

            tcn_models = reg.list_models(architecture="TCN")
            assert len(tcn_models) == 1

    def test_delete_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model = _make_model()
            scaler = StandardScaler()
            scaler.fit(np.random.randn(20, INPUT_SIZE))

            model_id = reg.save_model(model, scaler, {"asset": "GLD"}, [], {})
            assert len(reg.list_models()) == 1

            reg.delete_model(model_id)
            assert len(reg.list_models()) == 0

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            with pytest.raises(FileNotFoundError):
                reg.load_model("nonexistent", TCNForecaster, input_size=INPUT_SIZE)

    def test_metadata_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model = _make_model()
            scaler = StandardScaler()
            scaler.fit(np.random.randn(20, INPUT_SIZE))

            model_id = reg.save_model(
                model, scaler,
                config={"asset": "GLD", "architecture": "TCN"},
                feature_names=["a", "b", "c"],
                training_summary={"epochs": 5},
                evaluation_summary={"mse": 0.001},
            )
            _, _, meta = reg.load_model(
                model_id, TCNForecaster,
                input_size=INPUT_SIZE,
                hidden_size=32,
                num_layers=1,
                forecast_steps=FORECAST_STEPS,
                quantiles=QUANTILES,
            )
            assert "model_id" in meta
            assert "created_at" in meta
            assert "config" in meta
            assert "feature_names" in meta
            assert meta["feature_names"] == ["a", "b", "c"]
            assert "training_summary" in meta
            assert "evaluation_summary" in meta

    def test_custom_label_persistence(self):
        """Custom label is saved and retrieved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model = _make_model()
            scaler = StandardScaler()
            scaler.fit(np.random.randn(20, INPUT_SIZE))

            custom_label = "MyAwesomeModel_v1"
            model_id = reg.save_model(
                model, scaler,
                config={"asset": "GLD", "architecture": "TCN"},
                feature_names=[],
                training_summary={},
                label=custom_label,
            )

            _, _, meta = reg.load_model(
                model_id, TCNForecaster,
                input_size=INPUT_SIZE,
                hidden_size=32,
                num_layers=1,
                forecast_steps=FORECAST_STEPS,
                quantiles=QUANTILES,
            )
            assert meta["label"] == custom_label

    def test_label_validation_empty(self):
        """Empty label raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model = _make_model()
            scaler = StandardScaler()
            scaler.fit(np.random.randn(20, INPUT_SIZE))

            with pytest.raises(ValueError, match="cannot be empty"):
                reg.save_model(
                    model, scaler,
                    config={},
                    feature_names=[],
                    training_summary={},
                    label="   ",  # whitespace-only
                )

    def test_label_validation_too_long(self):
        """Label exceeding max length raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model = _make_model()
            scaler = StandardScaler()
            scaler.fit(np.random.randn(20, INPUT_SIZE))

            long_label = "x" * 61  # MAX_LABEL_LENGTH = 60
            with pytest.raises(ValueError, match="too long"):
                reg.save_model(
                    model, scaler,
                    config={},
                    feature_names=[],
                    training_summary={},
                    label=long_label,
                )

    def test_label_auto_generated_if_omitted(self):
        """If no label provided, auto-generates from asset + architecture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model = _make_model()
            scaler = StandardScaler()
            scaler.fit(np.random.randn(20, INPUT_SIZE))

            model_id = reg.save_model(
                model, scaler,
                config={"asset": "BTC-USD", "architecture": "LSTM"},
                feature_names=[],
                training_summary={},
                # label omitted
            )

            _, _, meta = reg.load_model(
                model_id, TCNForecaster,
                input_size=INPUT_SIZE,
                hidden_size=32,
                num_layers=1,
                forecast_steps=FORECAST_STEPS,
                quantiles=QUANTILES,
            )
            assert "BTC-USD" in meta["label"]
            assert "LSTM" in meta["label"]

    def test_delete_all_models_requires_confirmation(self):
        """delete_all_models raises if confirmed=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            with pytest.raises(ValueError, match="confirmed=True"):
                reg.delete_all_models(confirmed=False)

    def test_delete_all_models_no_filter(self):
        """Delete all models without asset filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model = _make_model()
            scaler = StandardScaler()
            scaler.fit(np.random.randn(20, INPUT_SIZE))

            reg.save_model(model, scaler, {"asset": "GLD"}, [], {})
            reg.save_model(model, scaler, {"asset": "SLV"}, [], {})
            reg.save_model(model, scaler, {"asset": "BTC-USD"}, [], {})
            assert len(reg.list_models()) == 3

            deleted = reg.delete_all_models(confirmed=True)
            assert deleted == 3
            assert len(reg.list_models()) == 0

    def test_delete_all_models_with_asset_filter(self):
        """Delete all models for a specific asset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model = _make_model()
            scaler = StandardScaler()
            scaler.fit(np.random.randn(20, INPUT_SIZE))

            reg.save_model(model, scaler, {"asset": "GLD"}, [], {})
            reg.save_model(model, scaler, {"asset": "GLD"}, [], {})
            reg.save_model(model, scaler, {"asset": "SLV"}, [], {})
            assert len(reg.list_models()) == 3

            deleted = reg.delete_all_models(asset="GLD", confirmed=True)
            assert deleted == 2
            assert len(reg.list_models()) == 1
            assert reg.list_models()[0]["asset"] == "SLV"

    def test_delete_model_not_found(self):
        """delete_model raises FileNotFoundError for missing model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            with pytest.raises(FileNotFoundError, match="not found"):
                reg.delete_model("nonexistent_model_id")

