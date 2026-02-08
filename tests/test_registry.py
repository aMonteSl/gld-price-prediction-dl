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
