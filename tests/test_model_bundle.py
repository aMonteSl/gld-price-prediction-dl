"""Tests for ModelBundle and load_bundle (registry-backed inference)."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from gldpred.models import GRUForecaster, LSTMForecaster, TCNForecaster
from gldpred.registry import ModelBundle, ModelRegistry
from conftest import BATCH_SIZE, FORECAST_STEPS, INPUT_SIZE, QUANTILES, SEQ_LENGTH


def _make_model(arch_cls=TCNForecaster):
    return arch_cls(
        input_size=INPUT_SIZE,
        hidden_size=32,
        num_layers=1,
        forecast_steps=FORECAST_STEPS,
        quantiles=QUANTILES,
    )


def _make_scaler():
    scaler = StandardScaler()
    scaler.fit(np.random.randn(20, INPUT_SIZE))
    return scaler


def _save_model(reg, arch_name="TCN", arch_cls=TCNForecaster, asset="GLD"):
    """Save a model to the registry and return model_id."""
    model = _make_model(arch_cls)
    scaler = _make_scaler()
    model_id = reg.save_model(
        model, scaler,
        config={
            "asset": asset,
            "architecture": arch_name,
            "input_size": INPUT_SIZE,
            "hidden_size": 32,
            "num_layers": 1,
            "forecast_steps": FORECAST_STEPS,
            "quantiles": list(QUANTILES),
            "seq_length": SEQ_LENGTH,
        },
        feature_names=[f"feat_{i}" for i in range(INPUT_SIZE)],
        training_summary={"epochs": 10, "final_val_loss": 0.01},
    )
    return model_id


class TestModelBundle:
    """Test ModelBundle duck-type contract."""

    def test_predict_output_shape(self):
        model = _make_model()
        scaler = _make_scaler()
        bundle = ModelBundle(
            model=model,
            scaler=scaler,
            metadata={"architecture": "TCN"},
            model_id="test-id",
            label="Test Model",
            asset="GLD",
            architecture="TCN",
            quantiles_tuple=QUANTILES,
        )
        X = np.random.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE).astype(np.float32)
        out = bundle.predict(X)
        assert out.shape == (BATCH_SIZE, FORECAST_STEPS, len(QUANTILES))

    def test_quantiles_property(self):
        bundle = ModelBundle(
            model=_make_model(),
            scaler=_make_scaler(),
            metadata={},
            model_id="test",
            label="test",
            asset="GLD",
            architecture="TCN",
            quantiles_tuple=QUANTILES,
        )
        assert bundle.quantiles == QUANTILES

    def test_device_property(self):
        bundle = ModelBundle(
            model=_make_model(),
            scaler=_make_scaler(),
            metadata={},
            model_id="test",
            label="test",
            asset="GLD",
            architecture="TCN",
        )
        assert bundle.device in (torch.device("cpu"), torch.device("cuda", 0))


class TestLoadBundle:
    """Test ModelRegistry.load_bundle()."""

    @pytest.mark.parametrize("arch_name,arch_cls", [
        ("TCN", TCNForecaster),
        ("GRU", GRUForecaster),
        ("LSTM", LSTMForecaster),
    ])
    def test_load_bundle_roundtrip(self, arch_name, arch_cls):
        """Save a model, load_bundle it, verify predict works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model_id = _save_model(reg, arch_name, arch_cls)

            bundle = reg.load_bundle(model_id)
            assert isinstance(bundle, ModelBundle)
            assert bundle.model_id == model_id
            assert bundle.architecture == arch_name
            assert bundle.asset == "GLD"
            assert bundle.quantiles_tuple == QUANTILES

            # Verify predict works
            X = np.random.randn(2, SEQ_LENGTH, INPUT_SIZE).astype(np.float32)
            out = bundle.predict(X)
            assert out.shape == (2, FORECAST_STEPS, len(QUANTILES))

    def test_load_bundle_preserves_weights(self):
        """Verify loaded model produces same output as original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model = _make_model()
            scaler = _make_scaler()

            # Run original prediction
            X = np.random.randn(1, SEQ_LENGTH, INPUT_SIZE).astype(np.float32)
            model.eval()
            n, s, f = X.shape
            X_s = scaler.transform(X.reshape(-1, f)).reshape(n, s, f)
            X_t = torch.FloatTensor(X_s)
            with torch.no_grad():
                original_out = model(X_t).numpy()

            # Save and reload
            model_id = reg.save_model(
                model, scaler,
                config={
                    "asset": "GLD",
                    "architecture": "TCN",
                    "input_size": INPUT_SIZE,
                    "hidden_size": 32,
                    "num_layers": 1,
                    "forecast_steps": FORECAST_STEPS,
                    "quantiles": list(QUANTILES),
                },
                feature_names=[f"f{i}" for i in range(INPUT_SIZE)],
                training_summary={"epochs": 5, "final_val_loss": 0.02},
            )

            bundle = reg.load_bundle(model_id)
            loaded_out = bundle.predict(X)

            np.testing.assert_allclose(original_out, loaded_out, atol=1e-5)

    def test_load_bundle_feature_names(self):
        """Verify feature names are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            names = [f"feat_{i}" for i in range(INPUT_SIZE)]
            model_id = _save_model(reg)
            bundle = reg.load_bundle(model_id)
            assert bundle.feature_names == names

    def test_load_bundle_config(self):
        """Verify config dict is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model_id = _save_model(reg)
            bundle = reg.load_bundle(model_id)
            assert bundle.config["seq_length"] == SEQ_LENGTH
            assert bundle.config["forecast_steps"] == FORECAST_STEPS

    def test_load_bundle_invalid_id(self):
        """load_bundle raises FileNotFoundError for missing model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            with pytest.raises(FileNotFoundError):
                reg.load_bundle("nonexistent-model-id")

    def test_load_bundle_label(self):
        """Verify label is set from metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ModelRegistry(base_dir=tmpdir)
            model_id = _save_model(reg, asset="BTC-USD")
            bundle = reg.load_bundle(model_id)
            assert bundle.asset == "BTC-USD"
            assert isinstance(bundle.label, str)
            assert len(bundle.label) > 0
