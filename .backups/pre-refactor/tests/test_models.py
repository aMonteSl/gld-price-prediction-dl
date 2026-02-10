"""Tests for GRUForecaster, LSTMForecaster, TCNForecaster."""
from __future__ import annotations

import pytest
import torch

from gldpred.models import GRUForecaster, LSTMForecaster, TCNForecaster
from conftest import BATCH_SIZE, FORECAST_STEPS, INPUT_SIZE, NUM_QUANTILES, QUANTILES, SEQ_LENGTH

MODELS = [
    ("GRU", GRUForecaster),
    ("LSTM", LSTMForecaster),
    ("TCN", TCNForecaster),
]


@pytest.mark.parametrize("name,cls", MODELS)
def test_output_shape(name, cls):
    """Forward pass produces (B, K, Q) output."""
    model = cls(
        input_size=INPUT_SIZE,
        hidden_size=32,
        num_layers=1,
        forecast_steps=FORECAST_STEPS,
        quantiles=QUANTILES,
    )
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    out = model(x)
    assert out.shape == (BATCH_SIZE, FORECAST_STEPS, NUM_QUANTILES)


@pytest.mark.parametrize("name,cls", MODELS)
def test_gradient_flow(name, cls):
    """Gradients flow to all parameters."""
    model = cls(
        input_size=INPUT_SIZE,
        hidden_size=32,
        num_layers=1,
        forecast_steps=FORECAST_STEPS,
        quantiles=QUANTILES,
    )
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    out = model(x)
    loss = out.sum()
    loss.backward()

    for pname, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {pname}"
        assert not torch.all(p.grad == 0), f"Zero gradient for {pname}"


@pytest.mark.parametrize("name,cls", MODELS)
def test_different_forecast_steps(name, cls):
    """Models work with different K values."""
    for k in [1, 3, 10]:
        model = cls(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            forecast_steps=k,
            quantiles=QUANTILES,
        )
        x = torch.randn(2, SEQ_LENGTH, INPUT_SIZE)
        out = model(x)
        assert out.shape == (2, k, NUM_QUANTILES)


@pytest.mark.parametrize("name,cls", MODELS)
def test_different_quantiles(name, cls):
    """Models work with different numbers of quantiles."""
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    model = cls(
        input_size=INPUT_SIZE,
        hidden_size=32,
        num_layers=1,
        forecast_steps=FORECAST_STEPS,
        quantiles=qs,
    )
    x = torch.randn(2, SEQ_LENGTH, INPUT_SIZE)
    out = model(x)
    assert out.shape == (2, FORECAST_STEPS, len(qs))


@pytest.mark.parametrize("name,cls", MODELS)
def test_eval_no_grad(name, cls):
    """Inference mode works and produces finite values."""
    model = cls(
        input_size=INPUT_SIZE,
        hidden_size=32,
        num_layers=1,
        forecast_steps=FORECAST_STEPS,
        quantiles=QUANTILES,
    )
    model.eval()
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name,cls", MODELS)
def test_single_sample(name, cls):
    """Models handle batch size 1."""
    model = cls(
        input_size=INPUT_SIZE,
        hidden_size=32,
        num_layers=1,
        forecast_steps=FORECAST_STEPS,
        quantiles=QUANTILES,
    )
    x = torch.randn(1, SEQ_LENGTH, INPUT_SIZE)
    out = model(x)
    assert out.shape == (1, FORECAST_STEPS, NUM_QUANTILES)
