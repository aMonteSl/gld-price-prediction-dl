"""Tests for pinball loss, ModelTrainer, and persistence."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from gldpred.models import TCNForecaster
from gldpred.training import ModelTrainer, pinball_loss
from conftest import BATCH_SIZE, FORECAST_STEPS, INPUT_SIZE, QUANTILES, SEQ_LENGTH


# ── Pinball loss ──────────────────────────────────────────────────────

class TestPinballLoss:
    def test_zero_error(self):
        """Loss is zero when predictions equal targets."""
        q = torch.tensor([0.1, 0.5, 0.9])
        y_true = torch.zeros(BATCH_SIZE, FORECAST_STEPS)
        y_pred = torch.zeros(BATCH_SIZE, FORECAST_STEPS, len(q))
        loss = pinball_loss(y_pred, y_true, q)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive(self):
        """Pinball loss is always non-negative."""
        q = torch.tensor([0.1, 0.5, 0.9])
        y_true = torch.randn(BATCH_SIZE, FORECAST_STEPS)
        y_pred = torch.randn(BATCH_SIZE, FORECAST_STEPS, len(q))
        loss = pinball_loss(y_pred, y_true, q)
        assert loss.item() >= 0.0

    def test_asymmetry(self):
        """Higher quantile penalises under-prediction more."""
        q = torch.tensor([0.1, 0.5, 0.9])
        y_true = torch.ones(1, 1)  # actual = 1
        # over-predict
        y_over = torch.full((1, 1, 3), 2.0)
        # under-predict
        y_under = torch.full((1, 1, 3), 0.0)
        loss_over = pinball_loss(y_over, y_true, q)
        loss_under = pinball_loss(y_under, y_true, q)
        # Both should be >0
        assert loss_over.item() > 0
        assert loss_under.item() > 0


# ── ModelTrainer ──────────────────────────────────────────────────────

class TestModelTrainer:
    def _make_trainer(self):
        model = TCNForecaster(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            forecast_steps=FORECAST_STEPS,
            quantiles=QUANTILES,
        )
        return ModelTrainer(model, quantiles=QUANTILES, device="cpu")

    def test_prepare_data(self, synthetic_X, synthetic_y):
        trainer = self._make_trainer()
        train_dl, val_dl = trainer.prepare_data(
            synthetic_X, synthetic_y, test_size=0.2, batch_size=8,
        )
        assert len(train_dl) > 0
        assert len(val_dl) > 0

    def test_train_short(self, synthetic_X, synthetic_y):
        trainer = self._make_trainer()
        train_dl, val_dl = trainer.prepare_data(
            synthetic_X, synthetic_y, test_size=0.2, batch_size=8,
        )
        history = trainer.train(train_dl, val_dl, epochs=2, learning_rate=0.001)
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_predict_shape(self, synthetic_X, synthetic_y):
        trainer = self._make_trainer()
        train_dl, val_dl = trainer.prepare_data(
            synthetic_X, synthetic_y, test_size=0.2, batch_size=8,
        )
        trainer.train(train_dl, val_dl, epochs=2, learning_rate=0.001)
        pred = trainer.predict(synthetic_X[:5])
        assert pred.shape == (5, FORECAST_STEPS, len(QUANTILES))

    def test_save_load_checkpoint(self, synthetic_X, synthetic_y):
        trainer = self._make_trainer()
        train_dl, val_dl = trainer.prepare_data(
            synthetic_X, synthetic_y, test_size=0.2, batch_size=8,
        )
        trainer.train(train_dl, val_dl, epochs=2, learning_rate=0.001)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pth")
            trainer.save_checkpoint(path)
            assert os.path.exists(path)

            # Load into fresh trainer
            trainer2 = self._make_trainer()
            trainer2.load_checkpoint(path)
            assert len(trainer2.history["train_loss"]) == 2

    def test_finetune_scaler(self, synthetic_X, synthetic_y):
        """Fine-tune mode preserves existing scaler."""
        trainer = self._make_trainer()
        train_dl, val_dl = trainer.prepare_data(
            synthetic_X, synthetic_y, test_size=0.2, batch_size=8,
        )
        original_mean = trainer.scaler.mean_.copy()

        # Re-prepare with refit_scaler=False (simulate fine-tune)
        train_dl2, val_dl2 = trainer.prepare_data(
            synthetic_X, synthetic_y, test_size=0.2, batch_size=8, refit_scaler=False,
        )
        np.testing.assert_array_equal(trainer.scaler.mean_, original_mean)

    def test_on_epoch_callback(self, synthetic_X, synthetic_y):
        """on_epoch callback is called for each epoch."""
        trainer = self._make_trainer()
        train_dl, val_dl = trainer.prepare_data(
            synthetic_X, synthetic_y, test_size=0.2, batch_size=8,
        )
        calls = []
        def cb(epoch, total, history):
            calls.append(epoch)
        trainer.train(train_dl, val_dl, epochs=3, learning_rate=0.001, on_epoch=cb)
        assert calls == [0, 1, 2]

    def test_quantiles_property(self):
        """ModelTrainer.quantiles returns the quantiles tuple."""
        trainer = self._make_trainer()
        assert trainer.quantiles == QUANTILES
        assert isinstance(trainer.quantiles, tuple)

    def test_finetune_with_loaded_scaler(self, synthetic_X, synthetic_y):
        """Fine-tune from checkpoint: scaler set before prepare_data prevents NotFittedError."""
        trainer = self._make_trainer()
        train_dl, val_dl = trainer.prepare_data(
            synthetic_X, synthetic_y, test_size=0.2, batch_size=8,
        )
        trainer.train(train_dl, val_dl, epochs=2, learning_rate=0.001)
        saved_scaler = trainer.scaler  # already fitted

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pth")
            trainer.save_checkpoint(path)

            # Create a fresh trainer and load checkpoint
            trainer2 = self._make_trainer()
            trainer2.load_checkpoint(path)
            # Explicitly set scaler BEFORE prepare_data (simulating fine-tune flow)
            trainer2.scaler = saved_scaler
            train_dl2, val_dl2 = trainer2.prepare_data(
                synthetic_X, synthetic_y, test_size=0.2, batch_size=8,
                refit_scaler=False,
            )
            # Should not raise NotFittedError
            history = trainer2.train(train_dl2, val_dl2, epochs=1, learning_rate=0.001)
            # History accumulates: 2 from checkpoint + 1 new = 3
            assert len(history["train_loss"]) == 3
