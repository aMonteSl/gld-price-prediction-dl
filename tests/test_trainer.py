"""Tests for ModelTrainer — data prep, training, prediction, save/load."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from gldpred.models import GRURegressor, GRUClassifier, GRUMultiTask
from gldpred.training import ModelTrainer

from conftest import BATCH, HIDDEN, LAYERS, N_FEATURES, N_SAMPLES, SEQ_LEN


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

class TestPrepareData:
    def test_regression_loaders(self, synthetic_X, synthetic_y_reg):
        model = GRURegressor(N_FEATURES, HIDDEN, LAYERS)
        trainer = ModelTrainer(model, task="regression")
        tl, vl = trainer.prepare_data(synthetic_X, synthetic_y_reg, batch_size=BATCH)
        assert len(tl) > 0
        assert len(vl) > 0

    def test_classification_loaders(self, synthetic_X, synthetic_y_cls):
        model = GRUClassifier(N_FEATURES, HIDDEN, LAYERS)
        trainer = ModelTrainer(model, task="classification")
        tl, vl = trainer.prepare_data(synthetic_X, synthetic_y_cls, batch_size=BATCH)
        assert len(tl) > 0

    def test_multitask_loaders(self, synthetic_X, synthetic_y_mt):
        model = GRUMultiTask(N_FEATURES, HIDDEN, LAYERS)
        trainer = ModelTrainer(model, task="multitask")
        tl, vl = trainer.prepare_data(synthetic_X, synthetic_y_mt, batch_size=BATCH)
        # Each batch y should have 2 columns
        batch_X, batch_y = next(iter(tl))
        assert batch_y.shape[1] == 2

    def test_invalid_task_raises(self):
        model = GRURegressor(N_FEATURES, HIDDEN, LAYERS)
        with pytest.raises(ValueError, match="task must be"):
            ModelTrainer(model, task="invalid")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

class TestTraining:
    def test_regression_loss_decreases(self, synthetic_X, synthetic_y_reg):
        model = GRURegressor(N_FEATURES, HIDDEN, LAYERS)
        trainer = ModelTrainer(model, task="regression")
        tl, vl = trainer.prepare_data(synthetic_X, synthetic_y_reg, batch_size=BATCH)
        history = trainer.train(tl, vl, epochs=10, learning_rate=0.01)
        assert len(history["train_loss"]) == 10
        # Loss should generally decrease (allow some noise)
        assert history["train_loss"][-1] <= history["train_loss"][0] * 1.5

    def test_multitask_training(self, synthetic_X, synthetic_y_mt):
        model = GRUMultiTask(N_FEATURES, HIDDEN, LAYERS)
        trainer = ModelTrainer(model, task="multitask", w_reg=1.0, w_cls=1.0)
        tl, vl = trainer.prepare_data(synthetic_X, synthetic_y_mt, batch_size=BATCH)
        history = trainer.train(tl, vl, epochs=5)
        assert len(history["train_loss"]) == 5
        assert all(isinstance(v, float) for v in history["train_loss"])


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_regression_predict_shape(self, synthetic_X, synthetic_y_reg):
        model = GRURegressor(N_FEATURES, HIDDEN, LAYERS)
        trainer = ModelTrainer(model, task="regression")
        tl, vl = trainer.prepare_data(synthetic_X, synthetic_y_reg, batch_size=BATCH)
        trainer.train(tl, vl, epochs=2)
        preds = trainer.predict(synthetic_X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (N_SAMPLES,)

    def test_multitask_predict_returns_tuple(self, synthetic_X, synthetic_y_mt):
        model = GRUMultiTask(N_FEATURES, HIDDEN, LAYERS)
        trainer = ModelTrainer(model, task="multitask")
        tl, vl = trainer.prepare_data(synthetic_X, synthetic_y_mt, batch_size=BATCH)
        trainer.train(tl, vl, epochs=2)
        result = trainer.predict(synthetic_X)
        assert isinstance(result, tuple)
        reg_preds, cls_probs = result
        assert reg_preds.shape == (N_SAMPLES,)
        assert cls_probs.shape == (N_SAMPLES,)
        # cls_probs should be 0–1 (sigmoid applied)
        assert cls_probs.min() >= 0.0
        assert cls_probs.max() <= 1.0


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, synthetic_X, synthetic_y_reg):
        model = GRURegressor(N_FEATURES, HIDDEN, LAYERS)
        trainer = ModelTrainer(model, task="regression")
        tl, vl = trainer.prepare_data(synthetic_X, synthetic_y_reg, batch_size=BATCH)
        trainer.train(tl, vl, epochs=2)
        preds_before = trainer.predict(synthetic_X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pth")
            trainer.save_model(path)

            # Create a fresh model + trainer and load
            model2 = GRURegressor(N_FEATURES, HIDDEN, LAYERS)
            trainer2 = ModelTrainer(model2, task="regression")
            trainer2.load_model(path)
            preds_after = trainer2.predict(synthetic_X)

        np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)
