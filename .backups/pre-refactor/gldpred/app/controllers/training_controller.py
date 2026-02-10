"""Training controller for Streamlit orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Type

import torch.nn as nn

from gldpred.app import state
from gldpred.app.controllers.model_loader import activate_after_training
from gldpred.diagnostics import DiagnosticsAnalyzer
from gldpred.models import TCNForecaster
from gldpred.registry import ModelRegistry
from gldpred.services.evaluation_service import evaluate_on_validation
from gldpred.services.training_service import build_sequences, train_model
from gldpred.training import ModelTrainer


@dataclass
class TrainingResult:
    """Result payload for a completed training run."""

    model_id: str
    label: str


def run_training(
    *,
    df,
    feature_names,
    daily_returns,
    arch_map: Dict[str, Type[nn.Module]],
    arch_name: str,
    hidden_size: int,
    num_layers: int,
    forecast_steps: int,
    seq_length: int,
    quantiles: Tuple[float, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    is_finetune: bool,
    base_model_id: Optional[str],
    label_input: str,
    on_epoch: Callable,
    early_stopping: bool = False,
    patience: int = 5,
) -> TrainingResult:
    """Run training or fine-tuning and update session state."""
    model_cls = arch_map.get(arch_name, TCNForecaster)

    if is_finetune and base_model_id:
        reg = ModelRegistry()
        base_meta = next(
            (m for m in reg.list_models() if m["model_id"] == base_model_id),
            None,
        )
        if base_meta is None:
            raise ValueError("Model not found in registry.")

        cfg = base_meta.get("config", {})
        X, y = build_sequences(
            df,
            daily_returns,
            seq_length=cfg.get("seq_length", 20),
            forecast_steps=cfg.get("forecast_steps", 20),
        )

        base_cls = arch_map.get(cfg.get("architecture", "TCN"), TCNForecaster)
        model, scaler, _ = reg.load_model(
            base_model_id,
            base_cls,
            input_size=len(feature_names),
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_layers", 2),
            forecast_steps=cfg.get("forecast_steps", 20),
            quantiles=tuple(cfg.get("quantiles", [0.1, 0.5, 0.9])),
        )

        expected_features = len(base_meta.get("feature_names", []))
        if expected_features and expected_features != len(feature_names):
            raise ValueError(
                f"Feature mismatch: expected {expected_features}, got {len(feature_names)}"
            )

        q = tuple(cfg.get("quantiles", [0.1, 0.5, 0.9]))
        trainer = ModelTrainer(model, quantiles=q, device="cpu")
        trainer.scaler = scaler
        history = train_model(
            trainer,
            X,
            y,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            on_epoch=on_epoch,
            refit_scaler=False,
            early_stopping=early_stopping,
            patience=patience,
        )
        train_losses = history["train_loss"]
        val_losses = history["val_loss"]
        quantiles_list = list(q)
        used_arch = cfg.get("architecture", arch_name)
        used_hidden = cfg.get("hidden_size", hidden_size)
        used_layers = cfg.get("num_layers", num_layers)
        used_steps = cfg.get("forecast_steps", forecast_steps)
        used_seq = cfg.get("seq_length", seq_length)
    else:
        X, y = build_sequences(
            df,
            daily_returns,
            seq_length=seq_length,
            forecast_steps=forecast_steps,
        )
        model = model_cls(
            input_size=len(feature_names),
            hidden_size=hidden_size,
            num_layers=num_layers,
            forecast_steps=forecast_steps,
            quantiles=quantiles,
        )
        trainer = ModelTrainer(model, quantiles=quantiles, device="cpu")
        history = train_model(
            trainer,
            X,
            y,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            on_epoch=on_epoch,
            refit_scaler=True,
            early_stopping=early_stopping,
            patience=patience,
        )
        train_losses = history["train_loss"]
        val_losses = history["val_loss"]
        quantiles_list = list(quantiles)
        used_arch = arch_name
        used_hidden = hidden_size
        used_layers = num_layers
        used_steps = forecast_steps
        used_seq = seq_length

    state.put(state.KEY_TRAINER, trainer)
    state.put(state.KEY_TRAIN_LOSSES, train_losses)
    state.put(state.KEY_VAL_LOSSES, val_losses)
    state.put(state.KEY_SUGGESTIONS_APPLIED, False)

    diag = DiagnosticsAnalyzer.analyze({
        "train_loss": train_losses,
        "val_loss": val_losses,
    })
    state.put(state.KEY_DIAG_RESULT, diag)

    traj_metrics, quant_metrics = evaluate_on_validation(
        trainer, X, y, quantiles_list
    )
    state.put(state.KEY_TRAJ_METRICS, traj_metrics)
    state.put(state.KEY_QUANT_METRICS, quant_metrics)

    config_dict = {
        "asset": state.get(state.KEY_ASSET, "GLD"),
        "architecture": used_arch,
        "hidden_size": used_hidden,
        "num_layers": used_layers,
        "forecast_steps": used_steps,
        "quantiles": list(quantiles),
        "seq_length": used_seq,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    training_summary = {
        "epochs": epochs,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "diagnostics_verdict": diag.verdict if diag else None,
    }
    eval_summary = {**(traj_metrics or {}), **(quant_metrics or {})}

    reg = ModelRegistry()
    model_id = reg.save_model(
        model=trainer.model,
        scaler=trainer.scaler,
        config=config_dict,
        feature_names=feature_names,
        training_summary=training_summary,
        evaluation_summary=eval_summary,
        label=label_input.strip() or None,
    )
    state.put(state.KEY_LAST_MODEL_ID, model_id)

    # Set the just-trained model as the active model for inference
    activate_after_training()

    meta = next((m for m in reg.list_models() if m["model_id"] == model_id), {})
    saved_label = meta.get("label", model_id)
    return TrainingResult(model_id=model_id, label=saved_label)
