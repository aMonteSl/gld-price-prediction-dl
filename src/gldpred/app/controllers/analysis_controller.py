"""Analysis controller for feature importance and model insights."""
from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np

from gldpred.training import ModelTrainer


def compute_feature_importance_ablation(
    trainer: ModelTrainer,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
) -> List[Dict[str, float]]:
    """Compute feature importance via ablation (zeroing out).
    
    Args:
        trainer: Trained ModelTrainer instance.
        X_val: Validation features (N, seq_len, F).
        y_val: Validation targets (N, K).
        feature_names: List of feature column names.
        top_k: Number of top features to return.
    
    Returns:
        List of dicts with keys: feature, importance, rank.
        Higher importance = bigger loss increase when feature is removed.
    """
    # Baseline: loss with all features
    baseline_preds = trainer.predict(X_val)
    baseline_loss = float(
        np.mean((y_val[:, :, np.newaxis] - baseline_preds) ** 2)
    )
    
    importances = []
    for feat_idx in range(X_val.shape[-1]):
        # Ablate: zero out this feature
        X_ablated = X_val.copy()
        X_ablated[:, :, feat_idx] = 0
        
        ablated_preds = trainer.predict(X_ablated)
        ablated_loss = float(
            np.mean((y_val[:, :, np.newaxis] - ablated_preds) ** 2)
        )
        
        importance = ablated_loss - baseline_loss
        importances.append({
            "feature": feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat_{feat_idx}",
            "importance": importance,
        })
    
    # Sort descending by importance
    importances.sort(key=lambda x: abs(x["importance"]), reverse=True)
    
    # Add rank
    for rank, item in enumerate(importances[:top_k], start=1):
        item["rank"] = rank
    
    return importances[:top_k]


def compute_feature_importance_gradient(
    trainer: Any,
    X_val: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
) -> List[Dict[str, float]]:
    """Compute feature importance via input gradients (faster, approximate).
    
    Args:
        trainer: Trained ModelTrainer instance.
        X_val: Validation features (N, seq_len, F).
        feature_names: List of feature column names.
        top_k: Number of top features to return.
    
    Returns:
        List of dicts with keys: feature, importance, rank.
        Based on mean absolute gradient w.r.t. input features.
    """
    import torch
    
    trainer.model.eval()
    n, s, f = X_val.shape
    X_s = trainer.scaler.transform(X_val.reshape(-1, f)).reshape(n, s, f)
    X_t = torch.FloatTensor(X_s).to(trainer.device)
    X_t.requires_grad = True
    
    # Forward pass
    out = trainer.model(X_t)
    
    # Compute gradients w.r.t. input
    grad_outputs = torch.ones_like(out)
    grads = torch.autograd.grad(
        outputs=out,
        inputs=X_t,
        grad_outputs=grad_outputs,
        create_graph=False,
    )[0]
    
    # Mean absolute gradient per feature (averaged across samples and time)
    grad_importance = grads.abs().mean(dim=(0, 1)).cpu().detach().numpy()
    
    importances = []
    for feat_idx in range(f):
        importances.append({
            "feature": feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat_{feat_idx}",
            "importance": float(grad_importance[feat_idx]),
        })
    
    # Sort descending
    importances.sort(key=lambda x: x["importance"], reverse=True)
    
    # Add rank
    for rank, item in enumerate(importances[:top_k], start=1):
        item["rank"] = rank
    
    return importances[:top_k]
