"""Smart configuration suggester based on asset characteristics."""
from __future__ import annotations

from typing import Dict, Any

import pandas as pd


# Valid options for UI sliders (must match components.py)
VALID_HIDDEN_SIZE = [32, 48, 64, 96, 128]
VALID_BATCH_SIZE = [16, 32, 64, 128, 256]
VALID_LR = [0.0001, 0.0005, 0.001, 0.005, 0.01]


def _nearest_valid(value: int, options: list) -> int:
    """Return nearest valid option from list."""
    return min(options, key=lambda x: abs(x - value))


def _nearest_valid_float(value: float, options: list) -> float:
    """Return nearest valid option from list (for floats)."""
    return min(options, key=lambda x: abs(x - value))


def suggest_training_config(asset: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Suggest training hyperparameters based on asset and recent volatility.
    
    Args:
        asset: Asset ticker (GLD, SLV, BTC-USD, PALL).
        df: Historical price dataframe with technical indicators.
    
    Returns:
        Dict with suggested hyperparameters: seq_length, hidden_size, epochs, batch_size, learning_rate.
    """
    # Base configurations per asset type
    base_configs = {
        "GLD": {
            "seq_length": 20,
            "hidden_size": 64,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_layers": 2,
        },
        "SLV": {
            "seq_length": 20,
            "hidden_size": 64,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_layers": 2,
        },
        "BTC-USD": {
            "seq_length": 30,       # Crypto: longer memory
            "hidden_size": 96,      # Crypto: more capacity
            "epochs": 100,          # Crypto: needs more training
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_layers": 2,
        },
        "PALL": {
            "seq_length": 20,
            "hidden_size": 64,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_layers": 2,
        },
    }
    
    # Get base config or default to GLD
    config = base_configs.get(asset, base_configs["GLD"]).copy()
    
    # Adjust based on historical volatility
    if not df.empty and "daily_change_pct" in df.columns:
        recent_vol = df["daily_change_pct"].tail(60).std()  # last 60 days volatility
        
        if recent_vol > 2.0:
            # High volatility: more capacity + longer training
            config["hidden_size"] = _nearest_valid(
                min(128, config["hidden_size"] + 32), VALID_HIDDEN_SIZE
            )
            config["epochs"] = min(150, config["epochs"] + 30)
            config["seq_length"] = min(40, config["seq_length"] + 10)
        elif recent_vol < 0.5:
            # Low volatility: simpler model can suffice
            config["hidden_size"] = _nearest_valid(
                max(32, config["hidden_size"] - 16), VALID_HIDDEN_SIZE
            )
            config["epochs"] = max(30, config["epochs"] - 10)
    
    # Adjust based on dataset size
    n_records = len(df)
    if n_records < 1000:
        # Small dataset: reduce capacity to avoid overfit
        config["hidden_size"] = _nearest_valid(
            max(32, config["hidden_size"] // 2), VALID_HIDDEN_SIZE
        )
        config["epochs"] = max(20, config["epochs"] // 2)
        config["seq_length"] = max(10, config["seq_length"] - 5)
    elif n_records > 5000:
        # Large dataset: can afford bigger model
        config["hidden_size"] = _nearest_valid(
            min(128, config["hidden_size"] + 16), VALID_HIDDEN_SIZE
        )
        config["batch_size"] = _nearest_valid(
            min(64, config["batch_size"] * 2), VALID_BATCH_SIZE
        )
    
    # Ensure all values are valid for UI sliders
    config["hidden_size"] = _nearest_valid(config["hidden_size"], VALID_HIDDEN_SIZE)
    config["batch_size"] = _nearest_valid(config["batch_size"], VALID_BATCH_SIZE)
    config["learning_rate"] = _nearest_valid_float(config["learning_rate"], VALID_LR)
    
    return config


def get_config_rationale(asset: str, df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """Generate human-readable explanation for suggested config.
    
    Args:
        asset: Asset ticker.
        df: Historical dataframe.
        config: Suggested configuration.
    
    Returns:
        Explanation string.
    """
    n = len(df)
    vol = df["daily_change_pct"].tail(60).std() if not df.empty and "daily_change_pct" in df.columns else 1.0
    
    parts = []
    parts.append(f"Asset: {asset}")
    parts.append(f"Records: {n}")
    parts.append(f"Volatility (60d std): {vol:.2f}%")
    
    if vol > 2.0:
        parts.append("→ Alta volatilidad detectada: modelo más complejo")
    elif vol < 0.5:
        parts.append("→ Baja volatilidad detectada: modelo simplificado")
    
    if n < 1000:
        parts.append("→ Dataset pequeño: reducir capacidad para evitar overfit")
    elif n > 5000:
        parts.append("→ Dataset grande: modelo más grande permitido")
    
    return " | ".join(parts)
