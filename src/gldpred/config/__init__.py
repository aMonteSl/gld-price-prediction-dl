"""Configuration dataclasses for the GLD forecasting pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from gldpred.config.assets import ASSET_CATALOG, AssetInfo, get_asset_info, supported_tickers

SUPPORTED_ASSETS = ("GLD", "SLV", "BTC-USD", "PALL")


@dataclass
class DataConfig:
    """Configuration for data loading.

    Data is loaded from the asset's first available date to today (auto).
    """

    ticker: str = "GLD"


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    architecture: str = "TCN"  # "GRU", "LSTM", or "TCN"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    forecast_steps: int = 20
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    seq_length: int = 20
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    test_size: float = 0.2


@dataclass
class DecisionConfig:
    """Configuration for the recommendation engine."""

    horizon_days: int = 5
    min_expected_return: float = 0.008  # 0.8 % over decision window
    trend_sma_short: int = 50
    trend_sma_long: int = 200
    max_volatility: dict = field(
        default_factory=lambda: {"default": 0.02, "BTC-USD": 0.05}
    )
    model_health_gate: bool = True


@dataclass
class AppConfig:
    """Top-level configuration aggregating all sub-configs."""

    page_title: str = "Multi-Asset Price Forecasting"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
