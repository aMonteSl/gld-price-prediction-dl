"""Configuration dataclasses for the GLD price prediction pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for data loading."""

    ticker: str = "GLD"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.end_date is None:
            self.end_date = datetime.now()
        if self.start_date is None:
            self.start_date = self.end_date - timedelta(days=365 * 5)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    model_type: str = "GRU"           # "GRU", "LSTM", or "TCN"
    task: str = "regression"          # "regression", "classification", or "multitask"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    seq_length: int = 20
    horizon: int = 5                  # 1, 5, or 20 days
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    test_size: float = 0.2
    # Multi-task loss weights
    w_reg: float = 1.0
    w_cls: float = 1.0
    # Classification buy threshold (for multi-task & classification labels)
    buy_threshold: float = 0.003


@dataclass
class AppConfig:
    """Top-level configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
