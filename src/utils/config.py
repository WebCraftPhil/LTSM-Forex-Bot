"""
Configuration utilities for the LSTM trading bot.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["EURUSD"])
    timeframes: list[str] = Field(default_factory=lambda: ["15m", "30m", "1h", "2h"])
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    data_source: str = "csv"
    data_dir: str = "data/raw"
    cache_dir: str = "data/cache"

    features: Dict[str, Any] = Field(default_factory=dict)
    technical_indicators: list[str] = Field(default_factory=list)
    lagged_features: Dict[str, list[int]] = Field(default_factory=dict)
    calendar_features: list[str] = Field(default_factory=list)
    regime_features: bool = False
    n_regimes: int = 3

class ModelConfig(BaseModel):
    architecture: str = "multi_lstm_fusion"
    input_size: int = 64
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 60
    fusion_strategy: str = "concatenate"
    output_mode: str = "classification"
    num_classes: int = 3
    use_layer_norm: bool = True
    use_attention: bool = False

class TrainingConfig(BaseModel):
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    loss_function: str = "focal_loss"
    label_smoothing: float = 0.1
    optimizer: str = "adam"
    scheduler: str = "cosine_annealing"
    patience: int = 10
    min_delta: float = 0.001
    optuna_trials: int = 50
    optuna_timeout: int = 3600

class BacktestConfig(BaseModel):
    broker: str = "backtrader"
    commission: float = 0.0002
    slippage: float = 0.0001
    stake: float = 10000.0

    risk: Dict[str, Any] = Field(default_factory=dict)
    max_position_size: float = 0.02
    max_positions: int = 5
    stop_loss: float = 0.02
    take_profit: float = 0.04
    trailing_stop: bool = True
    circuit_breaker_drawdown: float = 0.15

    warmup_periods: int = 100
    benchmark: str = "buy_and_hold"

class LiveConfig(BaseModel):
    mode: str = "paper"
    broker: str = "alpaca"
    realtime_data_provider: str = "alpaca"
    reconnect_attempts: int = 5
    reconnect_delay: int = 5
    rate_limit_delay: float = 0.1
    execution_delay: int = 1
    order_timeout: int = 30
    daily_loss_limit: float = 0.05
    max_drawdown: float = 0.20

    enable_alerts: bool = False
    alert_webhook: str = ""
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)

class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/trading_bot.log"
    max_size: int = 10485760  # 10MB
    backup_count: int = 5
    metrics: list[str] = Field(default_factory=list)

class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    backtest: BacktestConfig
    live: LiveConfig
    logging: LoggingConfig

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)

def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    config_dict = config.dict()

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

def get_env_var(key: str, default: Any = None) -> Any:
    """Get environment variable with optional default."""
    return os.getenv(key, default)

# Global config instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
