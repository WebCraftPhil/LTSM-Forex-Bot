"""
Logging utilities for the LSTM trading bot.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from .config import get_config

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)

def setup_logging(name: str = "lstm_trading_bot") -> logging.Logger:
    """Setup structured logging for the trading bot."""
    config = get_config()
    log_config = config.logging

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_config.level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    json_formatter = JSONFormatter()

    # File handler with rotation
    log_file = Path(log_config.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_config.max_size,
        backupCount=log_config.backup_count
    )
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    # Console handler for development
    if log_config.level.upper() == "DEBUG":
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger

class MetricsLogger:
    """Logger for tracking trading metrics and performance."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_trade(self, symbol: str, action: str, price: float,
                  quantity: float, pnl: float = 0.0, **kwargs):
        """Log individual trade execution."""
        extra_fields = {
            'event_type': 'trade',
            'symbol': symbol,
            'action': action,
            'price': price,
            'quantity': quantity,
            'pnl': pnl,
            **kwargs
        }

        self.logger.info(f"Trade: {action} {quantity} {symbol} @ {price}", extra_fields=extra_fields)

    def log_signal(self, symbol: str, signal: str, confidence: float,
                   price: float, **kwargs):
        """Log trading signal generation."""
        extra_fields = {
            'event_type': 'signal',
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'price': price,
            **kwargs
        }

        self.logger.info(f"Signal: {signal} {symbol} (conf: {confidence".3f"})", extra_fields=extra_fields)

    def log_model_metrics(self, epoch: int, train_loss: float,
                          val_loss: float, metrics: dict, **kwargs):
        """Log model training metrics."""
        extra_fields = {
            'event_type': 'model_metrics',
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics,
            **kwargs
        }

        metrics_str = ", ".join([f"{k}: {v".4f"}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: train_loss={train_loss".4f"}, val_loss={val_loss".4f"}, {metrics_str}",
                        extra_fields=extra_fields)

    def log_backtest_results(self, results: dict, **kwargs):
        """Log backtest performance results."""
        extra_fields = {
            'event_type': 'backtest_results',
            'results': results,
            **kwargs
        }

        self.logger.info(f"Backtest completed: {results.get('total_return', 0)".2%"} return, "
                       f"Sharpe: {results.get('sharpe_ratio', 0)".3f"}, "
                       f"Max DD: {results.get('max_drawdown', 0)".2%"}",
                       extra_fields=extra_fields)

    def log_risk_event(self, event_type: str, description: str, **kwargs):
        """Log risk management events."""
        extra_fields = {
            'event_type': 'risk_event',
            'risk_event_type': event_type,
            'description': description,
            **kwargs
        }

        self.logger.warning(f"Risk Event [{event_type}]: {description}", extra_fields=extra_fields)

    def log_system_event(self, event_type: str, description: str, **kwargs):
        """Log system-level events."""
        extra_fields = {
            'event_type': 'system_event',
            'system_event_type': event_type,
            'description': description,
            **kwargs
        }

        self.logger.info(f"System Event [{event_type}]: {description}", extra_fields=extra_fields)

def get_logger(name: str = "lstm_trading_bot") -> logging.Logger:
    """Get configured logger instance."""
    return logging.getLogger(name)

def get_metrics_logger(name: str = "lstm_trading_bot") -> MetricsLogger:
    """Get metrics logger instance."""
    logger = get_logger(name)
    return MetricsLogger(logger)
