# LSTM Multi-Timeframe Trading Bot

A production-ready LSTM-based trading bot for forex and cryptocurrency markets with multi-timeframe inputs, deep learning models, comprehensive backtesting, and live execution capabilities.

## 🚀 Features

- **Multi-Timeframe Analysis**: Combines 15m, 30m, 1h, and 2h timeframes for enhanced signal generation
- **Advanced LSTM Architectures**: Single LSTM and multi-LSTM fusion strategies with attention mechanisms
- **Comprehensive Feature Engineering**: Technical indicators, lagged features, calendar features, and regime detection
- **Robust Training Framework**: Optuna hyperparameter optimization with walk-forward validation
- **Realistic Backtesting**: Backtrader integration with slippage, commission, and risk management
- **Live Execution**: Paper and live trading modes with WebSocket streaming and broker abstraction
- **Risk Management**: Position sizing, stop-loss, circuit breakers, and drawdown protection
- **Production Ready**: Structured logging, configuration management, and deployment scripts

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Backtesting](#backtesting)
- [Live Trading](#live-trading)
- [Risk Management](#risk-management)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🛠 Installation

### Prerequisites

- Python 3.10+
- GPU support (optional, for faster training)
- Broker API keys (for live trading)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/lstm-trading-bot.git
   cd lstm-trading-bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## 🚀 Quick Start

### Research Mode (Google Colab)

1. Open the [Colab notebook](./notebooks/01_data_eda.ipynb)
2. Follow the step-by-step guide to:
   - Download historical data
   - Build multi-timeframe features
   - Train LSTM models with Optuna
   - Backtest strategies
   - Generate performance reports

### Local Development

```bash
# Download historical data
python -m src.data.loaders --symbols EURUSD,BTCUSD --start 2020-01-01 --end 2024-12-31

# Build features
python -m src.features.build_dataset --config config/config.yaml

# Train model with hyperparameter optimization
python -m src.training.train --config config/config.yaml --optuna 50

# Backtest strategy
python -m src.backtest.engine --config config/config.yaml --report reports/backtest.html

# Paper trading
python -m src.live.executor --config config/config.yaml --mode paper
```

## ⚙️ Configuration

The bot uses a comprehensive YAML configuration system. Key sections:

```yaml
# Data Configuration
data:
  symbols: ["EURUSD", "BTCUSD"]
  timeframes: ["15m", "30m", "1h", "2h"]
  data_source: "csv"  # csv, alpaca, binance, oanda

# Model Architecture
model:
  architecture: "multi_lstm_fusion"  # single_lstm, multi_lstm_fusion
  hidden_size: 128
  num_layers: 2
  dropout: 0.2

# Training Parameters
training:
  batch_size: 32
  num_epochs: 100
  optuna_trials: 50

# Risk Management
backtest:
  risk:
    max_position_size: 0.02
    stop_loss: 0.02
    circuit_breaker_drawdown: 0.15
```

See [config/config.yaml](./config/config.yaml) for the complete configuration reference.

## 📁 Project Structure

```
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── data/                  # Data storage (gitignored)
│   ├── raw/              # Raw OHLCV data
│   └── cache/            # Processed features
├── notebooks/             # Jupyter notebooks
│   ├── 01_data_eda.ipynb          # Data exploration
│   ├── 02_train_optuna.ipynb      # Model training
│   └── 03_backtest.ipynb          # Backtesting
├── src/                   # Source code
│   ├── data/             # Data loading utilities
│   │   └── loaders.py
│   ├── features/         # Feature engineering
│   │   └── build_dataset.py
│   ├── models/           # LSTM architectures
│   │   └── lstm_fusion.py
│   ├── training/         # Training framework
│   │   ├── train.py
│   │   └── metrics.py
│   ├── backtest/         # Backtesting engine
│   │   ├── engine.py
│   │   └── strategy.py
│   ├── live/             # Live trading system
│   │   ├── broker_base.py
│   │   ├── streamer.py
│   │   └── executor.py
│   └── utils/            # Utilities
│       ├── config.py
│       ├── logging.py
│       └── times.py
├── tests/                # Test files
├── docs/                 # Documentation
└── requirements.txt       # Dependencies
```

## 🔄 Data Pipeline

### Data Sources

The bot supports multiple data sources:

- **CSV Files**: Local historical data storage
- **Alpaca**: US equities and crypto data
- **Binance**: Cryptocurrency data
- **OANDA**: Forex data

### Feature Engineering

Multi-timeframe features include:

- **Technical Indicators**: RSI, MACD, Stochastic, ATR, Bollinger Bands
- **Price Features**: Moving averages, VWAP, volatility
- **Lagged Returns**: Multiple horizon returns
- **Calendar Features**: Hour, day of week, session indicators
- **Regime Features**: Hidden Markov Model states (optional)

### Data Leakage Protection

- Strict temporal ordering in feature generation
- Proper train/validation/test splits
- Shifted targets to prevent look-ahead bias

## 🧠 Model Architecture

### Single LSTM Architecture
```python
# Concatenated multi-timeframe features fed to single LSTM
features = concatenate_timeframes(tf_15m, tf_30m, tf_1h, tf_2h)
lstm_out = LSTM(hidden_size, num_layers)(features)
predictions = Dense(num_classes)(lstm_out)
```

### Multi-LSTM Fusion Architecture
```python
# Separate LSTM per timeframe with late fusion
lstm_15m = LSTM(hidden_size)(tf_15m_features)
lstm_30m = LSTM(hidden_size)(tf_30m_features)
lstm_1h = LSTM(hidden_size)(tf_1h_features)
lstm_2h = LSTM(hidden_size)(tf_2h_features)

# Fusion strategies: concatenate, attention, dense layers
fused = concatenate([lstm_15m, lstm_30m, lstm_1h, lstm_2h])
predictions = Dense(num_classes)(fused)
```

## 🎯 Training

### Hyperparameter Optimization

Uses Optuna for automatic hyperparameter tuning:

- Learning rate, hidden size, dropout rates
- Sequence length, fusion strategy
- Loss function weights and regularization

### Training Modes

- **Standard Training**: Fixed hyperparameters
- **Optuna Optimization**: Automated hyperparameter search
- **Walk-Forward Validation**: Time-series aware cross-validation

### Metrics

Comprehensive performance tracking:
- Sharpe ratio, maximum drawdown, Calmar ratio
- Hit ratio, profit factor, total return
- Classification metrics (precision, recall, F1)

## 📊 Backtesting

### Backtrader Integration

Realistic backtesting with:
- Commission and slippage modeling
- Order types and execution delays
- Position sizing and risk constraints
- Multiple timeframe signal alignment

### Performance Analysis

Generates comprehensive reports:
- Equity curves and drawdown charts
- Rolling performance metrics
- Trade-by-trade analysis
- Risk-adjusted return metrics

## 🔴 Live Trading

### Execution Modes

- **Paper Trading**: Risk-free testing with real market data
- **Live Trading**: Real money execution with full risk management

### Broker Support

- **Alpaca**: US equities and crypto
- **OANDA**: Forex markets
- **Binance**: Cryptocurrency exchange

### Real-time Features

- WebSocket streaming for live data
- Automatic reconnection and rate limiting
- Order execution with timeout handling
- Comprehensive logging and monitoring

## 🛡️ Risk Management

### Position Sizing

- Volatility-adjusted position sizing
- Maximum capital per trade limits
- Concurrent position constraints

### Risk Controls

- Hard stop-loss and take-profit levels
- Trailing stop mechanisms
- Circuit breakers on excessive drawdown
- Daily loss limits and position timeouts

### Circuit Breakers

Automatic trading suspension for:
- Excessive portfolio drawdown
- Stale market data
- Model uncertainty spikes
- Connection failures

## 🚢 Deployment

### Google Colab

For research and experimentation:
1. Upload project files to Colab
2. Install dependencies: `!pip install -r requirements.txt`
3. Run notebooks for training and backtesting

### Linux Server (Linode/AWS)

For 24/7 operation:

1. **Server Setup**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.10 python3.10-venv

   # Create user and setup directories
   sudo useradd -m -s /bin/bash tradingbot
   sudo mkdir -p /opt/tradingbot
   sudo chown tradingbot:tradingbot /opt/tradingbot
   ```

2. **Application Setup**
   ```bash
   # As tradingbot user
   cd /opt/tradingbot
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Systemd Service**
   ```bash
   # Create service file
   sudo tee /etc/systemd/system/tradingbot.service > /dev/null <<EOF
   [Unit]
   Description=LSTM Trading Bot
   After=network.target

   [Service]
   Type=simple
   User=tradingbot
   WorkingDirectory=/opt/tradingbot
   Environment=PATH=/opt/tradingbot/venv/bin
   ExecStart=/opt/tradingbot/venv/bin/python -m src.live.executor --config config/config.yaml --mode live
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   EOF

   # Enable and start service
   sudo systemctl enable tradingbot
   sudo systemctl start tradingbot
   ```

4. **Monitoring**
   ```bash
   # Check service status
   sudo systemctl status tradingbot

   # View logs
   sudo journalctl -u tradingbot -f

   # Check application logs
   tail -f logs/trading_bot.log
   ```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x scripts/*.sh

# Create non-root user
RUN useradd -m -s /bin/bash tradingbot
USER tradingbot

CMD ["python", "-m", "src.live.executor", "--config", "config/config.yaml", "--mode", "live"]
```

## 📚 API Reference

### Data Loaders

```python
from src.data.loaders import load_ohlcv_data

# Load multi-timeframe data
data = load_ohlcv_data(
    symbols=['EURUSD', 'BTCUSD'],
    timeframes=['15m', '30m', '1h', '2h'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    source='alpaca'
)
```

### Model Training

```python
from src.training.train import train_model

# Train with Optuna optimization
best_model = train_model(
    config_path='config/config.yaml',
    optuna_trials=50,
    save_path='models/best_model.pth'
)
```

### Backtesting

```python
from src.backtest.engine import run_backtest

# Run comprehensive backtest
results = run_backtest(
    config_path='config/config.yaml',
    model_path='models/best_model.pth',
    report_path='reports/backtest.html'
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/

# Run specific test file
pytest tests/test_features.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyTorch, pandas, and scikit-learn
- Backtesting powered by backtrader
- Hyperparameter optimization via Optuna
- Real-time data streaming with websockets

## 📞 Support

- 📧 Email: support@tradingbot.com
- 💬 Discord: [Join our community](https://discord.gg/tradingbot)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/lstm-trading-bot/issues)
- 📖 Wiki: [Documentation](https://github.com/yourusername/lstm-trading-bot/wiki)

---

**Happy Trading! 🚀📈**