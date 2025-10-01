# LSTM Trading Bot - Complete Implementation Guide

## 🚀 Overview

This guide provides step-by-step instructions to take the scaffolded LSTM trading bot from a project template to a fully operational trading system. Follow these steps in order for the best results.

## 📋 Table of Contents

1. [Immediate Setup](#immediate-setup)
2. [Data Collection](#data-collection)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Backtesting](#backtesting)
6. [Live Trading Setup](#live-trading-setup)
7. [Risk Management Configuration](#risk-management-configuration)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)
11. [Future Enhancements](#future-enhancements)

## 1. Immediate Setup

### 1.1 Environment Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch, pandas, backtrader; print('All dependencies installed successfully')"
```

### 1.2 Configuration Setup

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env with your API credentials
nano .env  # or your preferred editor

# Required for live trading:
# API_KEY_ALPACA=your_key_here
# API_SECRET_ALPACA=your_secret_here
```

### 1.3 Project Structure Verification

```bash
# Verify all files are present
ls -la
# Should show: config/, src/, notebooks/, docs/, requirements.txt, etc.

# Test configuration loading
python -c "from src.utils.config import get_config; config = get_config(); print('Config loaded successfully')"
```

## 2. Data Collection

### 2.1 Choose Your Data Sources

**For Development/Research:**
- Use CSV files with historical data
- Download from Yahoo Finance, Alpha Vantage, or similar

**For Live Trading:**
- Alpaca (US stocks/crypto)
- Binance (crypto)
- OANDA (forex)

### 2.2 Collect Historical Data

**Option A: CSV Files (Recommended for initial development)**

```bash
# Create data/raw directory
mkdir -p data/raw

# Download sample data (replace with your data source)
# Example: EURUSD 1h data from 2020-2024
# Place CSV files in data/raw/ with format: SYMBOL_TIMEFRAME.csv
# Columns: timestamp,open,high,low,close,volume

# Verify data format
python -c "
import pandas as pd
df = pd.read_csv('data/raw/EURUSD_1h.csv')
print(f'Data shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Date range: {df.timestamp.min()} to {df.timestamp.max()}')
"
```

**Option B: Alpaca API (Live data)**

```bash
# Set API credentials in .env first
# Then load data via CLI
python main.py data load \
    --symbols EURUSD BTCUSD \
    --timeframes 15m 30m 1h 2h \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --source alpaca
```

### 2.3 Data Quality Checks

```python
# Run data quality analysis
import pandas as pd
from src.data.loaders import load_ohlcv_data

# Load and inspect data
data = load_ohlcv_data(
    symbols=['EURUSD'],
    timeframes=['1h'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    source='csv'
)

df = data['EURUSD']['1h']
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate timestamps: {df.index.duplicated().sum()}")
print(f"Price gaps: {(df['close'] - df['close'].shift(1)).abs().max()}")
```

## 3. Feature Engineering

### 3.1 Build Training Dataset

```bash
# Build features from loaded data
python main.py features build \
    --config config/config.yaml \
    --data-path data/loaded_data.pkl \
    --output data/features.pkl \
    --sequence-length 60
```

### 3.2 Feature Analysis

```python
# Analyze built features
import pickle
with open('data/features.pkl', 'rb') as f:
    X, y = pickle.load(f)

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {len(y)}")
print(f"Number of features: {X.shape[2]}")

# Check feature distributions
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
for i in range(min(10, X.shape[2])):
    plt.subplot(2, 5, i+1)
    plt.hist(X[:, 0, i], bins=50)
    plt.title(f'Feature {i}')
plt.tight_layout()
plt.show()
```

## 4. Model Training

### 4.1 Hyperparameter Optimization

```bash
# Run Optuna optimization (this takes time!)
python main.py training train \
    --config config/config.yaml \
    --data-path data/features.pkl \
    --output models/best_model.pth \
    --optuna 50
```

**Expected Output:**
```
Trial 49 finished with value: -0.2345 and parameters: {...}
Best trial: 42
Best value: -0.1234
Best parameters: {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, ...}
Model training completed. Saved to models/best_model.pth
```

### 4.2 Manual Training (if needed)

```python
# For custom training without Optuna
from src.training.train import train_model
import pickle

with open('data/features.pkl', 'rb') as f:
    X, y = pickle.load(f)

model = train_model(
    X=X,
    y=y,
    config_path='config/config.yaml',
    save_path='models/manual_model.pth'
)
```

### 4.3 Model Validation

```python
# Load and test model
from src.models.lstm_fusion import load_model
import torch

model = load_model('models/best_model.pth')
model.eval()

# Test prediction
with torch.no_grad():
    sample_input = torch.randn(1, 60, model.input_size)
    prediction = model(sample_input)
    print(f"Sample prediction shape: {prediction.shape}")
```

## 5. Backtesting

### 5.1 Run Comprehensive Backtest

```bash
# Run backtest with the trained model
python main.py backtest run \
    --config config/config.yaml \
    --data-path data/test_data.pkl \
    --model-path models/best_model.pth \
    --report reports/backtest_results.html
```

### 5.2 Analyze Results

**Key Metrics to Evaluate:**
- **Sharpe Ratio**: > 1.0 is good, > 2.0 is excellent
- **Max Drawdown**: < 15% is acceptable, < 10% is good
- **Win Rate**: > 50% is decent, > 60% is good
- **Profit Factor**: > 1.5 is good, > 2.0 is excellent

```python
# Load and analyze backtest results
import pickle
with open('reports/backtest_results.pkl', 'rb') as f:
    results = pickle.load(f)

print("Backtest Summary:")
print(f"Total Return: {results['portfolio']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Total Trades: {results['num_trades']}")
print(f"Win Rate: {results['winning_trades']/results['num_trades']:.2%}")
```

### 5.3 Risk-Adjusted Performance

```python
# Calculate risk metrics
from src.training.metrics import TradingMetrics

metrics = TradingMetrics()
portfolio_returns = [trade['pnl'] for trade in results['trades']]

risk_metrics = metrics.calculate_trade_statistics(portfolio_returns)
print(f"Calmar Ratio: {risk_metrics.get('calmar_ratio', 0):.3f}")
print(f"Sortino Ratio: {risk_metrics.get('sortino_ratio', 0):.3f}")
```

## 6. Live Trading Setup

### 6.1 Paper Trading (Recommended First)

```bash
# Start paper trading for validation
python main.py live run \
    --config config/config.yaml \
    --model-path models/best_model.pth \
    --mode paper \
    --api-key $API_KEY_ALPACA \
    --api-secret $API_SECRET_ALPACA
```

**Monitor in another terminal:**
```bash
# Check logs
tail -f logs/trading_bot.log

# Check portfolio value
# The bot will log performance every execution cycle
```

### 6.2 Risk Management Configuration

Before going live, adjust risk parameters:

```yaml
# config/config.yaml - Risk Section
backtest:
  risk:
    max_position_size: 0.02  # 2% per trade
    max_positions: 3         # Conservative for live
    stop_loss: 0.015         # 1.5% stop loss
    take_profit: 0.03        # 3% take profit
    circuit_breaker_drawdown: 0.10  # 10% circuit breaker
```

### 6.3 Live Trading (Production)

```bash
# Only after successful paper trading!
python main.py live run \
    --config config/config.yaml \
    --model-path models/best_model.pth \
    --mode live \
    --api-key $API_KEY_ALPACA \
    --api-secret $API_SECRET_ALPACA
```

## 7. Risk Management Configuration

### 7.1 Position Sizing Strategies

**Conservative (Recommended for beginners):**
```python
# Fixed percentage sizing
position_size = portfolio_value * 0.02  # 2% per trade
```

**Advanced (Kelly Criterion):**
```python
# Calculate optimal position size
win_rate = 0.55  # From backtesting
avg_win = 0.02   # 2% average win
avg_loss = 0.01  # 1% average loss

kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
position_size = portfolio_value * min(kelly_pct, 0.05)  # Cap at 5%
```

### 7.2 Circuit Breakers

**Daily Loss Limit:**
- Stop trading if daily loss > 3%
- Reset at midnight UTC

**Drawdown Protection:**
- Stop trading if portfolio drops > 15% from peak
- Manual reset required

**Position Limits:**
- Maximum 5 concurrent positions
- Maximum 2% of portfolio per position

## 8. Monitoring and Maintenance

### 8.1 Real-time Monitoring

```bash
# Monitor logs
tail -f logs/trading_bot.log | grep -E "(Trade|Signal|Risk|Error)"

# Monitor performance
python -c "
import json
with open('logs/trading_bot.log', 'r') as f:
    for line in f:
        if 'portfolio_value' in line:
            print(line.strip())
"
```

### 8.2 Daily Health Checks

```python
# Check system health
import subprocess
import sys

def health_check():
    # Check if model files exist
    assert os.path.exists('models/best_model.pth'), "Model file missing"

    # Check if data is recent
    data_files = glob.glob('data/cache/*.parquet')
    assert len(data_files) > 0, "No cached data found"

    # Check API connectivity
    try:
        # Test API call
        pass
    except:
        print("API connectivity issue")

    print("All health checks passed!")

health_check()
```

### 8.3 Model Retraining Schedule

**Weekly Retraining:**
```bash
# Every Sunday at 2 AM
# Add to crontab:
# 0 2 * * 0 /path/to/venv/bin/python /path/to/project/scripts/retrain_model.py
```

**Performance Monitoring:**
- Track live Sharpe ratio vs backtested
- Monitor maximum drawdown
- Alert on significant performance degradation

## 9. Troubleshooting

### 9.1 Common Issues

**Issue: Model not training**
```python
# Check data shapes
import pickle
with open('data/features.pkl', 'rb') as f:
    X, y = pickle.load(f)

print(f"X shape: {X.shape}")  # Should be (samples, sequence_length, features)
print(f"y shape: {len(y)}")    # Should match samples
```

**Issue: Poor backtest performance**
```python
# Check for overfitting
# Compare train/validation metrics
# Reduce model complexity if needed
# Increase regularization (dropout, L2)
```

**Issue: Live trading connection errors**
```python
# Check API credentials
python -c "
import os
print('API Key:', os.getenv('API_KEY_ALPACA')[:10] + '...' if os.getenv('API_KEY_ALPACA') else 'Not set')
print('API Secret:', os.getenv('API_SECRET_ALPACA')[:10] + '...' if os.getenv('API_SECRET_ALPACA') else 'Not set')
"

# Test broker connection
from src.live.broker_alpaca import AlpacaBroker
broker = AlpacaBroker()
print("Connected:", broker.is_connected())
```

### 9.2 Debugging Tools

**Feature Debugging:**
```python
# Check feature distributions
import seaborn as sns
import matplotlib.pyplot as plt

features_df = pd.DataFrame(X[:, 0, :])  # First timestep
plt.figure(figsize=(15, 10))
sns.boxplot(data=features_df)
plt.xticks(rotation=45)
plt.show()
```

**Model Debugging:**
```python
# Check model predictions
model.eval()
with torch.no_grad():
    sample = torch.randn(1, 60, model.input_size)
    pred = model(sample)
    print(f"Prediction: {pred.item():.4f}")
```

## 10. Performance Optimization

### 10.1 Training Optimization

```python
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimize batch size
# Start with 32, increase until memory limit
batch_size = 64  # Adjust based on your hardware
```

### 10.2 Inference Optimization

```python
# Model quantization for faster inference
import torch.quantization

# Quantize model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'models/quantized_model.pth')
```

### 10.3 Memory Optimization

```python
# Use mixed precision training
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 11. Future Enhancements

### 11.1 Advanced Features

**Ensemble Models:**
```python
# Combine multiple models
models = [
    load_model('models/model_v1.pth'),
    load_model('models/model_v2.pth'),
    load_model('models/model_v3.pth')
]

# Ensemble prediction
predictions = [model(input_tensor) for model in models]
ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
```

**Reinforcement Learning:**
```python
# Future enhancement: Train with RL
# Use portfolio returns as rewards
# Implement actor-critic or PPO algorithms
```

### 11.2 Multi-Asset Trading

```python
# Extend to multiple assets
symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 'ETHUSD']

# Train separate models per asset or shared model
# Implement portfolio optimization
```

### 11.3 Advanced Risk Management

**Dynamic Position Sizing:**
```python
# Adjust position size based on market volatility
volatility = calculate_market_volatility()
position_size = base_size * (1 / (1 + volatility * 10))
```

**Correlation-Based Risk:**
```python
# Avoid correlated positions
correlation_matrix = calculate_correlations(positions)
if correlation_matrix.max() > 0.7:  # High correlation
    reduce_position_sizes()
```

## 12. Deployment Checklist

### Before Going Live:

- [ ] Backtesting shows consistent profitability (>6 months)
- [ ] Sharpe ratio > 1.0 across multiple market conditions
- [ ] Maximum drawdown < 20% in backtests
- [ ] Paper trading successful for >1 month
- [ ] Risk management parameters tested and validated
- [ ] Monitoring and alerting systems in place
- [ ] Emergency stop procedures documented
- [ ] API rate limits and costs understood

### Production Deployment:

- [ ] Server setup with 24/7 uptime
- [ ] Database backups configured
- [ ] Log aggregation and monitoring
- [ ] Automated model retraining pipeline
- [ ] Performance tracking dashboard
- [ ] Incident response procedures

## 13. Resources and Support

### Documentation:
- [README.md](README.md) - Project overview and setup
- [AGENTS.md](AGENTS.md) - Development guidelines
- [API Documentation](docs/) - Detailed API reference

### Notebooks:
- [01_data_eda.ipynb](notebooks/01_data_eda.ipynb) - Data exploration
- [02_train_optuna.ipynb](notebooks/02_train_optuna.ipynb) - Model training
- [03_backtest.ipynb](notebooks/03_backtest.ipynb) - Strategy evaluation
- [04_live_trading.ipynb](notebooks/04_live_trading.ipynb) - Live trading

### Monitoring:
- Check `logs/trading_bot.log` for system status
- Monitor portfolio value and risk metrics
- Set up alerts for critical events

---

## 🎯 Success Metrics

**Target Performance:**
- **Sharpe Ratio**: > 1.5
- **Max Drawdown**: < 15%
- **Win Rate**: > 55%
- **Profit Factor**: > 1.8

**Risk Management:**
- **Daily Loss Limit**: < 3%
- **Position Concentration**: < 5% per trade
- **Circuit Breaker Response**: < 30 seconds

**Operational:**
- **Uptime**: > 99.5%
- **Model Accuracy**: > 90% of backtested performance
- **Response Time**: < 1 second for signals

---

**Happy Trading! 🚀📈**

*This guide will evolve as you gain experience with the system. Start conservatively, monitor closely, and gradually increase complexity and position sizes as you validate performance.*
