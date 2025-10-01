# LSTM Trading Bot - Agent Development Guide

## Project Overview

This document provides guidelines for automated agents (Cursor, Copilot, Claude, etc.) and human contributors working on the LSTM-based multi-timeframe trading bot project.

## 1. Required Checks

### Code Quality
- **Linting**: Run `python -m py_compile` on all Python files
- **Type Checking**: Ensure proper type hints throughout
- **Import Validation**: Verify all imports work correctly
- **Configuration Validation**: Test YAML configuration loading

### Testing
- **Unit Tests**: Run tests for individual modules
- **Integration Tests**: Test data pipeline and model training
- **Backtesting Validation**: Verify strategy performance
- **Import Tests**: Ensure all modules can be imported

### Dependency Management
- **Requirements Check**: Verify `requirements.txt` includes all dependencies
- **Version Compatibility**: Ensure PyTorch, pandas, etc. versions are compatible
- **Optional Dependencies**: Test functionality without optional packages

## 2. Project Structure

```
├── config/                    # Configuration files
│   └── config.yaml           # Main configuration
├── data/                     # Data storage (gitignored)
│   ├── raw/                 # Raw OHLCV data
│   └── cache/               # Processed features
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_eda.ipynb    # Data exploration
│   ├── 02_train_optuna.ipynb # Model training
│   ├── 03_backtest.ipynb     # Backtesting
│   └── 04_live_trading.ipynb # Live trading
├── src/                      # Source code
│   ├── data/                # Data loading
│   │   └── loaders.py
│   ├── features/            # Feature engineering
│   │   ├── build_dataset.py
│   │   └── indicators.py
│   ├── models/              # LSTM architectures
│   │   └── lstm_fusion.py
│   ├── training/            # Training framework
│   │   ├── train.py
│   │   └── metrics.py
│   ├── backtest/            # Backtesting engine
│   │   ├── engine.py
│   │   └── strategy.py
│   ├── live/                # Live trading
│   │   ├── broker_base.py
│   │   ├── broker_alpaca.py
│   │   ├── streamer.py
│   │   └── executor.py
│   └── utils/               # Utilities
│       ├── config.py
│       ├── logging.py
│       ├── seeds.py
│       └── times.py
├── tests/                   # Test files
├── docs/                    # Documentation
│   └── agents/              # Agent-specific docs
└── requirements.txt          # Dependencies
```

## 3. Coding Conventions

### Python Standards
- **Language**: Python 3.10+ with type hints
- **Style**: Follow PEP 8 with 4-space indentation
- **Imports**: Standard library first, then third-party, then local
- **Documentation**: Use Google-style docstrings
- **Error Handling**: Comprehensive try-catch with logging

### Project-Specific Conventions
- **Configuration**: All parameters in `config/config.yaml`
- **Logging**: Use structured logging with `src.utils.logging`
- **Error Handling**: Graceful degradation with informative messages
- **Testing**: Unit tests for all public functions

### File Naming
- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

## 4. Development Workflow

### Agent Development Process
1. **Read Memory**: Check `docs/agents/ledger.json` for existing work
2. **Understand Intent**: Clarify requirements before implementation
3. **Plan Changes**: Create minimal, testable implementation plan
4. **Implement**: Write clean, documented code
5. **Test**: Validate functionality and edge cases
6. **Document**: Update relevant documentation
7. **Log**: Add entry to `docs/agents/ledger.json`

### Human Contribution Process
1. **Issue Creation**: Create GitHub issue for proposed changes
2. **Branch Creation**: Create feature branch from `main`
3. **Implementation**: Follow coding conventions
4. **Testing**: Run all relevant tests
5. **PR Creation**: Submit pull request with description
6. **Review**: Address review comments
7. **Merge**: Merge after approval

## 5. Module-Specific Guidelines

### Data Pipeline (`src/data/`, `src/features/`)
- **Data Sources**: Support CSV, Alpaca, Binance, OANDA
- **Feature Engineering**: Technical indicators, lagged features, calendar features
- **Data Quality**: Handle missing data, outliers, and anomalies
- **Performance**: Efficient processing for large datasets

### Model Architecture (`src/models/`)
- **LSTM Variants**: Single LSTM, multi-LSTM fusion
- **Attention Mechanisms**: Multi-head attention implementation
- **Regularization**: Dropout, layer normalization
- **Output Modes**: Regression and classification

### Training Framework (`src/training/`)
- **Optimization**: Optuna hyperparameter tuning
- **Validation**: Walk-forward cross-validation
- **Metrics**: Trading-specific performance measures
- **Checkpointing**: Model saving and loading

### Backtesting (`src/backtest/`)
- **Framework**: Backtrader integration
- **Risk Management**: Position sizing, stop losses
- **Performance**: Comprehensive metrics and reporting
- **Realism**: Slippage, commission, latency modeling

### Live Trading (`src/live/`)
- **Broker Abstraction**: Support multiple exchanges
- **Streaming**: Real-time market data
- **Execution**: Order management and risk controls
- **Monitoring**: Performance tracking and alerting

## 6. Configuration Management

### Main Configuration (`config/config.yaml`)
- **Data Settings**: Symbols, timeframes, sources
- **Model Parameters**: Architecture, hyperparameters
- **Training Settings**: Batch size, epochs, optimization
- **Risk Parameters**: Position sizing, circuit breakers
- **Live Trading**: Broker settings, execution parameters

### Environment Variables
```bash
# Broker API credentials
API_KEY_ALPACA=your_key
API_SECRET_ALPACA=your_secret
API_KEY_BINANCE=your_key
API_KEY_OANDA=your_key

# Optional: Alert webhooks
SLACK_WEBHOOK=your_webhook_url
DISCORD_WEBHOOK=your_webhook_url
```

## 7. Testing Strategy

### Unit Tests
- **Data Loaders**: Test data loading from all sources
- **Feature Engineering**: Validate indicator calculations
- **Model Components**: Test LSTM layers and fusion strategies
- **Utilities**: Test configuration and logging

### Integration Tests
- **End-to-End Pipeline**: Data loading → features → training → prediction
- **Backtesting**: Strategy execution with realistic conditions
- **Live Trading**: Paper trading execution and risk management

### Performance Tests
- **Training Speed**: Model training time on different hardware
- **Memory Usage**: Memory consumption during processing
- **Scalability**: Performance with larger datasets

## 8. Deployment Guidelines

### Google Colab
1. Upload project files to Colab environment
2. Install dependencies: `!pip install -r requirements.txt`
3. Run notebooks in sequence for complete workflow
4. Use GPU runtime for faster training

### Linux Server
1. **System Setup**: Ubuntu/Debian with Python 3.10+
2. **Dependencies**: Install via `pip install -r requirements.txt`
3. **Configuration**: Set environment variables for API keys
4. **Service Setup**: Use systemd for 24/7 operation
5. **Monitoring**: Configure logging and alerting

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py", "live", "run", "--mode", "paper"]
```

## 9. Risk Management

### Position Sizing
- **Fixed Percentage**: Fixed % of portfolio per trade
- **Kelly Criterion**: Optimal position sizing based on win rate
- **Volatility Adjusted**: Position size based on market volatility

### Circuit Breakers
- **Drawdown Limits**: Stop trading if portfolio drops too much
- **Daily Loss Limits**: Stop trading if daily loss exceeds threshold
- **Position Limits**: Maximum number of concurrent positions
- **Time-based**: Stop trading during high-risk periods

### Monitoring
- **Performance Tracking**: Real-time P&L and risk metrics
- **Alert System**: Notifications for risk events
- **Health Checks**: System status and connectivity monitoring

## 10. Performance Optimization

### Training Optimization
- **GPU Acceleration**: Use CUDA for faster training
- **Batch Processing**: Optimize batch sizes for hardware
- **Mixed Precision**: Use float16 for memory efficiency
- **Model Parallelism**: Distribute model across multiple GPUs

### Inference Optimization
- **Model Quantization**: Reduce model size for faster inference
- **Batch Inference**: Process multiple predictions together
- **Caching**: Cache frequent calculations and features

### Data Optimization
- **Efficient Storage**: Use Parquet for compressed data storage
- **Lazy Loading**: Load data only when needed
- **Memory Mapping**: Use memory-mapped files for large datasets

## 11. Troubleshooting

### Common Issues
- **Import Errors**: Check Python path and dependencies
- **Configuration Errors**: Validate YAML syntax and required fields
- **Data Issues**: Check data format and missing values
- **Model Errors**: Verify tensor shapes and data types

### Debugging Tools
- **Logging**: Comprehensive logging at all levels
- **Profiling**: Performance profiling for bottlenecks
- **Visualization**: Plot data and model outputs for inspection
- **Interactive Debugging**: Use IPython for step-through debugging

## 12. Future Enhancements

### Model Improvements
- **Transformer Architectures**: Add attention-based models
- **Ensemble Methods**: Combine multiple model predictions
- **Reinforcement Learning**: Train using RL for better adaptation

### Feature Enhancements
- **Alternative Data**: News, sentiment, macroeconomic indicators
- **Advanced Indicators**: More sophisticated technical analysis
- **Regime Detection**: Machine learning-based market regime classification

### System Improvements
- **Multi-Asset Trading**: Handle multiple asset classes
- **Portfolio Optimization**: Modern portfolio theory integration
- **Risk Parity**: Equal risk contribution across positions

## 13. Contributing Guidelines

### Code Contributions
1. Follow existing code style and conventions
2. Add tests for new functionality
3. Update documentation for API changes
4. Use meaningful commit messages

### Issue Management
- Use clear, descriptive issue titles
- Provide detailed descriptions and reproduction steps
- Label issues appropriately (bug, enhancement, documentation)
- Reference related issues and PRs

### Review Process
- All PRs require at least one review
- Address review comments promptly
- Ensure CI checks pass before merge
- Update changelog for significant changes

## 14. Maintenance

### Regular Tasks
- **Model Retraining**: Update models with new data
- **Performance Review**: Analyze live vs backtested performance
- **Risk Review**: Adjust risk parameters based on performance
- **Dependency Updates**: Keep packages current and secure

### Monitoring
- **System Health**: Monitor server and application health
- **Performance Metrics**: Track key performance indicators
- **Error Rates**: Monitor and address error patterns
- **Resource Usage**: Track CPU, memory, and disk usage

## 15. Support and Resources

### Documentation
- **README.md**: Project overview and setup instructions
- **API Documentation**: Generated from docstrings
- **Colab Notebooks**: Interactive tutorials and examples
- **Configuration Guide**: Detailed parameter explanations

### Community
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and discussions
- **Wiki**: Additional documentation and guides

### Development Tools
- **IDE Support**: Cursor, VS Code with Python extensions
- **Linting**: Pylint, flake8 for code quality
- **Formatting**: Black for consistent code formatting
- **Testing**: Pytest for unit and integration tests

---

This guide ensures consistent development practices and high-quality code across all contributors and automated agents working on the LSTM trading bot project.
