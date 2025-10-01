#!/usr/bin/env python3
"""
Main CLI entry point for LSTM Trading Bot.
"""
import argparse
import sys
from pathlib import Path

def main():
    """Main CLI dispatcher."""

    parser = argparse.ArgumentParser(
        description='LSTM Multi-Timeframe Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Data loading
  python main.py data load --symbols EURUSD BTCUSD --start 2020-01-01 --end 2024-12-31

  # Feature building
  python main.py features build --config config/config.yaml --data-path data/loaded.pkl

  # Model training
  python main.py training train --config config/config.yaml --data-path data/features.pkl

  # Backtesting
  python main.py backtest run --config config/config.yaml --model-path models/best_model.pth

  # Live trading
  python main.py live run --config config/config.yaml --model-path models/best_model.pth --mode paper
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Data commands
    data_parser = subparsers.add_parser('data', help='Data operations')
    data_subparsers = data_parser.add_subparsers(dest='subcommand', help='Data subcommands')

    load_parser = data_subparsers.add_parser('load', help='Load OHLCV data')
    load_parser.add_argument('--symbols', nargs='+', required=True)
    load_parser.add_argument('--timeframes', nargs='+', default=['15m', '30m', '1h', '2h'])
    load_parser.add_argument('--start', required=True)
    load_parser.add_argument('--end', required=True)
    load_parser.add_argument('--source', default='csv', choices=['csv', 'alpaca', 'binance', 'oanda'])

    # Features commands
    features_parser = subparsers.add_parser('features', help='Feature engineering')
    features_subparsers = features_parser.add_subparsers(dest='subcommand', help='Features subcommands')

    build_parser = features_subparsers.add_parser('build', help='Build features from data')
    build_parser.add_argument('--config', required=True)
    build_parser.add_argument('--data-path', required=True)
    build_parser.add_argument('--output', required=True)
    build_parser.add_argument('--sequence-length', type=int, default=60)

    # Training commands
    training_parser = subparsers.add_parser('training', help='Model training')
    training_subparsers = training_parser.add_subparsers(dest='subcommand', help='Training subcommands')

    train_parser = training_subparsers.add_parser('train', help='Train LSTM model')
    train_parser.add_argument('--config', required=True)
    train_parser.add_argument('--data-path', required=True)
    train_parser.add_argument('--output', required=True)
    train_parser.add_argument('--optuna', type=int, default=0)

    # Backtest commands
    backtest_parser = subparsers.add_parser('backtest', help='Backtesting')
    backtest_subparsers = backtest_parser.add_subparsers(dest='subcommand', help='Backtest subcommands')

    run_parser = backtest_subparsers.add_parser('run', help='Run backtest')
    run_parser.add_argument('--config', required=True)
    run_parser.add_argument('--data-path', required=True)
    run_parser.add_argument('--model-path', required=True)
    run_parser.add_argument('--report', default='reports/backtest.html')

    # Live trading commands
    live_parser = subparsers.add_parser('live', help='Live trading')
    live_subparsers = live_parser.add_subparsers(dest='subcommand', help='Live subcommands')

    live_run_parser = live_subparsers.add_parser('run', help='Run live trading')
    live_run_parser.add_argument('--config', required=True)
    live_run_parser.add_argument('--model-path', required=True)
    live_run_parser.add_argument('--mode', default='paper', choices=['paper', 'live'])
    live_run_parser.add_argument('--api-key')
    live_run_parser.add_argument('--api-secret')

    # Global options
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Import and run appropriate module
    try:
        if args.command == 'data':
            if args.subcommand == 'load':
                from src.data.loaders_cli import main as data_main
                # Override sys.argv for the subcommand
                sys.argv = ['data_load'] + [f'--{k}={v}' if v is not None else f'--{k}' for k, v in vars(args).items() if k not in ['command', 'subcommand'] and v is not None]
                data_main()

        elif args.command == 'features':
            if args.subcommand == 'build':
                from src.features.build_dataset_cli import main as features_main
                sys.argv = ['features_build'] + [f'--{k}={v}' if v is not None else f'--{k}' for k, v in vars(args).items() if k not in ['command', 'subcommand'] and v is not None]
                features_main()

        elif args.command == 'training':
            if args.subcommand == 'train':
                from src.training.train_cli import main as training_main
                sys.argv = ['training_train'] + [f'--{k}={v}' if v is not None else f'--{k}' for k, v in vars(args).items() if k not in ['command', 'subcommand'] and v is not None]
                training_main()

        elif args.command == 'backtest':
            if args.subcommand == 'run':
                from src.backtest.engine_cli import main as backtest_main
                sys.argv = ['backtest_run'] + [f'--{k}={v}' if v is not None else f'--{k}' for k, v in vars(args).items() if k not in ['command', 'subcommand'] and v is not None]
                backtest_main()

        elif args.command == 'live':
            if args.subcommand == 'run':
                from src.live.executor_cli import main as live_main
                sys.argv = ['live_run'] + [f'--{k}={v}' if v is not None else f'--{k}' for k, v in vars(args).items() if k not in ['command', 'subcommand'] and v is not None]
                live_main()

    except ImportError as e:
        print(f"Error importing module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running command: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
