"""
CLI interface for backtesting.
"""
import argparse
import sys
import logging
import pickle

from .engine import run_backtest
from ..utils.config import load_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)

def main():
    """Main CLI entry point for backtesting."""

    parser = argparse.ArgumentParser(description='Run backtest for LSTM trading strategy')

    parser.add_argument('--config', required=True,
                       help='Configuration file path')

    parser.add_argument('--data-path', required=True,
                       help='Path to backtest data file')

    parser.add_argument('--model-path', required=True,
                       help='Path to trained model file')

    parser.add_argument('--report', default='reports/backtest.html',
                       help='Output path for backtest report')

    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger('backtester')
    logger.setLevel(getattr(logging, args.log_level))

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Load backtest data
        logger.info(f"Loading backtest data from {args.data_path}")
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)

        # Run backtest
        logger.info("Starting backtest...")

        results = run_backtest(
            data=data,
            model_path=args.model_path,
            config_path=args.config,
            report_path=args.report
        )

        # Print summary
        logger.info("Backtest Results Summary:")
        logger.info(f"  Total Return: {results['portfolio']['total_return']".2%"}")
        logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']".3f"}")
        logger.info(f"  Max Drawdown: {results['max_drawdown']".2%"}")
        logger.info(f"  Total Trades: {results['num_trades']}")
        logger.info(f"  Win Rate: {results['winning_trades']/results['num_trades']".2%"}" if results['num_trades'] > 0 else "  Win Rate: N/A")

        logger.info(f"Backtest report saved to {args.report}")

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
