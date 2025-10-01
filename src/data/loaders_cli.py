"""
CLI interface for data loading.
"""
import argparse
import sys
from datetime import datetime
import logging

from .loaders import load_ohlcv_data
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)

def main():
    """Main CLI entry point for data loading."""

    parser = argparse.ArgumentParser(description='Load OHLCV data for trading bot')

    parser.add_argument('--symbols', nargs='+', required=True,
                       help='Trading symbols (e.g., EURUSD BTCUSD)')

    parser.add_argument('--timeframes', nargs='+', default=['15m', '30m', '1h', '2h'],
                       help='Timeframes to load (e.g., 15m 30m 1h 2h)')

    parser.add_argument('--start', required=True,
                       help='Start date (YYYY-MM-DD)')

    parser.add_argument('--end', required=True,
                       help='End date (YYYY-MM-DD)')

    parser.add_argument('--source', default='csv',
                       choices=['csv', 'alpaca', 'binance', 'oanda'],
                       help='Data source')

    parser.add_argument('--output', default='data/loaded',
                       help='Output directory for loaded data')

    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger('data_loader')
    logger.setLevel(getattr(logging, args.log_level))

    try:
        logger.info(f"Loading data for symbols: {args.symbols}")
        logger.info(f"Timeframes: {args.timeframes}")
        logger.info(f"Date range: {args.start} to {args.end}")
        logger.info(f"Source: {args.source}")

        # Load data
        data = load_ohlcv_data(
            symbols=args.symbols,
            timeframes=args.timeframes,
            start_date=args.start,
            end_date=args.end,
            source=args.source
        )

        # Save data (simplified)
        logger.info(f"Loaded data for {len(data)} symbols")

        for symbol, tf_data in data.items():
            logger.info(f"  {symbol}: {len(tf_data)} timeframes")

            for timeframe, df in tf_data.items():
                logger.info(f"    {timeframe}: {len(df)} records")

        logger.info("Data loading completed successfully")

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
