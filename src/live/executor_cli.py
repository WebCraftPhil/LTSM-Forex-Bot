"""
CLI interface for live trading execution.
"""
import argparse
import sys
import logging
import os

from .executor import run_executor
from ..utils.config import load_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)

def main():
    """Main CLI entry point for live trading."""

    parser = argparse.ArgumentParser(description='Run live trading executor')

    parser.add_argument('--config', required=True,
                       help='Configuration file path')

    parser.add_argument('--model-path', required=True,
                       help='Path to trained model file')

    parser.add_argument('--mode', default='paper',
                       choices=['paper', 'live'],
                       help='Trading mode')

    parser.add_argument('--api-key', default=None,
                       help='Broker API key (or set API_KEY_ALPACA env var)')

    parser.add_argument('--api-secret', default=None,
                       help='Broker API secret (or set API_SECRET_ALPACA env var)')

    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger('live_executor')
    logger.setLevel(getattr(logging, args.log_level))

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Get API credentials
        api_key = args.api_key or os.getenv('API_KEY_ALPACA')
        api_secret = args.api_secret or os.getenv('API_SECRET_ALPACA')

        if not api_key or not api_secret:
            logger.error("API credentials required. Set API_KEY_ALPACA and API_SECRET_ALPACA environment variables or use --api-key and --api-secret")
            sys.exit(1)

        logger.info(f"Starting {args.mode} trading with model: {args.model_path}")

        # Run executor
        run_executor(
            model_path=args.model_path,
            mode=args.mode,
            api_key=api_key,
            api_secret=api_secret
        )

    except Exception as e:
        logger.error(f"Error running live trading: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
