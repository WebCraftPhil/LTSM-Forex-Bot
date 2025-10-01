"""
CLI interface for feature engineering.
"""
import argparse
import sys
import logging
import pickle
from pathlib import Path

from .build_dataset import build_dataset, save_dataset
from ..utils.config import load_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)

def main():
    """Main CLI entry point for feature building."""

    parser = argparse.ArgumentParser(description='Build features for LSTM trading model')

    parser.add_argument('--config', required=True,
                       help='Configuration file path')

    parser.add_argument('--data-path', required=True,
                       help='Path to loaded data pickle file')

    parser.add_argument('--output', required=True,
                       help='Output path for features')

    parser.add_argument('--sequence-length', type=int, default=60,
                       help='Sequence length for LSTM')

    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger('feature_builder')
    logger.setLevel(getattr(logging, args.log_level))

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Load data
        logger.info(f"Loading data from {args.data_path}")
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)

        # Build features
        logger.info("Building features...")
        X, y = build_dataset(data, args.sequence_length)

        # Save features
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dataset(X, y, str(output_path))

        logger.info(f"Features saved to {args.output}")
        logger.info(f"Dataset shape: X={X.shape}, y={len(y)}")

    except Exception as e:
        logger.error(f"Error building features: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
