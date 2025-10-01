"""
CLI interface for model training.
"""
import argparse
import sys
import logging

from .train import train_model
from ..utils.config import load_config
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)

def main():
    """Main CLI entry point for model training."""

    parser = argparse.ArgumentParser(description='Train LSTM trading model')

    parser.add_argument('--config', required=True,
                       help='Configuration file path')

    parser.add_argument('--data-path', required=True,
                       help='Path to features data file')

    parser.add_argument('--output', required=True,
                       help='Output path for trained model')

    parser.add_argument('--optuna', type=int, default=0,
                       help='Number of Optuna trials for hyperparameter optimization')

    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger('model_trainer')
    logger.setLevel(getattr(logging, args.log_level))

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Load training data
        logger.info(f"Loading training data from {args.data_path}")

        import pickle
        with open(args.data_path, 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset['X'], dataset['y']

        logger.info(f"Training data shape: X={X.shape}, y={len(y)}")

        # Train model
        logger.info("Starting model training...")

        if args.optuna > 0:
            logger.info(f"Running hyperparameter optimization with {args.optuna} trials")

        model = train_model(
            X=X,
            y=y,
            config_path=args.config,
            optuna_trials=args.optuna,
            save_path=args.output
        )

        logger.info(f"Model training completed. Saved to {args.output}")

    except Exception as e:
        logger.error(f"Error training model: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
