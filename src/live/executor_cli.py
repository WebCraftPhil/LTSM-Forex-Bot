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

    parser.add_argument('--broker', default=None,
                       help='Broker name override (alpaca or tradelocker)')

    parser.add_argument('--api-key', default=None,
                       help='Broker API key (or set API_KEY_ALPACA env var)')

    parser.add_argument('--api-secret', default=None,
                       help='Broker API secret (or set API_SECRET_ALPACA env var)')

    parser.add_argument('--access-token', default=None,
                       help='TradeLocker access token')

    parser.add_argument('--email', default=None,
                       help='TradeLocker account email')

    parser.add_argument('--password', default=None,
                       help='TradeLocker account password')

    parser.add_argument('--server', default=None,
                       help='TradeLocker server name')

    parser.add_argument('--account-id', type=int, default=None,
                       help='TradeLocker account id')

    parser.add_argument('--acc-num', type=int, default=None,
                       help='TradeLocker account number')

    parser.add_argument('--base-url', default=None,
                       help='Broker API base URL override')

    parser.add_argument('--developer-api-key', default=None,
                       help='Optional TradeLocker developer API key')

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

        broker_name = (args.broker or config.live.broker or 'alpaca').lower()

        broker_options = {
            "base_url": args.base_url,
            "account_id": args.account_id,
            "acc_num": args.acc_num,
            "server": args.server,
            "access_token": args.access_token,
            "email": args.email,
            "password": args.password,
            "developer_api_key": args.developer_api_key,
        }
        broker_options = {key: value for key, value in broker_options.items() if value is not None}

        api_key = args.api_key
        api_secret = args.api_secret

        live_config = config.live

        if broker_name == "alpaca":
            api_key = api_key or live_config.api_key or os.getenv('API_KEY_ALPACA')
            api_secret = api_secret or live_config.api_secret or os.getenv('API_SECRET_ALPACA')

            if not api_key or not api_secret:
                logger.error(
                    "API credentials required. Set API_KEY_ALPACA and API_SECRET_ALPACA environment variables or use --api-key and --api-secret"
                )
                sys.exit(1)
        elif broker_name == "tradelocker":
            broker_options.setdefault(
                "email",
                live_config.tradelocker_email or os.getenv("TRADELOCKER_EMAIL") or api_key,
            )
            broker_options.setdefault(
                "password",
                live_config.tradelocker_password or os.getenv("TRADELOCKER_PASSWORD") or api_secret,
            )
            broker_options.setdefault(
                "access_token",
                live_config.tradelocker_access_token or os.getenv("TRADELOCKER_ACCESS_TOKEN"),
            )
            broker_options.setdefault(
                "server",
                live_config.tradelocker_server or os.getenv("TRADELOCKER_SERVER", "SERVER"),
            )
            broker_options.setdefault(
                "base_url",
                live_config.tradelocker_demo_base_url
                if args.mode == "paper"
                else live_config.tradelocker_live_base_url,
            )
            broker_options.setdefault(
                "account_id",
                live_config.tradelocker_account_id,
            )
            broker_options.setdefault(
                "acc_num",
                live_config.tradelocker_acc_num,
            )
            broker_options.setdefault(
                "developer_api_key",
                live_config.tradelocker_developer_api_key or os.getenv("TRADELOCKER_DEVELOPER_API_KEY"),
            )

            if not broker_options.get("access_token") and (
                not broker_options.get("email") or not broker_options.get("password")
            ):
                logger.error(
                    "TradeLocker credentials required. Provide --access-token or --email/--password, or set TRADELOCKER_ACCESS_TOKEN / TRADELOCKER_EMAIL / TRADELOCKER_PASSWORD."
                )
                sys.exit(1)
        else:
            logger.error(f"Unknown broker: {broker_name}")
            sys.exit(1)

        logger.info(f"Starting {args.mode} trading with model: {args.model_path}")

        # Run executor
        run_executor(
            model_path=args.model_path,
            mode=args.mode,
            api_key=api_key,
            api_secret=api_secret,
            broker_name=broker_name,
            **broker_options,
        )

    except Exception as e:
        logger.error(f"Error running live trading: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
