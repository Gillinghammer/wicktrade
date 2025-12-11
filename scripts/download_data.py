#!/usr/bin/env python3
"""
Download historical market data for backtesting.

Usage:
    python scripts/download_data.py --symbols SPY,QQQ,AAPL --timeframe 15Min
    python scripts/download_data.py --symbols SPY --timeframe 1Day --days 365
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wicktrade.core.data_manager import DataManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Download historical market data"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated list of symbols (e.g., SPY,QQQ,AAPL)"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15Min",
        choices=["1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"],
        help="Timeframe for bars"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of history to download"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/raw",
        help="Directory for cached data"
    )

    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    logger.info(f"Downloading data for: {', '.join(symbols)}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    # Initialize data manager
    dm = DataManager(cache_dir=args.cache_dir)

    # Download data
    results = dm.download_symbols(
        symbols=symbols,
        timeframe=args.timeframe,
        start=start_date,
        end=end_date,
    )

    # Print results
    print("\n" + "=" * 50)
    print("DOWNLOAD RESULTS")
    print("=" * 50)

    total_bars = 0
    for symbol, count in results.items():
        status = "OK" if count > 0 else "FAILED"
        print(f"  {symbol}: {count:,} bars [{status}]")
        total_bars += count

    print("-" * 50)
    print(f"  Total: {total_bars:,} bars")
    print("=" * 50 + "\n")

    # Show cache info
    cache_info = dm.get_cache_info()
    print("Cache Status:")
    for info in cache_info:
        print(
            f"  {info['symbol']} {info['timeframe']}: "
            f"{info['rows']:,} bars, {info['size_mb']:.2f} MB"
        )


if __name__ == "__main__":
    main()
