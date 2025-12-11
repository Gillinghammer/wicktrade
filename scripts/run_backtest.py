#!/usr/bin/env python3
"""
Run backtests on the wick trading strategy.

Usage:
    python scripts/run_backtest.py --symbol SPY --timeframe 15Min
    python scripts/run_backtest.py --symbol SPY --timeframe 15Min --start 2024-01-01
    python scripts/run_backtest.py --symbol SPY --compare-timeframes
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wicktrade.core.data_manager import DataManager
from wicktrade.core.types import BacktestConfig
from wicktrade.strategy.wick_strategy import WickStrategy
from wicktrade.backtest.engine import BacktestEngine
from wicktrade.metrics.reporter import Reporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_single_backtest(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    strategy_config: dict,
) -> dict:
    """Run a single backtest and return result."""
    dm = DataManager()

    # Get data
    candles = dm.get_candles(
        symbol=symbol,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        use_cache=True,
    )

    if not candles:
        logger.error(f"No data available for {symbol} {timeframe}")
        return None

    logger.info(f"Running backtest with {len(candles)} candles")

    # Create strategy
    strategy = WickStrategy(config=strategy_config)

    # Create config
    config = BacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        strategy_config=strategy_config,
    )

    # Run backtest
    engine = BacktestEngine(strategy, initial_capital)
    result = engine.run(candles, symbol, config)

    return result


def compare_timeframes(
    symbol: str,
    timeframes: list,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    strategy_config: dict,
):
    """Run backtests across multiple timeframes and compare."""
    results = {}

    for tf in timeframes:
        logger.info(f"\nRunning backtest for {symbol} {tf}...")
        result = run_single_backtest(
            symbol, tf, start_date, end_date, initial_capital, strategy_config
        )
        if result:
            results[tf] = result

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run wick strategy backtests"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Stock symbol to test"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15Min",
        help="Timeframe for bars"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Days of history (if start not specified)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital"
    )
    parser.add_argument(
        "--compare-timeframes",
        action="store_true",
        help="Compare across multiple timeframes"
    )
    parser.add_argument(
        "--show-trades",
        action="store_true",
        help="Show individual trades"
    )

    # Strategy parameters
    parser.add_argument("--min-trend-length", type=int, default=5)
    parser.add_argument("--wick-confirmation", type=float, default=0.5)
    parser.add_argument("--stop-loss-mult", type=float, default=1.5)

    args = parser.parse_args()

    # Parse dates
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.utcnow()

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=args.days)

    # Strategy config
    strategy_config = {
        "min_trend_length": args.min_trend_length,
        "wick_confirmation_ratio": args.wick_confirmation,
        "stop_loss_multiplier": args.stop_loss_mult,
    }

    reporter = Reporter()

    if args.compare_timeframes:
        # Compare multiple timeframes
        timeframes = ["5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"]
        results = compare_timeframes(
            args.symbol,
            timeframes,
            start_date,
            end_date,
            args.capital,
            strategy_config,
        )

        if results:
            reporter.compare_timeframes(results)

            # Show best performing
            best_tf = max(results, key=lambda k: results[k].metrics.get("sharpe_ratio", 0))
            print(f"\nBest timeframe: {best_tf}")
            reporter.print_summary(results[best_tf])
            reporter.print_success_criteria(results[best_tf])

    else:
        # Single backtest
        result = run_single_backtest(
            args.symbol,
            args.timeframe,
            start_date,
            end_date,
            args.capital,
            strategy_config,
        )

        if result:
            reporter.print_summary(result)
            reporter.print_success_criteria(result)

            if args.show_trades:
                reporter.print_trades(result.trades)


if __name__ == "__main__":
    main()
