#!/usr/bin/env python3
"""
VectorBT-based parameter optimization for the wick trading strategy.

Uses grid search to find optimal parameters across assets and timeframes.

Usage:
    python scripts/optimize_params.py --symbol SPY --timeframe 15Min
    python scripts/optimize_params.py --symbol QQQ --timeframe 30Min --grid-search
    python scripts/optimize_params.py --multi-asset
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wicktrade.core.data_manager import DataManager
from wicktrade.strategy.wick_strategy import WickStrategy
from wicktrade.backtest.engine import BacktestEngine
from wicktrade.metrics.reporter import Reporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Parameter grid for optimization
PARAM_GRID = {
    "stop_loss_multiplier": [0.8, 1.0, 1.1, 1.2],
    "wick_confirmation_ratio": [0.5, 0.6, 0.7],
    "trailing_ratio": [0.5, 0.6, 0.7, 0.8],
    "min_trend_length": [5, 7],
}

# Assets to test
STOCK_SYMBOLS = ["SPY", "QQQ", "AAPL", "NVDA"]
CRYPTO_SYMBOLS = ["ETH/USD", "SOL/USD"]  # BTC excluded due to price/position sizing constraints

# Timeframes to test
TIMEFRAMES = ["15Min", "30Min", "1Hour"]

# Minimum trades for statistical significance
MIN_TRADES = 50


def calculate_param_combinations() -> List[Dict]:
    """Generate all parameter combinations from grid."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    combinations = []
    for combo in product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)

    return combinations


def run_single_optimization(
    symbol: str,
    timeframe: str,
    params: Dict,
    dm: DataManager,
    initial_capital: float = 100000.0,
) -> Optional[Dict]:
    """
    Run a single backtest with given parameters.

    Returns:
        Dict with metrics if successful and meets minimum trade requirement,
        None otherwise.
    """
    try:
        # Get data
        candles = dm.get_candles(symbol, timeframe, use_cache=True)

        if len(candles) < 100:
            return None

        # Create strategy with params
        strategy = WickStrategy(config=params)

        # Run backtest
        engine = BacktestEngine(strategy, initial_capital=initial_capital)
        result = engine.run(candles, symbol)

        # Check minimum trades
        if len(result.trades) < MIN_TRADES:
            return None

        metrics = result.metrics

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
            "total_trades": len(result.trades),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "sortino_ratio": metrics.get("sortino_ratio", 0),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
            "total_return_pct": metrics.get("total_return_pct", 0),
            "avg_win": metrics.get("avg_win", 0),
            "avg_loss": metrics.get("avg_loss", 0),
        }

    except Exception as e:
        logger.warning(f"Error testing {symbol} {timeframe}: {e}")
        return None


def calculate_composite_score(result: Dict) -> float:
    """
    Calculate composite score for ranking results.

    Score = 0.3×profit_factor + 0.25×win_rate + 0.25×(1-drawdown) + 0.2×sharpe
    """
    pf = min(result.get("profit_factor", 0), 3.0) / 3.0  # Normalize to 0-1
    wr = result.get("win_rate", 0) / 100  # Already 0-100, convert to 0-1
    dd = 1 - min(result.get("max_drawdown_pct", 100), 100) / 100  # Lower is better

    # Sharpe can be negative, normalize to 0-1 range
    sharpe = result.get("sharpe_ratio", 0)
    sharpe_norm = max(0, min(sharpe + 1, 3)) / 3  # -1 to 2 → 0 to 1

    score = (
        0.30 * pf +
        0.25 * wr +
        0.25 * dd +
        0.20 * sharpe_norm
    )

    return score


def run_grid_search(
    symbols: List[str],
    timeframes: List[str],
    dm: DataManager,
    output_file: str = "data/processed/optimization_results.csv",
) -> pd.DataFrame:
    """
    Run full grid search across all parameter combinations.

    Args:
        symbols: List of symbols to test
        timeframes: List of timeframes to test
        dm: DataManager instance
        output_file: Path to save results CSV

    Returns:
        DataFrame with all results, sorted by composite score
    """
    param_combos = calculate_param_combinations()
    total_tests = len(symbols) * len(timeframes) * len(param_combos)

    logger.info(f"Starting grid search:")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Timeframes: {timeframes}")
    logger.info(f"  Parameter combinations: {len(param_combos)}")
    logger.info(f"  Total tests: {total_tests}")

    results = []
    completed = 0

    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"\nTesting {symbol} {timeframe}...")

            for params in param_combos:
                result = run_single_optimization(symbol, timeframe, params, dm)

                if result:
                    result["composite_score"] = calculate_composite_score(result)
                    results.append(result)

                completed += 1
                if completed % 50 == 0:
                    logger.info(f"Progress: {completed}/{total_tests} ({100*completed/total_tests:.1f}%)")

    # Convert to DataFrame
    if not results:
        logger.warning("No valid results found!")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Flatten params dict into columns
    params_df = pd.json_normalize(df["params"])
    df = pd.concat([df.drop("params", axis=1), params_df], axis=1)

    # Sort by composite score
    df = df.sort_values("composite_score", ascending=False)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")

    return df


def print_top_results(df: pd.DataFrame, n: int = 10):
    """Print top N results in a formatted table."""
    if df.empty:
        print("No results to display")
        return

    print("\n" + "="*80)
    print("TOP OPTIMIZATION RESULTS")
    print("="*80)

    cols = [
        "symbol", "timeframe", "profit_factor", "win_rate",
        "max_drawdown_pct", "total_trades", "composite_score",
        "stop_loss_multiplier", "wick_confirmation_ratio"
    ]

    display_df = df.head(n)[cols].copy()
    display_df["profit_factor"] = display_df["profit_factor"].round(2)
    display_df["win_rate"] = display_df["win_rate"].round(1)
    display_df["max_drawdown_pct"] = display_df["max_drawdown_pct"].round(2)
    display_df["composite_score"] = display_df["composite_score"].round(3)

    print(display_df.to_string(index=False))

    # Print best configuration details
    if len(df) > 0:
        best = df.iloc[0]
        print("\n" + "-"*80)
        print("BEST CONFIGURATION:")
        print("-"*80)
        print(f"  Symbol: {best['symbol']}")
        print(f"  Timeframe: {best['timeframe']}")
        print(f"  Stop Loss Multiplier: {best['stop_loss_multiplier']}")
        print(f"  Wick Confirmation Ratio: {best['wick_confirmation_ratio']}")
        print(f"  Trailing Ratio: {best['trailing_ratio']}")
        print(f"  Min Trend Length: {best['min_trend_length']}")
        print(f"\n  Profit Factor: {best['profit_factor']:.2f}")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Max Drawdown: {best['max_drawdown_pct']:.2f}%")
        print(f"  Total Trades: {best['total_trades']}")
        print(f"  Composite Score: {best['composite_score']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Optimize wick strategy parameters")
    parser.add_argument("--symbol", type=str, help="Single symbol to test")
    parser.add_argument("--timeframe", type=str, default="30Min", help="Timeframe")
    parser.add_argument("--grid-search", action="store_true", help="Run full grid search")
    parser.add_argument("--multi-asset", action="store_true", help="Test multiple assets")
    parser.add_argument("--output", type=str, default="data/processed/optimization_results.csv",
                       help="Output file for results")

    args = parser.parse_args()

    # Initialize data manager
    dm = DataManager()

    # Determine symbols to test
    if args.multi_asset:
        symbols = STOCK_SYMBOLS + CRYPTO_SYMBOLS
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = ["QQQ"]  # Default to best-performing asset

    # Determine timeframes
    if args.multi_asset or args.grid_search:
        timeframes = TIMEFRAMES
    else:
        timeframes = [args.timeframe]

    # Run optimization
    logger.info("="*60)
    logger.info("WICK STRATEGY PARAMETER OPTIMIZATION")
    logger.info("="*60)

    results = run_grid_search(
        symbols=symbols,
        timeframes=timeframes,
        dm=dm,
        output_file=args.output,
    )

    # Display results
    print_top_results(results)

    # Summary statistics
    if not results.empty:
        profitable = results[results["profit_factor"] > 1.0]
        target_pf = results[results["profit_factor"] >= 1.3]

        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Total configurations tested: {len(results)}")
        print(f"Profitable (PF > 1.0): {len(profitable)} ({100*len(profitable)/len(results):.1f}%)")
        print(f"Target PF >= 1.3: {len(target_pf)} ({100*len(target_pf)/len(results):.1f}%)")

        if len(profitable) > 0:
            print(f"\nBest Profit Factor: {results['profit_factor'].max():.2f}")
            print(f"Best Win Rate: {results['win_rate'].max():.1f}%")


if __name__ == "__main__":
    main()
