#!/usr/bin/env python3
"""
Walk-forward validation for the wick trading strategy.

Splits data into train/test periods to validate that optimized parameters
work on unseen data. This helps detect overfitting.

Usage:
    python scripts/walk_forward.py --symbol NVDA --timeframe 1Hour
    python scripts/walk_forward.py --symbol QQQ --timeframe 30Min
    python scripts/walk_forward.py --all-best  # Test all best configs from optimization
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wicktrade.core.data_manager import DataManager
from wicktrade.strategy.wick_strategy import WickStrategy
from wicktrade.backtest.engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Best configurations from optimization results
BEST_CONFIGS = [
    {
        "symbol": "NVDA",
        "timeframe": "1Hour",
        "params": {
            "stop_loss_multiplier": 1.2,
            "wick_confirmation_ratio": 0.5,
            "trailing_ratio": 0.8,
            "min_trend_length": 5,
        }
    },
    {
        "symbol": "QQQ",
        "timeframe": "30Min",
        "params": {
            "stop_loss_multiplier": 1.0,
            "wick_confirmation_ratio": 0.5,
            "trailing_ratio": 0.6,
            "min_trend_length": 5,
        }
    },
    {
        "symbol": "AAPL",
        "timeframe": "1Hour",
        "params": {
            "stop_loss_multiplier": 1.1,
            "wick_confirmation_ratio": 0.5,
            "trailing_ratio": 0.7,
            "min_trend_length": 5,
        }
    },
    {
        "symbol": "ETH/USD",
        "timeframe": "1Hour",
        "params": {
            "stop_loss_multiplier": 1.0,
            "wick_confirmation_ratio": 0.5,
            "trailing_ratio": 0.6,
            "min_trend_length": 5,
        }
    },
]


def run_backtest_on_candles(
    candles: List,
    params: Dict,
    initial_capital: float = 100000.0,
) -> Optional[Dict]:
    """
    Run backtest on a list of candles with given parameters.

    Returns:
        Dict with metrics if successful, None otherwise.
    """
    if len(candles) < 100:
        return None

    try:
        strategy = WickStrategy(config=params)
        engine = BacktestEngine(strategy, initial_capital=initial_capital)
        result = engine.run(candles, "TEST")

        if len(result.trades) < 10:  # Minimum trades for validity
            return None

        metrics = result.metrics

        return {
            "total_trades": len(result.trades),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "sortino_ratio": metrics.get("sortino_ratio", 0),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
            "total_return_pct": metrics.get("total_return_pct", 0),
        }

    except Exception as e:
        logger.warning(f"Error in backtest: {e}")
        return None


def walk_forward_validation(
    symbol: str,
    timeframe: str,
    params: Dict,
    dm: DataManager,
    train_ratio: float = 0.7,
    initial_capital: float = 100000.0,
) -> Dict:
    """
    Perform walk-forward validation on a symbol/timeframe/params combination.

    Args:
        symbol: Stock or crypto symbol
        timeframe: Timeframe string
        params: Strategy parameters to test
        dm: DataManager instance
        train_ratio: Fraction of data for training (default 70%)
        initial_capital: Starting capital

    Returns:
        Dict containing train metrics, test metrics, and overfit ratio
    """
    logger.info(f"Running walk-forward validation for {symbol} {timeframe}")
    logger.info(f"Parameters: {params}")

    # Get all candles
    candles = dm.get_candles(symbol, timeframe, use_cache=True)

    if len(candles) < 200:
        logger.warning(f"Insufficient data: {len(candles)} candles")
        return {"error": "Insufficient data"}

    # Split into train/test
    split_idx = int(len(candles) * train_ratio)
    train_candles = candles[:split_idx]
    test_candles = candles[split_idx:]

    train_start = train_candles[0].timestamp if train_candles else None
    train_end = train_candles[-1].timestamp if train_candles else None
    test_start = test_candles[0].timestamp if test_candles else None
    test_end = test_candles[-1].timestamp if test_candles else None

    logger.info(f"Train period: {train_start} to {train_end} ({len(train_candles)} candles)")
    logger.info(f"Test period: {test_start} to {test_end} ({len(test_candles)} candles)")

    # Run backtest on train period
    train_metrics = run_backtest_on_candles(train_candles, params, initial_capital)

    # Run backtest on test period
    test_metrics = run_backtest_on_candles(test_candles, params, initial_capital)

    if train_metrics is None:
        logger.warning("Train period backtest failed")
        return {"error": "Train backtest failed"}

    if test_metrics is None:
        logger.warning("Test period backtest failed")
        return {"error": "Test backtest failed"}

    # Calculate overfit ratio
    # Use profit factor as primary metric
    train_pf = train_metrics.get("profit_factor", 0)
    test_pf = test_metrics.get("profit_factor", 0)

    if test_pf > 0:
        overfit_ratio = train_pf / test_pf
    else:
        overfit_ratio = float('inf')

    # Calculate Sharpe-based overfit ratio as secondary
    train_sharpe = train_metrics.get("sharpe_ratio", 0)
    test_sharpe = test_metrics.get("sharpe_ratio", 0)

    if test_sharpe > 0:
        sharpe_overfit_ratio = train_sharpe / test_sharpe
    elif train_sharpe <= 0 and test_sharpe <= 0:
        sharpe_overfit_ratio = 1.0  # Both negative is not overfitting
    else:
        sharpe_overfit_ratio = float('inf')

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "params": params,
        "train_period": f"{train_start} to {train_end}",
        "test_period": f"{test_start} to {test_end}",
        "train_candles": len(train_candles),
        "test_candles": len(test_candles),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "overfit_ratio_pf": overfit_ratio,
        "overfit_ratio_sharpe": sharpe_overfit_ratio,
        "validation_passed": overfit_ratio < 1.5 and test_pf > 1.0,
    }

    return result


def print_validation_result(result: Dict):
    """Print formatted validation results."""
    if "error" in result:
        print(f"\nERROR: {result['error']}")
        return

    print("\n" + "="*80)
    print(f"WALK-FORWARD VALIDATION: {result['symbol']} {result['timeframe']}")
    print("="*80)

    print(f"\nParameters:")
    for k, v in result['params'].items():
        print(f"  {k}: {v}")

    print(f"\nData Split:")
    print(f"  Train: {result['train_period']} ({result['train_candles']} candles)")
    print(f"  Test:  {result['test_period']} ({result['test_candles']} candles)")

    print(f"\n{'Metric':<25} {'Train':>12} {'Test':>12} {'Ratio':>12}")
    print("-"*60)

    train = result['train_metrics']
    test = result['test_metrics']

    metrics_to_show = [
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Win Rate (%)", "win_rate", "{:.1f}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
        ("Max Drawdown (%)", "max_drawdown_pct", "{:.2f}"),
        ("Total Return (%)", "total_return_pct", "{:.2f}"),
        ("Total Trades", "total_trades", "{:.0f}"),
    ]

    for label, key, fmt in metrics_to_show:
        train_val = train.get(key, 0)
        test_val = test.get(key, 0)

        if test_val != 0 and key in ["profit_factor", "sharpe_ratio"]:
            ratio = train_val / test_val if test_val > 0 else float('inf')
            ratio_str = f"{ratio:.2f}" if ratio < 100 else "inf"
        else:
            ratio_str = "-"

        print(f"{label:<25} {fmt.format(train_val):>12} {fmt.format(test_val):>12} {ratio_str:>12}")

    print("-"*60)

    # Validation verdict
    passed = result['validation_passed']
    overfit_ratio = result['overfit_ratio_pf']
    test_pf = test.get('profit_factor', 0)

    print(f"\nOVERFIT RATIO (PF): {overfit_ratio:.2f}")
    print(f"TEST PROFIT FACTOR: {test_pf:.2f}")

    if passed:
        print("\n✅ VALIDATION PASSED")
        print("   - Overfit ratio < 1.5")
        print("   - Test profit factor > 1.0")
    else:
        print("\n❌ VALIDATION FAILED")
        if overfit_ratio >= 1.5:
            print(f"   - Overfit ratio {overfit_ratio:.2f} >= 1.5 (OVERFITTING DETECTED)")
        if test_pf <= 1.0:
            print(f"   - Test profit factor {test_pf:.2f} <= 1.0 (NOT PROFITABLE ON UNSEEN DATA)")


def run_all_validations(dm: DataManager) -> pd.DataFrame:
    """Run validation on all best configurations."""
    results = []

    for config in BEST_CONFIGS:
        result = walk_forward_validation(
            symbol=config["symbol"],
            timeframe=config["timeframe"],
            params=config["params"],
            dm=dm,
        )

        print_validation_result(result)

        if "error" not in result:
            results.append({
                "symbol": result["symbol"],
                "timeframe": result["timeframe"],
                "train_pf": result["train_metrics"]["profit_factor"],
                "test_pf": result["test_metrics"]["profit_factor"],
                "train_wr": result["train_metrics"]["win_rate"],
                "test_wr": result["test_metrics"]["win_rate"],
                "overfit_ratio": result["overfit_ratio_pf"],
                "passed": result["validation_passed"],
            })

    if results:
        df = pd.DataFrame(results)

        # Save results
        output_path = Path("data/processed/walk_forward_results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        # Print summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        passed_count = df["passed"].sum()
        total_count = len(df)
        print(f"\nConfigurations tested: {total_count}")
        print(f"Validations passed: {passed_count} ({100*passed_count/total_count:.1f}%)")

        print(f"\n{'Symbol':<12} {'Timeframe':<10} {'Train PF':>10} {'Test PF':>10} {'Ratio':>10} {'Status':>10}")
        print("-"*70)
        for _, row in df.iterrows():
            status = "✅ PASS" if row["passed"] else "❌ FAIL"
            print(f"{row['symbol']:<12} {row['timeframe']:<10} {row['train_pf']:>10.2f} {row['test_pf']:>10.2f} {row['overfit_ratio']:>10.2f} {status:>10}")

        return df

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation for wick strategy")
    parser.add_argument("--symbol", type=str, help="Symbol to test")
    parser.add_argument("--timeframe", type=str, default="1Hour", help="Timeframe")
    parser.add_argument("--all-best", action="store_true", help="Test all best configurations")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train/test split ratio")

    args = parser.parse_args()

    # Initialize data manager
    dm = DataManager()

    if args.all_best:
        # Test all best configurations
        run_all_validations(dm)
    elif args.symbol:
        # Find matching config or use defaults
        matching_config = None
        for config in BEST_CONFIGS:
            if config["symbol"] == args.symbol and config["timeframe"] == args.timeframe:
                matching_config = config
                break

        if matching_config:
            params = matching_config["params"]
        else:
            # Use default params
            params = {
                "stop_loss_multiplier": 1.0,
                "wick_confirmation_ratio": 0.5,
                "trailing_ratio": 0.7,
                "min_trend_length": 5,
            }

        result = walk_forward_validation(
            symbol=args.symbol,
            timeframe=args.timeframe,
            params=params,
            dm=dm,
            train_ratio=args.train_ratio,
        )

        print_validation_result(result)
    else:
        print("Please specify --symbol or --all-best")
        parser.print_help()


if __name__ == "__main__":
    main()
