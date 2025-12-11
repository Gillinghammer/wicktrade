#!/usr/bin/env python3
"""
Slippage and commission stress test for the wick trading strategy.

Tests whether the strategy remains profitable when realistic
transaction costs are added.

Usage:
    python scripts/stress_test.py --symbol NVDA --timeframe 1Hour
    python scripts/stress_test.py --all-best
    python scripts/stress_test.py --symbol QQQ --slippage 0.02  # 0.02% slippage
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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

# Slippage scenarios to test
SLIPPAGE_SCENARIOS = [
    {"name": "No costs", "slippage_pct": 0.0, "commission_per_share": 0.0},
    {"name": "Low costs", "slippage_pct": 0.01, "commission_per_share": 0.0},
    {"name": "Medium costs", "slippage_pct": 0.02, "commission_per_share": 0.005},
    {"name": "High costs", "slippage_pct": 0.05, "commission_per_share": 0.01},
    {"name": "Extreme costs", "slippage_pct": 0.10, "commission_per_share": 0.02},
]


def apply_slippage_to_trades(
    trades: List,
    slippage_pct: float,
    commission_per_share: float,
) -> List[Dict]:
    """
    Apply slippage and commission to a list of trades.

    Args:
        trades: List of TradeOutcome objects
        slippage_pct: Slippage as percentage (0.01 = 0.01%)
        commission_per_share: Commission per share in dollars

    Returns:
        List of modified trade dicts with adjusted PnL
    """
    adjusted_trades = []

    for trade in trades:
        # Original values
        entry_price = trade.entry_price
        exit_price = trade.exit_price
        quantity = trade.quantity
        original_pnl = trade.pnl

        # Apply slippage (worse entry, worse exit)
        # Entry slippage: pay more to buy
        slippage_entry = entry_price * (slippage_pct / 100)
        adjusted_entry = entry_price + slippage_entry

        # Exit slippage: receive less when selling
        slippage_exit = exit_price * (slippage_pct / 100)
        adjusted_exit = exit_price - slippage_exit

        # Commission (both entry and exit)
        total_commission = 2 * quantity * commission_per_share

        # Calculate adjusted PnL
        adjusted_pnl = (adjusted_exit - adjusted_entry) * quantity - total_commission

        adjusted_trades.append({
            "original_pnl": original_pnl,
            "adjusted_pnl": adjusted_pnl,
            "slippage_cost": (slippage_entry + slippage_exit) * quantity,
            "commission_cost": total_commission,
            "total_cost": (slippage_entry + slippage_exit) * quantity + total_commission,
            "quantity": quantity,
            "is_winner": adjusted_pnl > 0,
            "original_winner": original_pnl > 0,
        })

    return adjusted_trades


def calculate_adjusted_metrics(adjusted_trades: List[Dict]) -> Dict:
    """Calculate metrics from adjusted trades."""
    if not adjusted_trades:
        return {}

    total_trades = len(adjusted_trades)
    winners = [t for t in adjusted_trades if t["is_winner"]]
    losers = [t for t in adjusted_trades if not t["is_winner"]]

    win_rate = 100 * len(winners) / total_trades if total_trades > 0 else 0

    total_wins = sum(t["adjusted_pnl"] for t in winners)
    total_losses = abs(sum(t["adjusted_pnl"] for t in losers))

    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    net_pnl = sum(t["adjusted_pnl"] for t in adjusted_trades)
    total_slippage = sum(t["slippage_cost"] for t in adjusted_trades)
    total_commission = sum(t["commission_cost"] for t in adjusted_trades)
    total_costs = sum(t["total_cost"] for t in adjusted_trades)

    # Original metrics for comparison
    original_net_pnl = sum(t["original_pnl"] for t in adjusted_trades)
    original_winners = len([t for t in adjusted_trades if t["original_winner"]])
    original_win_rate = 100 * original_winners / total_trades if total_trades > 0 else 0

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "net_pnl": net_pnl,
        "total_slippage": total_slippage,
        "total_commission": total_commission,
        "total_costs": total_costs,
        "original_net_pnl": original_net_pnl,
        "original_win_rate": original_win_rate,
        "pnl_reduction_pct": 100 * (1 - net_pnl / original_net_pnl) if original_net_pnl > 0 else 0,
    }


def run_stress_test(
    symbol: str,
    timeframe: str,
    params: Dict,
    dm: DataManager,
    slippage_scenarios: List[Dict] = None,
    initial_capital: float = 100000.0,
) -> Dict:
    """
    Run stress test with multiple slippage scenarios.

    Returns:
        Dict with results for each scenario
    """
    if slippage_scenarios is None:
        slippage_scenarios = SLIPPAGE_SCENARIOS

    logger.info(f"Running stress test for {symbol} {timeframe}")

    # Get candles and run base backtest
    candles = dm.get_candles(symbol, timeframe, use_cache=True)

    if len(candles) < 100:
        return {"error": "Insufficient data"}

    # Run base backtest
    strategy = WickStrategy(config=params)
    engine = BacktestEngine(strategy, initial_capital=initial_capital)
    result = engine.run(candles, symbol)

    if len(result.trades) < 10:
        return {"error": "Insufficient trades"}

    # Test each slippage scenario
    scenario_results = []

    for scenario in slippage_scenarios:
        adjusted_trades = apply_slippage_to_trades(
            result.trades,
            slippage_pct=scenario["slippage_pct"],
            commission_per_share=scenario["commission_per_share"],
        )

        metrics = calculate_adjusted_metrics(adjusted_trades)
        metrics["scenario_name"] = scenario["name"]
        metrics["slippage_pct"] = scenario["slippage_pct"]
        metrics["commission_per_share"] = scenario["commission_per_share"]
        metrics["still_profitable"] = metrics["profit_factor"] > 1.0

        scenario_results.append(metrics)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "params": params,
        "base_trades": len(result.trades),
        "base_pf": result.metrics.get("profit_factor", 0),
        "base_wr": result.metrics.get("win_rate", 0),
        "scenarios": scenario_results,
    }


def print_stress_test_result(result: Dict):
    """Print formatted stress test results."""
    if "error" in result:
        print(f"\nERROR: {result['error']}")
        return

    print("\n" + "="*90)
    print(f"SLIPPAGE STRESS TEST: {result['symbol']} {result['timeframe']}")
    print("="*90)

    print(f"\nBase backtest: {result['base_trades']} trades, PF={result['base_pf']:.2f}, WR={result['base_wr']:.1f}%")

    print(f"\n{'Scenario':<18} {'Slip%':>8} {'Comm$':>8} {'Net PnL':>12} {'PF':>8} {'WR%':>8} {'Costs':>10} {'Status':>12}")
    print("-"*90)

    for scenario in result['scenarios']:
        status = "✅ PROFIT" if scenario["still_profitable"] else "❌ LOSS"
        print(f"{scenario['scenario_name']:<18} "
              f"{scenario['slippage_pct']:>8.2f} "
              f"{scenario['commission_per_share']:>8.3f} "
              f"${scenario['net_pnl']:>11,.0f} "
              f"{scenario['profit_factor']:>8.2f} "
              f"{scenario['win_rate']:>8.1f} "
              f"${scenario['total_costs']:>9,.0f} "
              f"{status:>12}")

    # Find breakeven point
    profitable_scenarios = [s for s in result['scenarios'] if s["still_profitable"]]
    unprofitable_scenarios = [s for s in result['scenarios'] if not s["still_profitable"]]

    if profitable_scenarios and unprofitable_scenarios:
        max_profitable = max(profitable_scenarios, key=lambda x: x["slippage_pct"])
        min_unprofitable = min(unprofitable_scenarios, key=lambda x: x["slippage_pct"])
        print(f"\nBreakeven point: Between {max_profitable['slippage_pct']}% and {min_unprofitable['slippage_pct']}% slippage")
    elif profitable_scenarios:
        print(f"\n✅ Profitable in ALL scenarios (up to {max(s['slippage_pct'] for s in result['scenarios'])}% slippage)")
    else:
        print(f"\n❌ Unprofitable in ALL scenarios")

    # Recommendation
    medium_scenario = next((s for s in result['scenarios'] if s['scenario_name'] == 'Medium costs'), None)
    if medium_scenario:
        print(f"\n📊 With realistic costs (0.02% slip, $0.005/share): ", end="")
        if medium_scenario["still_profitable"]:
            print(f"PROFITABLE (PF={medium_scenario['profit_factor']:.2f})")
        else:
            print(f"NOT PROFITABLE (PF={medium_scenario['profit_factor']:.2f})")


def run_all_stress_tests(dm: DataManager) -> pd.DataFrame:
    """Run stress tests on all best configurations."""
    results = []

    for config in BEST_CONFIGS:
        result = run_stress_test(
            symbol=config["symbol"],
            timeframe=config["timeframe"],
            params=config["params"],
            dm=dm,
        )

        print_stress_test_result(result)

        if "error" not in result:
            # Get medium costs scenario for summary
            medium = next((s for s in result['scenarios'] if s['scenario_name'] == 'Medium costs'), None)
            if medium:
                results.append({
                    "symbol": result["symbol"],
                    "timeframe": result["timeframe"],
                    "base_pf": result["base_pf"],
                    "adjusted_pf": medium["profit_factor"],
                    "base_wr": result["base_wr"],
                    "adjusted_wr": medium["win_rate"],
                    "total_costs": medium["total_costs"],
                    "pnl_reduction_pct": medium["pnl_reduction_pct"],
                    "still_profitable": medium["still_profitable"],
                })

    if results:
        df = pd.DataFrame(results)

        # Save results
        output_path = Path("data/processed/stress_test_results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        # Print summary
        print("\n" + "="*90)
        print("STRESS TEST SUMMARY (Medium Costs: 0.02% slippage + $0.005/share)")
        print("="*90)

        passed_count = df["still_profitable"].sum()
        total_count = len(df)
        print(f"\nConfigurations tested: {total_count}")
        print(f"Still profitable with costs: {passed_count} ({100*passed_count/total_count:.1f}%)")

        print(f"\n{'Symbol':<12} {'Timeframe':<10} {'Base PF':>10} {'Adj PF':>10} {'Costs':>12} {'PnL Drop':>10} {'Status':>10}")
        print("-"*80)
        for _, row in df.iterrows():
            status = "✅ PASS" if row["still_profitable"] else "❌ FAIL"
            print(f"{row['symbol']:<12} {row['timeframe']:<10} {row['base_pf']:>10.2f} {row['adjusted_pf']:>10.2f} ${row['total_costs']:>11,.0f} {row['pnl_reduction_pct']:>9.1f}% {status:>10}")

        return df

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Slippage stress test for wick strategy")
    parser.add_argument("--symbol", type=str, help="Symbol to test")
    parser.add_argument("--timeframe", type=str, default="1Hour", help="Timeframe")
    parser.add_argument("--all-best", action="store_true", help="Test all best configurations")
    parser.add_argument("--slippage", type=float, help="Custom slippage percentage to test")

    args = parser.parse_args()

    # Initialize data manager
    dm = DataManager()

    if args.all_best:
        run_all_stress_tests(dm)
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
            params = {
                "stop_loss_multiplier": 1.0,
                "wick_confirmation_ratio": 0.5,
                "trailing_ratio": 0.7,
                "min_trend_length": 5,
            }

        # Custom slippage scenario if specified
        if args.slippage:
            scenarios = [
                {"name": "No costs", "slippage_pct": 0.0, "commission_per_share": 0.0},
                {"name": "Custom", "slippage_pct": args.slippage, "commission_per_share": 0.005},
            ]
        else:
            scenarios = None

        result = run_stress_test(
            symbol=args.symbol,
            timeframe=args.timeframe,
            params=params,
            dm=dm,
            slippage_scenarios=scenarios,
        )

        print_stress_test_result(result)
    else:
        print("Please specify --symbol or --all-best")
        parser.print_help()


if __name__ == "__main__":
    main()
