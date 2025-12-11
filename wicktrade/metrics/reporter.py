"""
Reporting utilities for backtest results.
Generates formatted output and summaries.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from wicktrade.core.types import BacktestResult, TradeOutcome

logger = logging.getLogger(__name__)


class Reporter:
    """
    Generates formatted reports from backtest results.

    Supports:
    - Console output
    - Trade-by-trade analysis
    - Timeframe comparison
    """

    def __init__(self):
        """Initialize reporter."""
        pass

    def print_summary(self, result: BacktestResult) -> None:
        """
        Print a formatted summary of backtest results.

        Args:
            result: BacktestResult to summarize
        """
        metrics = result.metrics
        config = result.config

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        # Configuration
        print(f"\nSymbol: {config.symbol}")
        print(f"Timeframe: {config.timeframe}")
        print(f"Period: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${metrics.get('initial_capital', 0):,.2f}")

        # Performance
        print("\n--- PERFORMANCE ---")
        print(f"Final Equity: ${metrics.get('final_equity', 0):,.2f}")
        print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Total PnL: ${metrics.get('total_pnl', 0):,.2f}")
        print(f"Total Fees: ${metrics.get('total_fees', 0):,.2f}")

        # Risk Metrics
        print("\n--- RISK METRICS ---")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Annual Volatility: {metrics.get('annual_volatility', 0) * 100:.2f}%")

        # Trade Statistics
        print("\n--- TRADE STATISTICS ---")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Avg Win: ${metrics.get('avg_win', 0):,.2f}")
        print(f"Avg Loss: ${metrics.get('avg_loss', 0):,.2f}")
        print(f"Avg Win/Loss Ratio: {metrics.get('avg_win_loss_ratio', 0):.2f}")

        # Hold Time
        print("\n--- HOLD TIME ---")
        print(f"Avg Hold: {metrics.get('avg_hold_time_minutes', 0):.1f} min")
        print(f"Max Hold: {metrics.get('max_hold_time_minutes', 0):.1f} min")
        print(f"Min Hold: {metrics.get('min_hold_time_minutes', 0):.1f} min")

        # Streaks
        print("\n--- STREAKS ---")
        print(f"Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}")
        print(f"Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}")

        # Exit Reasons
        exit_reasons = metrics.get("exit_reasons", {})
        if exit_reasons:
            print("\n--- EXIT REASONS ---")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
                pct = count / metrics.get("total_trades", 1) * 100
                print(f"{reason}: {count} ({pct:.1f}%)")

        print("\n" + "=" * 60 + "\n")

    def print_trades(
        self,
        trades: List[TradeOutcome],
        limit: int = 20,
    ) -> None:
        """
        Print trade-by-trade details.

        Args:
            trades: List of trades to display
            limit: Maximum number of trades to show
        """
        print("\n" + "-" * 80)
        print("TRADE DETAILS (most recent)")
        print("-" * 80)

        header = (
            f"{'Entry':<20} {'Exit':<20} {'Entry$':>10} {'Exit$':>10} "
            f"{'PnL':>10} {'PnL%':>8} {'Reason':<15}"
        )
        print(header)
        print("-" * 80)

        recent_trades = trades[-limit:] if len(trades) > limit else trades

        for trade in recent_trades:
            entry_str = trade.entry_time.strftime("%Y-%m-%d %H:%M")
            exit_str = trade.exit_time.strftime("%Y-%m-%d %H:%M")
            reason = trade.exit_reason.value if hasattr(trade.exit_reason, "value") else str(trade.exit_reason)

            row = (
                f"{entry_str:<20} {exit_str:<20} "
                f"${trade.entry_price:>9.2f} ${trade.exit_price:>9.2f} "
                f"${trade.pnl:>9.2f} {trade.pnl_pct:>7.2f}% "
                f"{reason:<15}"
            )
            print(row)

        print("-" * 80)
        print(f"Showing {len(recent_trades)} of {len(trades)} trades\n")

    def compare_timeframes(
        self,
        results: Dict[str, BacktestResult],
    ) -> None:
        """
        Print comparison across multiple timeframes.

        Args:
            results: Dict mapping timeframe to BacktestResult
        """
        print("\n" + "=" * 100)
        print("TIMEFRAME COMPARISON")
        print("=" * 100)

        header = (
            f"{'Timeframe':<10} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>8} "
            f"{'Trades':>8} {'WinRate':>8} {'PF':>8} {'AvgHold':>10}"
        )
        print(header)
        print("-" * 100)

        for tf, result in sorted(results.items()):
            m = result.metrics
            row = (
                f"{tf:<10} {m.get('total_return_pct', 0):>10.2f} "
                f"{m.get('sharpe_ratio', 0):>8.2f} {m.get('max_drawdown_pct', 0):>8.2f} "
                f"{m.get('total_trades', 0):>8} {m.get('win_rate', 0):>7.1f}% "
                f"{m.get('profit_factor', 0):>8.2f} "
                f"{m.get('avg_hold_time_minutes', 0):>9.1f}m"
            )
            print(row)

        print("=" * 100 + "\n")

    def generate_report_dict(self, result: BacktestResult) -> Dict:
        """
        Generate a dictionary report suitable for JSON serialization.

        Args:
            result: BacktestResult

        Returns:
            Dict with all report data
        """
        return {
            "config": {
                "symbol": result.config.symbol,
                "timeframe": result.config.timeframe,
                "start_date": result.config.start_date.isoformat(),
                "end_date": result.config.end_date.isoformat(),
                "initial_capital": result.config.initial_capital,
            },
            "metrics": result.metrics,
            "trades": [
                {
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason.value if hasattr(t.exit_reason, "value") else str(t.exit_reason),
                    "hold_time_minutes": t.hold_time_minutes,
                }
                for t in result.trades
            ],
        }

    def print_success_criteria(self, result: BacktestResult) -> None:
        """
        Print assessment against success criteria.

        Args:
            result: BacktestResult to assess
        """
        m = result.metrics

        print("\n" + "=" * 50)
        print("SUCCESS CRITERIA CHECK")
        print("=" * 50)

        criteria = [
            ("Sharpe Ratio > 1.0", m.get("sharpe_ratio", 0) > 1.0, m.get("sharpe_ratio", 0)),
            ("Max Drawdown < 25%", m.get("max_drawdown_pct", 100) < 25, m.get("max_drawdown_pct", 0)),
            ("Win Rate > 50%", m.get("win_rate", 0) > 50, m.get("win_rate", 0)),
            ("Profit Factor > 1.3", m.get("profit_factor", 0) > 1.3, m.get("profit_factor", 0)),
            ("Trades >= 100", m.get("total_trades", 0) >= 100, m.get("total_trades", 0)),
        ]

        passed = 0
        for name, success, value in criteria:
            status = "PASS" if success else "FAIL"
            symbol = "✓" if success else "✗"
            print(f"  {symbol} {name}: {value:.2f} [{status}]")
            if success:
                passed += 1

        print("-" * 50)
        print(f"  Passed: {passed}/{len(criteria)}")

        if passed == len(criteria):
            print("\n  ★ All criteria met! Ready for paper trading.")
        else:
            print(f"\n  ⚠ {len(criteria) - passed} criteria not met. Continue optimizing.")

        print("=" * 50 + "\n")
