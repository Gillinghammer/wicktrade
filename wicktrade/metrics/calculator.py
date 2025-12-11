"""
Performance metrics calculator for backtesting results.
Computes standard trading metrics: Sharpe, drawdown, win rate, etc.
"""

import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from wicktrade.core.types import TradeOutcome

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates trading performance metrics.

    Supports standard metrics:
    - Sharpe Ratio
    - Sortino Ratio
    - Max Drawdown
    - Win Rate
    - Profit Factor
    - Average Win/Loss
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,  # Annual risk-free rate
        trading_days: int = 252,        # Trading days per year
    ):
        """
        Initialize calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            trading_days: Trading days per year for annualization
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def calculate_all(
        self,
        trades: List[TradeOutcome],
        equity_curve: pd.Series,
        initial_capital: float,
    ) -> Dict[str, float]:
        """
        Calculate all metrics.

        Args:
            trades: List of completed trades
            equity_curve: Series of equity values over time
            initial_capital: Starting capital

        Returns:
            Dict of metric name to value
        """
        metrics = {}

        # Basic metrics
        metrics["total_trades"] = len(trades)
        metrics["initial_capital"] = initial_capital

        if not trades:
            metrics["total_return_pct"] = 0.0
            metrics["sharpe_ratio"] = 0.0
            metrics["sortino_ratio"] = 0.0
            metrics["max_drawdown_pct"] = 0.0
            metrics["win_rate"] = 0.0
            metrics["profit_factor"] = 0.0
            metrics["avg_win"] = 0.0
            metrics["avg_loss"] = 0.0
            metrics["avg_win_loss_ratio"] = 0.0
            return metrics

        # Final equity
        final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
        metrics["final_equity"] = final_equity
        metrics["total_return_pct"] = ((final_equity - initial_capital) / initial_capital) * 100

        # Trade-based metrics
        metrics.update(self._calculate_trade_metrics(trades))

        # Equity curve metrics
        if len(equity_curve) > 1:
            metrics.update(self._calculate_equity_metrics(equity_curve))
        else:
            metrics["sharpe_ratio"] = 0.0
            metrics["sortino_ratio"] = 0.0
            metrics["calmar_ratio"] = 0.0

        # Drawdown
        metrics["max_drawdown_pct"] = self.calculate_max_drawdown(equity_curve)

        # Total fees
        metrics["total_fees"] = sum(t.fees for t in trades)

        return metrics

    def _calculate_trade_metrics(self, trades: List[TradeOutcome]) -> Dict[str, float]:
        """Calculate metrics based on individual trades."""
        metrics = {}

        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        # Win rate
        metrics["win_rate"] = (len(winners) / len(trades) * 100) if trades else 0

        # PnL stats
        metrics["total_pnl"] = sum(t.pnl for t in trades)
        metrics["gross_profit"] = sum(t.pnl for t in winners) if winners else 0
        metrics["gross_loss"] = abs(sum(t.pnl for t in losers)) if losers else 0

        # Profit factor
        if metrics["gross_loss"] > 0:
            metrics["profit_factor"] = metrics["gross_profit"] / metrics["gross_loss"]
        else:
            metrics["profit_factor"] = float("inf") if metrics["gross_profit"] > 0 else 0

        # Average win/loss
        metrics["avg_win"] = np.mean([t.pnl for t in winners]) if winners else 0
        metrics["avg_loss"] = np.mean([t.pnl for t in losers]) if losers else 0

        # Win/loss ratio
        if metrics["avg_loss"] != 0:
            metrics["avg_win_loss_ratio"] = abs(metrics["avg_win"] / metrics["avg_loss"])
        else:
            metrics["avg_win_loss_ratio"] = float("inf") if metrics["avg_win"] > 0 else 0

        # Average trade
        metrics["avg_trade_pnl"] = np.mean([t.pnl for t in trades])
        metrics["avg_trade_pnl_pct"] = np.mean([t.pnl_pct for t in trades])

        # Hold time
        metrics["avg_hold_time_minutes"] = np.mean([t.hold_time_minutes for t in trades])
        metrics["max_hold_time_minutes"] = max(t.hold_time_minutes for t in trades)
        metrics["min_hold_time_minutes"] = min(t.hold_time_minutes for t in trades)

        # MAE/MFE
        metrics["avg_mae"] = np.mean([t.max_adverse_excursion for t in trades])
        metrics["avg_mfe"] = np.mean([t.max_favorable_excursion for t in trades])

        # Consecutive wins/losses
        metrics["max_consecutive_wins"] = self._max_consecutive(trades, winner=True)
        metrics["max_consecutive_losses"] = self._max_consecutive(trades, winner=False)

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            reason = t.exit_reason.value if hasattr(t.exit_reason, "value") else str(t.exit_reason)
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        metrics["exit_reasons"] = exit_reasons

        return metrics

    def _calculate_equity_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate metrics based on equity curve."""
        metrics = {}

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        if len(returns) < 2:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
            }

        # Daily risk-free rate
        daily_rf = self.risk_free_rate / self.trading_days

        # Excess returns
        excess_returns = returns - daily_rf

        # Sharpe Ratio
        if returns.std() > 0:
            sharpe = (returns.mean() - daily_rf) / returns.std() * np.sqrt(self.trading_days)
        else:
            sharpe = 0.0
        metrics["sharpe_ratio"] = sharpe

        # Sortino Ratio (uses downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (returns.mean() - daily_rf) / downside_returns.std() * np.sqrt(self.trading_days)
        else:
            sortino = sharpe  # Fall back to Sharpe if no downside
        metrics["sortino_ratio"] = sortino

        # Calmar Ratio (annual return / max drawdown)
        annual_return = returns.mean() * self.trading_days
        max_dd = self.calculate_max_drawdown(equity_curve) / 100  # Convert from %
        if max_dd > 0:
            metrics["calmar_ratio"] = annual_return / max_dd
        else:
            metrics["calmar_ratio"] = float("inf") if annual_return > 0 else 0

        # Volatility
        metrics["annual_volatility"] = returns.std() * np.sqrt(self.trading_days)

        return metrics

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown percentage.

        Args:
            equity_curve: Series of equity values

        Returns:
            Max drawdown as positive percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown at each point
        drawdown = (equity_curve - running_max) / running_max * 100

        # Return max drawdown (as positive value)
        return abs(drawdown.min())

    def _max_consecutive(self, trades: List[TradeOutcome], winner: bool) -> int:
        """Calculate maximum consecutive wins or losses."""
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            if trade.is_winner == winner:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def calculate_sharpe(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sharpe ratio from returns series.

        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year for annualization

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        daily_rf = self.risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf

        return (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
