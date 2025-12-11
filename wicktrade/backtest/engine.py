"""
Backtesting engine for strategy evaluation.
Simulates trading strategy on historical data.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd

from wicktrade.core.types import (
    Candle, Signal, BacktestConfig, BacktestResult, Account
)
from wicktrade.strategy.base_strategy import BaseStrategy
from .portfolio import Portfolio
from .fees import AlpacaFeeModel

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Processes historical candles sequentially, generating signals
    and simulating trade execution.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 100000.0,
        fee_model: Optional[AlpacaFeeModel] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            strategy: Trading strategy to test
            initial_capital: Starting capital
            fee_model: Fee model for trade costs
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.fee_model = fee_model or AlpacaFeeModel()
        self.portfolio: Optional[Portfolio] = None

    def run(
        self,
        candles: List[Candle],
        symbol: str,
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            candles: List of historical candles
            symbol: Symbol being tested
            config: Backtest configuration

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        if not candles:
            raise ValueError("No candles provided for backtest")

        logger.info(
            f"Starting backtest: {symbol} with {len(candles)} candles "
            f"from {candles[0].timestamp} to {candles[-1].timestamp}"
        )

        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            fee_model=self.fee_model,
        )

        # Create config if not provided
        if config is None:
            config = BacktestConfig(
                symbol=symbol,
                timeframe="unknown",
                start_date=candles[0].timestamp,
                end_date=candles[-1].timestamp,
                initial_capital=self.initial_capital,
                strategy_config=self.strategy.config,
            )

        # Process each candle
        for i, candle in enumerate(candles):
            self._process_candle(candles, i, symbol)

        # Close any remaining positions at end
        if self.portfolio.has_position(symbol):
            last_candle = candles[-1]
            self.portfolio.close_position(
                symbol=symbol,
                fill_price=last_candle.close,
                timestamp=last_candle.timestamp,
                exit_reason="manual",
            )

        # Build equity curve
        equity_curve = pd.Series(
            {ts: eq for ts, eq in self.portfolio.equity_history},
            name="equity"
        )

        # Calculate metrics
        from wicktrade.metrics.calculator import MetricsCalculator
        calc = MetricsCalculator()
        metrics = calc.calculate_all(
            trades=self.portfolio.trades,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
        )

        result = BacktestResult(
            config=config,
            trades=self.portfolio.trades,
            equity_curve=equity_curve,
            metrics=metrics,
        )

        logger.info(
            f"Backtest complete: {len(result.trades)} trades, "
            f"Return: {metrics.get('total_return_pct', 0):.2f}%"
        )

        return result

    def _process_candle(
        self,
        candles: List[Candle],
        current_idx: int,
        symbol: str,
    ) -> None:
        """
        Process a single candle in the backtest.

        Args:
            candles: Full candle list
            current_idx: Current candle index
            symbol: Symbol being traded
        """
        candle = candles[current_idx]

        # Record equity at each bar
        self.portfolio.record_equity(candle.timestamp)

        # Get current account state
        account = self.portfolio.get_account(candle.timestamp)

        # Update position price if we have one
        if self.portfolio.has_position(symbol):
            self.portfolio.update_position_price(symbol, candle.close)

            # Check for exit
            position = self.portfolio.get_position(symbol)
            exit_info = self.strategy.should_exit(position, candles, current_idx, account)

            if exit_info:
                self._execute_exit(symbol, candle, exit_info)

        # Check for entry if no position
        if not self.portfolio.has_position(symbol):
            signal = self.strategy.should_enter(candles, current_idx, account)

            if signal:
                signal.symbol = symbol
                self._execute_entry(signal, candle, account)

    def _execute_entry(
        self,
        signal: Signal,
        candle: Candle,
        account: Account,
    ) -> None:
        """
        Execute entry based on signal.

        Args:
            signal: Entry signal
            candle: Current candle
            account: Current account state
        """
        # Validate signal
        if not self.strategy.validate_signal(signal, account):
            logger.debug(f"Signal validation failed for {signal.symbol}")
            return

        # Calculate position size
        shares = self.strategy.position_size(signal, account)

        if shares <= 0:
            logger.debug(f"Position size is zero for {signal.symbol}")
            return

        # Use close price as fill (next bar open approximation)
        fill_price = candle.close

        # Open position
        success = self.portfolio.open_position(
            symbol=signal.symbol,
            signal=signal,
            shares=shares,
            fill_price=fill_price,
            timestamp=candle.timestamp,
        )

        if success:
            logger.info(
                f"ENTRY: {signal.symbol} {shares} shares @ {fill_price:.2f} "
                f"SL={signal.stop_loss:.2f} T1={signal.initial_target:.2f}"
            )

    def _execute_exit(
        self,
        symbol: str,
        candle: Candle,
        exit_info: Dict[str, Any],
    ) -> None:
        """
        Execute exit based on exit info.

        Args:
            symbol: Symbol to exit
            candle: Current candle
            exit_info: Exit information dict
        """
        fill_price = exit_info.get("exit_price", candle.close)
        exit_pct = exit_info.get("exit_pct", 1.0)
        reason = exit_info.get("reason", "manual")

        trade = self.portfolio.close_position(
            symbol=symbol,
            fill_price=fill_price,
            timestamp=candle.timestamp,
            exit_reason=reason,
            partial_pct=exit_pct,
        )

        if trade:
            logger.info(
                f"EXIT: {symbol} {trade.quantity} shares @ {fill_price:.2f} "
                f"PnL: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%) "
                f"Reason: {reason}"
            )


def run_backtest(
    symbol: str,
    candles: List[Candle],
    strategy: BaseStrategy,
    initial_capital: float = 100000.0,
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        symbol: Stock symbol
        candles: Historical candle data
        strategy: Strategy to test
        initial_capital: Starting capital

    Returns:
        BacktestResult
    """
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=initial_capital,
    )

    return engine.run(candles, symbol)
