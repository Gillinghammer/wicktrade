"""
Portfolio tracking for backtesting.
Manages positions, cash, and trade history.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from wicktrade.core.types import (
    Position, Signal, TradeOutcome, Account, PositionSide, ExitReason,
    Candle
)
from .fees import AlpacaFeeModel

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Tracks portfolio state during backtesting.

    Manages:
    - Cash balance
    - Open positions
    - Trade history
    - Equity curve
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        fee_model: Optional[AlpacaFeeModel] = None,
    ):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting cash balance
            fee_model: Fee model for calculating trading costs
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.fee_model = fee_model or AlpacaFeeModel()

        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeOutcome] = []
        self.equity_history: List[tuple] = []  # [(timestamp, equity), ...]

        # Track max/min prices during positions for MAE/MFE
        self._position_extremes: Dict[str, dict] = {}

    @property
    def portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions)."""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value

    @property
    def buying_power(self) -> float:
        """Available buying power (cash for now, could add margin)."""
        return self.cash

    def get_account(self, timestamp: datetime) -> Account:
        """Get current account state as Account object."""
        return Account(
            cash=self.cash,
            buying_power=self.buying_power,
            portfolio_value=self.portfolio_value,
            day_trade_count=0,  # Not tracking for backtest
            positions=self.positions.copy(),
            timestamp=timestamp,
        )

    def open_position(
        self,
        symbol: str,
        signal: Signal,
        shares: float,
        fill_price: float,
        timestamp: datetime,
    ) -> bool:
        """
        Open a new position.

        Args:
            symbol: Stock symbol
            signal: Signal that triggered the entry
            shares: Number of shares
            fill_price: Execution price
            timestamp: Entry time

        Returns:
            True if position opened successfully
        """
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return False

        # Calculate fees
        fees = self.fee_model.calculate_fees("buy", shares, fill_price)
        total_cost = shares * fill_price + fees["total"]

        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for {symbol}: need {total_cost}, have {self.cash}")
            return False

        # Deduct cost
        self.cash -= total_cost

        # Create position
        position = Position(
            symbol=symbol,
            side=PositionSide.LONG,
            quantity=shares,
            avg_price=fill_price,
            market_value=shares * fill_price,
            unrealized_pnl=0.0,
            entry_time=timestamp,
            signal=signal,
            partial_exit_taken=False,
            trailing_stop_price=None,
        )

        self.positions[symbol] = position

        # Initialize extremes tracking
        self._position_extremes[symbol] = {
            "max_price": fill_price,
            "min_price": fill_price,
        }

        logger.debug(
            f"Opened position: {symbol} {shares} shares @ {fill_price:.2f} "
            f"(fees: {fees['total']:.2f})"
        )

        return True

    def close_position(
        self,
        symbol: str,
        fill_price: float,
        timestamp: datetime,
        exit_reason: str,
        partial_pct: float = 1.0,
    ) -> Optional[TradeOutcome]:
        """
        Close a position (fully or partially).

        Args:
            symbol: Stock symbol
            fill_price: Execution price
            timestamp: Exit time
            exit_reason: Reason for exit
            partial_pct: Percentage to close (1.0 = full)

        Returns:
            TradeOutcome if position closed, None otherwise
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None

        position = self.positions[symbol]
        close_shares = position.quantity * partial_pct

        # Calculate fees
        fees = self.fee_model.calculate_fees("sell", close_shares, fill_price)
        proceeds = close_shares * fill_price - fees["total"]

        # Add proceeds to cash
        self.cash += proceeds

        # Calculate PnL
        entry_cost = close_shares * position.avg_price
        pnl = proceeds - entry_cost
        pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0

        # Get extremes for MAE/MFE
        extremes = self._position_extremes.get(symbol, {})
        max_price = extremes.get("max_price", fill_price)
        min_price = extremes.get("min_price", fill_price)

        # Calculate MAE and MFE
        mae = ((min_price - position.avg_price) / position.avg_price) * 100
        mfe = ((max_price - position.avg_price) / position.avg_price) * 100

        # Calculate hold time
        hold_seconds = int((timestamp - position.entry_time).total_seconds())

        # Map exit reason string to enum
        reason_map = {
            "initial_target": ExitReason.INITIAL_TARGET,
            "trailing_stop": ExitReason.TRAILING_STOP,
            "max_target": ExitReason.MAX_TARGET,
            "stop_loss": ExitReason.STOP_LOSS,
            "trend_broken": ExitReason.TREND_BROKEN,
            "time_limit": ExitReason.TIME_LIMIT,
            "manual": ExitReason.MANUAL,
        }
        exit_reason_enum = reason_map.get(exit_reason, ExitReason.MANUAL)

        # Create trade outcome
        trade = TradeOutcome(
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.avg_price,
            exit_price=fill_price,
            quantity=close_shares,
            side=position.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            max_adverse_excursion=mae,
            max_favorable_excursion=mfe,
            hold_time_seconds=hold_seconds,
            exit_reason=exit_reason_enum,
            strategy_name=position.signal.strategy_name if position.signal else "unknown",
            fees=fees["total"],
        )

        self.trades.append(trade)

        # Update or remove position
        if partial_pct >= 1.0:
            del self.positions[symbol]
            del self._position_extremes[symbol]
        else:
            position.quantity -= close_shares
            position.market_value = position.quantity * fill_price
            position.partial_exit_taken = True

        logger.debug(
            f"Closed {'partial ' if partial_pct < 1 else ''}{symbol}: "
            f"{close_shares} shares @ {fill_price:.2f} "
            f"PnL: {pnl:.2f} ({pnl_pct:.2f}%) "
            f"Reason: {exit_reason}"
        )

        return trade

    def update_position_price(
        self,
        symbol: str,
        current_price: float,
    ) -> None:
        """
        Update position with current market price.

        Args:
            symbol: Stock symbol
            current_price: Current market price
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Update market value and unrealized PnL
        position.market_value = position.quantity * current_price
        position.unrealized_pnl = (current_price - position.avg_price) * position.quantity

        # Track extremes for MAE/MFE
        if symbol in self._position_extremes:
            self._position_extremes[symbol]["max_price"] = max(
                self._position_extremes[symbol]["max_price"], current_price
            )
            self._position_extremes[symbol]["min_price"] = min(
                self._position_extremes[symbol]["min_price"], current_price
            )

    def record_equity(self, timestamp: datetime) -> None:
        """Record current equity for equity curve."""
        self.equity_history.append((timestamp, self.portfolio_value))

    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol."""
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def get_summary(self) -> dict:
        """Get portfolio summary statistics."""
        total_pnl = sum(t.pnl for t in self.trades)
        winning_trades = [t for t in self.trades if t.is_winner]
        losing_trades = [t for t in self.trades if not t.is_winner]

        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.portfolio_value,
            "total_return": ((self.portfolio_value - self.initial_capital) / self.initial_capital) * 100,
            "total_pnl": total_pnl,
            "total_trades": len(self.trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            "total_fees": sum(t.fees for t in self.trades),
        }
