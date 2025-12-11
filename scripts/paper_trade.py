#!/usr/bin/env python3
"""
Paper trading script for the validated wick strategy.

Runs the strategy in real-time against Alpaca paper trading account.
Based on validation testing, only ETH/USD 1Hour passed both walk-forward
and stress tests, so this is the recommended configuration.

Usage:
    python scripts/paper_trade.py                    # Run with defaults (ETH/USD 1Hour)
    python scripts/paper_trade.py --symbol NVDA --timeframe 1Hour
    python scripts/paper_trade.py --dry-run          # Log signals without trading
"""

import argparse
import csv
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wicktrade.core.broker import AlpacaBroker
from wicktrade.core.data_manager import DataManager
from wicktrade.core.types import (
    Candle, Signal, Account, Position, Order, OrderSide, OrderType, PositionSide
)
from wicktrade.strategy.wick_strategy import WickStrategy

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/logs/paper_trading.log"),
    ]
)
logger = logging.getLogger(__name__)

# Validated configurations (passed both walk-forward AND stress tests)
VALIDATED_CONFIGS = {
    "ETH/USD": {
        "timeframe": "1Hour",
        "params": {
            "stop_loss_multiplier": 1.0,
            "wick_confirmation_ratio": 0.5,
            "trailing_ratio": 0.6,
            "min_trend_length": 5,
        },
        "expected_pf": 1.06,  # Post-stress-test profit factor
        "expected_wr": 68.7,  # Post-stress-test win rate
    },
}

# Risk parameters
RISK_LIMITS = {
    "max_position_pct": 5.0,      # Max 5% of portfolio in single position
    "max_daily_loss_pct": 2.0,    # Stop trading if daily loss exceeds 2%
    "max_concurrent_positions": 1, # Only one position at a time
    "min_buying_power_pct": 20.0, # Maintain 20% buying power buffer
}


class PaperTrader:
    """Paper trading manager for wick strategy."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        params: Dict,
        dry_run: bool = False,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize paper trader.

        Args:
            symbol: Trading symbol (e.g., 'ETH/USD')
            timeframe: Candle timeframe (e.g., '1Hour')
            params: Strategy parameters
            dry_run: If True, log signals without executing trades
            check_interval_seconds: How often to check for new signals
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params
        self.dry_run = dry_run
        self.check_interval = check_interval_seconds

        # Determine if this is crypto
        self.is_crypto = "/" in symbol

        # Initialize components
        self.broker = AlpacaBroker(paper=True)
        self.dm = DataManager()
        self.strategy = WickStrategy(config=params)

        # State tracking
        self.running = False
        self.last_signal_time: Optional[datetime] = None
        self.daily_pnl = 0.0
        self.starting_portfolio_value = 0.0
        self.trade_count = 0

        # Logging
        self.trade_log_path = Path("data/logs/paper_trades.csv")
        self._init_trade_log()

        logger.info(f"Paper trader initialized: {symbol} {timeframe}")
        logger.info(f"Parameters: {params}")
        logger.info(f"Dry run: {dry_run}")

    def _init_trade_log(self):
        """Initialize trade log CSV file."""
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.trade_log_path.exists():
            with open(self.trade_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "action", "quantity", "price",
                    "order_id", "pnl", "portfolio_value", "notes"
                ])

    def log_trade(
        self,
        action: str,
        quantity: float,
        price: float,
        order_id: str = "",
        pnl: float = 0.0,
        notes: str = "",
    ):
        """Log trade to CSV file."""
        account = self.broker.get_account()

        with open(self.trade_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                self.symbol,
                action,
                quantity,
                price,
                order_id,
                pnl,
                account.portfolio_value,
                notes,
            ])

    def get_recent_candles(self, limit: int = 100) -> List[Candle]:
        """Fetch recent candles for strategy analysis."""
        # Calculate start time based on timeframe and limit
        end = datetime.utcnow()

        if "Min" in self.timeframe:
            minutes = int(self.timeframe.replace("Min", ""))
            start = end - timedelta(minutes=minutes * limit * 2)
        elif "Hour" in self.timeframe:
            hours = int(self.timeframe.replace("Hour", ""))
            start = end - timedelta(hours=hours * limit * 2)
        else:  # Day
            start = end - timedelta(days=limit * 2)

        candles = self.dm.get_candles(
            self.symbol,
            self.timeframe,
            start=start,
            end=end,
            use_cache=False,  # Always fetch fresh data
        )

        # Return only the most recent 'limit' candles
        return candles[-limit:] if len(candles) > limit else candles

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk parameters.

        Uses fixed percentage of portfolio for position sizing.
        """
        account = self.broker.get_account()

        # Max position value (% of portfolio)
        max_position_value = account.portfolio_value * (RISK_LIMITS["max_position_pct"] / 100)

        # Calculate shares based on max position value
        max_shares = max_position_value / entry_price

        # Also ensure we have buying power
        buying_power_limit = account.buying_power * 0.8  # Use 80% of buying power
        bp_shares = buying_power_limit / entry_price

        # Take minimum of the two limits
        shares = min(max_shares, bp_shares)

        # For crypto, allow fractional; for stocks, round down
        if self.is_crypto:
            return round(shares, 4)
        else:
            return int(shares)

    def check_risk_limits(self) -> bool:
        """Check if risk limits allow trading."""
        account = self.broker.get_account()

        # Check daily loss limit
        if self.starting_portfolio_value > 0:
            daily_return_pct = ((account.portfolio_value / self.starting_portfolio_value) - 1) * 100
            if daily_return_pct < -RISK_LIMITS["max_daily_loss_pct"]:
                logger.warning(f"Daily loss limit hit: {daily_return_pct:.2f}%")
                return False

        # Check max concurrent positions
        if len(account.positions) >= RISK_LIMITS["max_concurrent_positions"]:
            logger.info(f"Max positions reached: {len(account.positions)}")
            return False

        # Check buying power buffer
        bp_pct = (account.buying_power / account.portfolio_value) * 100
        if bp_pct < RISK_LIMITS["min_buying_power_pct"]:
            logger.warning(f"Low buying power: {bp_pct:.1f}%")
            return False

        return True

    def check_for_signal(self) -> Optional[Signal]:
        """Check strategy for entry signal."""
        candles = self.get_recent_candles(limit=100)

        if len(candles) < 20:
            logger.warning(f"Insufficient candles: {len(candles)}")
            return None

        # Create mock account for strategy (it checks positions)
        account = self.broker.get_account()

        # Check for entry signal
        signal = self.strategy.should_enter(
            candles,
            current_idx=len(candles) - 1,
            account=account,
        )

        if signal:
            signal.symbol = self.symbol

        return signal

    def execute_entry(self, signal: Signal) -> Optional[str]:
        """Execute entry order based on signal."""
        if self.dry_run:
            logger.info(f"DRY RUN - Would enter: {signal}")
            self.log_trade(
                action="SIGNAL_ENTRY",
                quantity=0,
                price=signal.entry_price,
                notes=f"DRY RUN: stop={signal.stop_loss:.2f}, target={signal.initial_target:.2f}",
            )
            return None

        # Calculate position size
        quantity = self.calculate_position_size(signal.entry_price, signal.stop_loss)

        if quantity <= 0:
            logger.warning("Position size too small, skipping entry")
            return None

        # Create market order
        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )

        # Submit order
        order_id = self.broker.submit_order(order)

        if order_id:
            self.trade_count += 1
            self.last_signal_time = datetime.utcnow()

            self.log_trade(
                action="ENTRY",
                quantity=quantity,
                price=signal.entry_price,
                order_id=order_id,
                notes=f"stop={signal.stop_loss:.2f}, target={signal.initial_target:.2f}",
            )

            logger.info(
                f"ENTRY: {self.symbol} {quantity} @ ~{signal.entry_price:.2f} "
                f"stop={signal.stop_loss:.2f} target={signal.initial_target:.2f}"
            )

        return order_id

    def manage_position(self) -> bool:
        """
        Manage existing position (stop loss, trailing stop, targets).

        Returns:
            True if position was closed, False otherwise
        """
        account = self.broker.get_account()

        if self.symbol not in account.positions:
            return False

        position = account.positions[self.symbol]
        candles = self.get_recent_candles(limit=10)

        if not candles:
            return False

        current_price = candles[-1].close

        # Check for exit conditions using strategy
        exit_signal = self.strategy.should_exit(
            candles,
            current_idx=len(candles) - 1,
            position=position,
            account=account,
        )

        if exit_signal and exit_signal.signal_type.value == "flat":
            return self.execute_exit(position, current_price, exit_signal.rationale or "strategy_exit")

        return False

    def execute_exit(self, position: Position, price: float, reason: str) -> bool:
        """Execute exit order."""
        if self.dry_run:
            pnl = (price - position.avg_price) * position.quantity
            logger.info(f"DRY RUN - Would exit: {position.symbol} @ {price:.2f}, PnL=${pnl:.2f}")
            self.log_trade(
                action="SIGNAL_EXIT",
                quantity=position.quantity,
                price=price,
                pnl=pnl,
                notes=f"DRY RUN: reason={reason}",
            )
            return False

        # Close position via broker
        success = self.broker.close_position(self.symbol)

        if success:
            pnl = (price - position.avg_price) * position.quantity
            self.daily_pnl += pnl

            self.log_trade(
                action="EXIT",
                quantity=position.quantity,
                price=price,
                pnl=pnl,
                notes=f"reason={reason}",
            )

            logger.info(
                f"EXIT: {self.symbol} {position.quantity} @ ~{price:.2f} "
                f"PnL=${pnl:.2f} reason={reason}"
            )

        return success

    def run_iteration(self):
        """Run single iteration of trading loop."""
        account = self.broker.get_account()

        # Update starting value at start of day
        if self.starting_portfolio_value == 0:
            self.starting_portfolio_value = account.portfolio_value

        # Check if we have a position to manage
        if self.symbol in account.positions:
            self.manage_position()
            return

        # Check risk limits before looking for entries
        if not self.check_risk_limits():
            return

        # Look for entry signal
        signal = self.check_for_signal()

        if signal:
            logger.info(f"Signal detected: {signal.signal_type.value} confidence={signal.confidence:.2f}")
            self.execute_entry(signal)

    def run(self):
        """Run paper trading loop."""
        self.running = True

        # Ensure log directory exists
        Path("data/logs").mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("PAPER TRADING STARTED")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("=" * 60)

        # Get initial account state
        account = self.broker.get_account()
        self.starting_portfolio_value = account.portfolio_value
        logger.info(f"Starting portfolio value: ${account.portfolio_value:,.2f}")
        logger.info(f"Cash: ${account.cash:,.2f}")
        logger.info(f"Buying power: ${account.buying_power:,.2f}")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        iteration = 0
        while self.running:
            try:
                iteration += 1
                logger.debug(f"Iteration {iteration}")

                self.run_iteration()

                # Log status periodically
                if iteration % 10 == 0:
                    self._log_status()

                # Wait for next iteration
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                time.sleep(self.check_interval)

        logger.info("Paper trading stopped")

    def _log_status(self):
        """Log current status."""
        account = self.broker.get_account()
        positions_str = ", ".join(account.positions.keys()) if account.positions else "None"

        logger.info(
            f"Status: PV=${account.portfolio_value:,.2f} "
            f"Cash=${account.cash:,.2f} "
            f"Positions=[{positions_str}] "
            f"Trades={self.trade_count} "
            f"DailyPnL=${self.daily_pnl:,.2f}"
        )

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Paper trade the wick strategy")
    parser.add_argument(
        "--symbol", type=str, default="ETH/USD",
        help="Trading symbol (default: ETH/USD - the only validated config)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1Hour",
        help="Candle timeframe (default: 1Hour)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log signals without executing trades"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Check interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--show-validated", action="store_true",
        help="Show validated configurations and exit"
    )

    args = parser.parse_args()

    if args.show_validated:
        print("\nValidated Configurations (passed walk-forward + stress test):\n")
        for symbol, config in VALIDATED_CONFIGS.items():
            print(f"  {symbol} {config['timeframe']}")
            print(f"    Expected PF: {config['expected_pf']:.2f}")
            print(f"    Expected WR: {config['expected_wr']:.1f}%")
            print(f"    Params: {config['params']}")
            print()
        return

    # Get configuration
    if args.symbol in VALIDATED_CONFIGS:
        config = VALIDATED_CONFIGS[args.symbol]
        params = config["params"]
        timeframe = config["timeframe"]
        logger.info(f"Using validated configuration for {args.symbol}")
    else:
        logger.warning(
            f"{args.symbol} is NOT a validated configuration. "
            f"Only ETH/USD 1Hour passed both walk-forward and stress tests. "
            f"Proceeding with default parameters."
        )
        params = {
            "stop_loss_multiplier": 1.0,
            "wick_confirmation_ratio": 0.5,
            "trailing_ratio": 0.6,
            "min_trend_length": 5,
        }
        timeframe = args.timeframe

    # Create and run trader
    trader = PaperTrader(
        symbol=args.symbol,
        timeframe=timeframe,
        params=params,
        dry_run=args.dry_run,
        check_interval_seconds=args.interval,
    )

    trader.run()


if __name__ == "__main__":
    main()
