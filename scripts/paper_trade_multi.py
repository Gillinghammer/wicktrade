#!/usr/bin/env python3
"""
Multi-symbol paper trading script.

Runs multiple validated strategies in parallel, each in its own thread.
All 4 configurations that passed both walk-forward and stress tests.

Usage:
    python scripts/paper_trade_multi.py                # Run all validated configs
    python scripts/paper_trade_multi.py --dry-run     # Log signals without trading
"""

import argparse
import csv
import logging
import os
import signal
import sys
import threading
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/logs/paper_trading_multi.log"),
    ]
)
logger = logging.getLogger(__name__)

# All configurations that passed BOTH walk-forward AND stress tests
# Ranked by adjusted profit factor
VALIDATED_CONFIGS = [
    {
        "symbol": "SOL/USD",
        "timeframe": "4Hour",
        "params": {
            "stop_loss_multiplier": 1.0,
            "wick_confirmation_ratio": 0.5,
            "trailing_ratio": 0.6,
            "min_trend_length": 5,
        },
        "adj_pf": 1.25,  # Best performer
    },
    {
        "symbol": "ETH/USD",
        "timeframe": "1Hour",
        "params": {
            "stop_loss_multiplier": 1.0,
            "wick_confirmation_ratio": 0.5,
            "trailing_ratio": 0.6,
            "min_trend_length": 5,
        },
        "adj_pf": 1.06,
    },
    {
        "symbol": "TSLA",
        "timeframe": "30Min",
        "params": {
            "stop_loss_multiplier": 1.0,
            "wick_confirmation_ratio": 0.5,
            "trailing_ratio": 0.6,
            "min_trend_length": 5,
        },
        "adj_pf": 1.04,
    },
    {
        "symbol": "AMD",
        "timeframe": "1Hour",
        "params": {
            "stop_loss_multiplier": 1.0,
            "wick_confirmation_ratio": 0.5,
            "trailing_ratio": 0.6,
            "min_trend_length": 5,
        },
        "adj_pf": 1.02,
    },
]

# Risk parameters - shared across all strategies
RISK_LIMITS = {
    "max_position_pct": 3.0,       # Max 3% per position (lower since running 4 strategies)
    "max_total_exposure_pct": 12.0, # Max 12% total (3% x 4 strategies)
    "max_daily_loss_pct": 2.0,     # Stop all trading if daily loss exceeds 2%
    "max_concurrent_positions": 4,  # One per strategy
    "min_buying_power_pct": 50.0,  # Maintain 50% buying power buffer
}


class SymbolTrader:
    """Trader for a single symbol/strategy."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        params: Dict,
        broker: AlpacaBroker,
        dm: DataManager,
        dry_run: bool = False,
        trade_log_path: Path = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params
        self.broker = broker
        self.dm = dm
        self.dry_run = dry_run
        self.trade_log_path = trade_log_path

        self.is_crypto = "/" in symbol
        self.strategy = WickStrategy(config=params)
        self.logger = logging.getLogger(f"trader.{symbol.replace('/', '_')}")

        self.last_signal_time: Optional[datetime] = None
        self.trade_count = 0

    def get_check_interval(self) -> int:
        """Get appropriate check interval based on timeframe."""
        if "Min" in self.timeframe:
            minutes = int(self.timeframe.replace("Min", ""))
            return max(60, minutes * 30)  # Check every half-period, min 1 minute
        elif "Hour" in self.timeframe:
            hours = int(self.timeframe.replace("Hour", ""))
            return hours * 30 * 60  # Check every half-period
        else:
            return 3600  # 1 hour for daily

    def get_recent_candles(self, limit: int = 100) -> List[Candle]:
        """Fetch recent candles for strategy analysis."""
        end = datetime.utcnow()

        if "Min" in self.timeframe:
            minutes = int(self.timeframe.replace("Min", ""))
            start = end - timedelta(minutes=minutes * limit * 2)
        elif "Hour" in self.timeframe:
            hours = int(self.timeframe.replace("Hour", ""))
            start = end - timedelta(hours=hours * limit * 2)
        else:
            start = end - timedelta(days=limit * 2)

        candles = self.dm.get_candles(
            self.symbol,
            self.timeframe,
            start=start,
            end=end,
            use_cache=False,
        )

        return candles[-limit:] if len(candles) > limit else candles

    def calculate_position_size(self, entry_price: float) -> float:
        """Calculate position size based on risk parameters."""
        account = self.broker.get_account()

        # Max position value (% of portfolio)
        max_position_value = account.portfolio_value * (RISK_LIMITS["max_position_pct"] / 100)

        # Calculate shares
        max_shares = max_position_value / entry_price

        # Ensure buying power available
        buying_power_limit = account.buying_power * 0.2  # Use 20% of available buying power
        bp_shares = buying_power_limit / entry_price

        shares = min(max_shares, bp_shares)

        if self.is_crypto:
            return round(shares, 4)
        else:
            return int(shares)

    def log_trade(self, action: str, quantity: float, price: float,
                  order_id: str = "", pnl: float = 0.0, notes: str = ""):
        """Log trade to shared CSV file."""
        if self.trade_log_path is None:
            return

        account = self.broker.get_account()

        with open(self.trade_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                self.symbol,
                self.timeframe,
                action,
                quantity,
                price,
                order_id,
                pnl,
                account.portfolio_value,
                notes,
            ])

    def check_for_signal(self) -> Optional[Signal]:
        """Check strategy for entry signal."""
        candles = self.get_recent_candles(limit=100)

        if len(candles) < 20:
            self.logger.warning(f"Insufficient candles: {len(candles)}")
            return None

        account = self.broker.get_account()

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
            self.logger.info(f"DRY RUN - Would enter: {signal}")
            self.log_trade(
                action="SIGNAL_ENTRY",
                quantity=0,
                price=signal.entry_price,
                notes=f"DRY RUN: stop={signal.stop_loss:.2f}",
            )
            return None

        quantity = self.calculate_position_size(signal.entry_price)

        if quantity <= 0:
            self.logger.warning("Position size too small, skipping entry")
            return None

        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )

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

            self.logger.info(
                f"ENTRY: {quantity} @ ~{signal.entry_price:.2f} "
                f"stop={signal.stop_loss:.2f}"
            )

        return order_id

    def manage_position(self) -> bool:
        """Manage existing position. Returns True if position was closed."""
        account = self.broker.get_account()

        if self.symbol not in account.positions:
            return False

        position = account.positions[self.symbol]
        candles = self.get_recent_candles(limit=10)

        if not candles:
            return False

        current_price = candles[-1].close

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
            self.logger.info(f"DRY RUN - Would exit @ {price:.2f}, PnL=${pnl:.2f}")
            self.log_trade(
                action="SIGNAL_EXIT",
                quantity=position.quantity,
                price=price,
                pnl=pnl,
                notes=f"DRY RUN: reason={reason}",
            )
            return False

        success = self.broker.close_position(self.symbol)

        if success:
            pnl = (price - position.avg_price) * position.quantity

            self.log_trade(
                action="EXIT",
                quantity=position.quantity,
                price=price,
                pnl=pnl,
                notes=f"reason={reason}",
            )

            self.logger.info(f"EXIT: {position.quantity} @ ~{price:.2f} PnL=${pnl:.2f}")

        return success

    def run_iteration(self):
        """Run single iteration of trading loop."""
        account = self.broker.get_account()

        # Check if we have a position to manage
        if self.symbol in account.positions:
            self.manage_position()
            return

        # Check market hours for stocks
        if not self.is_crypto:
            if not self.broker.is_market_open():
                return

        # Look for entry signal
        signal = self.check_for_signal()

        if signal:
            self.logger.info(f"Signal detected: {signal.signal_type.value}")
            self.execute_entry(signal)


class MultiSymbolTrader:
    """Manages multiple symbol traders in parallel."""

    def __init__(self, configs: List[Dict], dry_run: bool = False):
        self.configs = configs
        self.dry_run = dry_run
        self.running = False

        # Shared components
        self.broker = AlpacaBroker(paper=True)
        self.dm = DataManager()

        # Trade log
        self.trade_log_path = Path("data/logs/paper_trades_multi.csv")
        self._init_trade_log()

        # Create traders
        self.traders: List[SymbolTrader] = []
        for config in configs:
            trader = SymbolTrader(
                symbol=config["symbol"],
                timeframe=config["timeframe"],
                params=config["params"],
                broker=self.broker,
                dm=self.dm,
                dry_run=dry_run,
                trade_log_path=self.trade_log_path,
            )
            self.traders.append(trader)

        self.threads: List[threading.Thread] = []
        self.starting_portfolio_value = 0.0
        self.daily_pnl = 0.0

    def _init_trade_log(self):
        """Initialize trade log CSV file."""
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.trade_log_path.exists():
            with open(self.trade_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "timeframe", "action", "quantity",
                    "price", "order_id", "pnl", "portfolio_value", "notes"
                ])

    def check_global_risk_limits(self) -> bool:
        """Check global risk limits across all strategies."""
        account = self.broker.get_account()

        # Check daily loss limit
        if self.starting_portfolio_value > 0:
            daily_return_pct = ((account.portfolio_value / self.starting_portfolio_value) - 1) * 100
            if daily_return_pct < -RISK_LIMITS["max_daily_loss_pct"]:
                logger.warning(f"GLOBAL: Daily loss limit hit: {daily_return_pct:.2f}%")
                return False

        # Check total exposure
        total_exposure = sum(
            pos.market_value for pos in account.positions.values()
        )
        exposure_pct = (total_exposure / account.portfolio_value) * 100

        if exposure_pct > RISK_LIMITS["max_total_exposure_pct"]:
            logger.warning(f"GLOBAL: Max exposure reached: {exposure_pct:.1f}%")
            return False

        # Check buying power buffer
        bp_pct = (account.buying_power / account.portfolio_value) * 100
        if bp_pct < RISK_LIMITS["min_buying_power_pct"]:
            logger.warning(f"GLOBAL: Low buying power: {bp_pct:.1f}%")
            return False

        return True

    def run_trader_loop(self, trader: SymbolTrader):
        """Run loop for a single trader."""
        check_interval = trader.get_check_interval()
        trader.logger.info(f"Started - checking every {check_interval}s")

        while self.running:
            try:
                # Check global risk limits
                if not self.check_global_risk_limits():
                    time.sleep(check_interval)
                    continue

                trader.run_iteration()
                time.sleep(check_interval)

            except Exception as e:
                trader.logger.error(f"Error: {e}", exc_info=True)
                time.sleep(check_interval)

        trader.logger.info("Stopped")

    def log_status(self):
        """Log overall status."""
        account = self.broker.get_account()
        positions_str = ", ".join(
            f"{s}({p.quantity:.4f})" for s, p in account.positions.items()
        ) or "None"

        total_trades = sum(t.trade_count for t in self.traders)

        logger.info(
            f"STATUS: PV=${account.portfolio_value:,.2f} "
            f"Cash=${account.cash:,.2f} "
            f"Positions=[{positions_str}] "
            f"TotalTrades={total_trades}"
        )

    def run(self):
        """Run all traders in parallel."""
        self.running = True

        # Ensure log directory exists
        Path("data/logs").mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("MULTI-SYMBOL PAPER TRADING STARTED")
        logger.info("=" * 70)
        logger.info(f"Running {len(self.traders)} strategies:")
        for config in self.configs:
            logger.info(f"  - {config['symbol']} {config['timeframe']} (Adj PF: {config.get('adj_pf', 'N/A')})")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("=" * 70)

        # Get initial account state
        account = self.broker.get_account()
        self.starting_portfolio_value = account.portfolio_value
        logger.info(f"Starting portfolio value: ${account.portfolio_value:,.2f}")
        logger.info(f"Cash: ${account.cash:,.2f}")
        logger.info(f"Buying power: ${account.buying_power:,.2f}")

        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # Start trader threads
        for trader in self.traders:
            thread = threading.Thread(
                target=self.run_trader_loop,
                args=(trader,),
                name=f"trader-{trader.symbol}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

        # Main loop - status logging
        status_interval = 300  # Log status every 5 minutes
        while self.running:
            time.sleep(status_interval)
            if self.running:
                self.log_status()

        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)

        logger.info("Multi-symbol paper trading stopped")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Multi-symbol paper trading")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log signals without executing trades"
    )
    parser.add_argument(
        "--crypto-only", action="store_true",
        help="Only trade crypto (24/7)"
    )
    parser.add_argument(
        "--stocks-only", action="store_true",
        help="Only trade stocks (market hours)"
    )

    args = parser.parse_args()

    # Filter configs if requested
    configs = VALIDATED_CONFIGS.copy()

    if args.crypto_only:
        configs = [c for c in configs if "/" in c["symbol"]]
        logger.info("Running crypto strategies only")
    elif args.stocks_only:
        configs = [c for c in configs if "/" not in c["symbol"]]
        logger.info("Running stock strategies only")

    if not configs:
        logger.error("No strategies to run!")
        return

    # Create and run multi-trader
    trader = MultiSymbolTrader(configs=configs, dry_run=args.dry_run)
    trader.run()


if __name__ == "__main__":
    main()
