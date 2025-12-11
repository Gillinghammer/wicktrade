"""
Alpaca broker integration using the alpaca-py SDK.
Handles order execution, position management, and market data fetching.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from .types import (
    Order, Fill, Position, Account, OrderSide, OrderType, PositionSide
)

load_dotenv()
logger = logging.getLogger(__name__)


# Timeframe mapping for alpaca-py
TIMEFRAME_MAP = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "30Min": TimeFrame(30, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
}


class AlpacaBroker:
    """Alpaca broker integration for paper and live trading using alpaca-py SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        """
        Initialize Alpaca broker connection.

        Args:
            api_key: Alpaca API key (defaults to ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (defaults to ALPACA_SECRET_KEY env var)
            paper: Use paper trading (default True)
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables or pass directly."
            )

        # Initialize trading client
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=paper,
        )

        # Initialize data client (for historical data)
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

        # Initialize crypto data client
        self.crypto_client = CryptoHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

        self.paper = paper
        logger.info(f"Alpaca broker initialized (paper={paper})")

    def get_account(self) -> Account:
        """Get current account information."""
        try:
            acct = self.trading_client.get_account()
            positions = self.get_positions()

            return Account(
                cash=float(acct.cash),
                buying_power=float(acct.buying_power),
                portfolio_value=float(acct.portfolio_value),
                day_trade_count=int(acct.daytrade_count),
                positions=positions,
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise

    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            positions = {}

            for pos in alpaca_positions:
                side = PositionSide.LONG if float(pos.qty) > 0 else PositionSide.SHORT
                positions[pos.symbol] = Position(
                    symbol=pos.symbol,
                    side=side,
                    quantity=abs(float(pos.qty)),
                    avg_price=float(pos.avg_entry_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    entry_time=datetime.utcnow(),  # Alpaca doesn't provide entry time
                )

            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

    def submit_order(self, order: Order) -> Optional[str]:
        """
        Submit an order to Alpaca.

        Args:
            order: Order object to submit

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Convert order side
            alpaca_side = (
                AlpacaOrderSide.BUY
                if order.side == OrderSide.BUY
                else AlpacaOrderSide.SELL
            )

            # Create appropriate request based on order type
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order.order_type == OrderType.LIMIT:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order.price,
                )
            else:
                logger.error(f"Unsupported order type: {order.order_type}")
                return None

            # Submit order
            alpaca_order = self.trading_client.submit_order(request)

            logger.info(
                f"Order submitted: {order.symbol} {order.side.value} "
                f"{order.quantity} @ {order.price or 'market'} "
                f"ID={alpaca_order.id}"
            )

            return str(alpaca_order.id)

        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of an order."""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return {
                "id": str(order.id),
                "status": order.status.value,
                "filled_qty": float(order.filled_qty or 0),
                "filled_avg_price": float(order.filled_avg_price or 0),
                "submitted_at": order.submitted_at,
                "filled_at": order.filled_at,
            }
        except Exception as e:
            logger.error(f"Error getting order status {order_id}: {e}")
            return None

    def close_position(self, symbol: str) -> bool:
        """Close a position by symbol."""
        try:
            self.trading_client.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all open positions."""
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            logger.info("All positions closed")
            return True
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical bar data.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe string (1Min, 5Min, 15Min, 30Min, 1Hour, 4Hour, 1Day)
            start: Start datetime (default: calculated from limit)
            end: End datetime (default: now)
            limit: Number of bars to fetch if start not specified

        Returns:
            DataFrame with OHLCV data, or None on error
        """
        try:
            # Get timeframe object
            tf = TIMEFRAME_MAP.get(timeframe)
            if tf is None:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None

            # Calculate start time if not provided
            if end is None:
                end = datetime.utcnow()
            if start is None:
                # Estimate start based on timeframe and limit
                if "Min" in timeframe:
                    minutes = int(timeframe.replace("Min", ""))
                    start = end - timedelta(minutes=minutes * limit * 2)
                elif "Hour" in timeframe:
                    hours = int(timeframe.replace("Hour", ""))
                    start = end - timedelta(hours=hours * limit * 2)
                else:  # Day
                    start = end - timedelta(days=limit * 2)

            # Create request with IEX feed (free tier)
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )

            # Fetch bars
            bars = self.data_client.get_stock_bars(request)

            # Convert to DataFrame
            if symbol in bars.data:
                bar_list = bars.data[symbol]
                data = []
                for bar in bar_list:
                    data.append({
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                    })

                df = pd.DataFrame(data)
                if not df.empty:
                    df.set_index("timestamp", inplace=True)
                    df = df.tail(limit)  # Limit to requested number
                return df

            return None

        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}")
            return None

    def get_crypto_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical crypto bar data.

        Args:
            symbol: Crypto symbol (e.g., "BTC/USD", "ETH/USD", "SOL/USD")
            timeframe: Timeframe string (1Min, 5Min, 15Min, 30Min, 1Hour, 4Hour, 1Day)
            start: Start datetime (default: calculated from limit)
            end: End datetime (default: now)
            limit: Number of bars to fetch if start not specified

        Returns:
            DataFrame with OHLCV data, or None on error
        """
        try:
            # Get timeframe object
            tf = TIMEFRAME_MAP.get(timeframe)
            if tf is None:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None

            # Calculate start time if not provided
            if end is None:
                end = datetime.utcnow()
            if start is None:
                # Estimate start based on timeframe and limit
                if "Min" in timeframe:
                    minutes = int(timeframe.replace("Min", ""))
                    start = end - timedelta(minutes=minutes * limit * 2)
                elif "Hour" in timeframe:
                    hours = int(timeframe.replace("Hour", ""))
                    start = end - timedelta(hours=hours * limit * 2)
                else:  # Day
                    start = end - timedelta(days=limit * 2)

            # Create crypto bars request
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
            )

            # Fetch bars
            bars = self.crypto_client.get_crypto_bars(request)

            # Convert to DataFrame
            if symbol in bars.data:
                bar_list = bars.data[symbol]
                data = []
                for bar in bar_list:
                    data.append({
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),  # Crypto volume can be fractional
                    })

                df = pd.DataFrame(data)
                if not df.empty:
                    df.set_index("timestamp", inplace=True)
                    df = df.tail(limit)  # Limit to requested number
                return df

            return None

        except Exception as e:
            logger.error(f"Error getting crypto bars for {symbol}: {e}")
            return None

    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """Get the most recent bar for a symbol."""
        try:
            bars = self.get_bars(symbol, timeframe="1Min", limit=1)
            if bars is not None and not bars.empty:
                row = bars.iloc[-1]
                return {
                    "timestamp": bars.index[-1],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
            return None
        except Exception as e:
            logger.error(f"Error getting latest bar for {symbol}: {e}")
            return None

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    def get_next_market_open(self) -> Optional[datetime]:
        """Get the next market open time."""
        try:
            clock = self.trading_client.get_clock()
            return clock.next_open
        except Exception as e:
            logger.error(f"Error getting next market open: {e}")
            return None
