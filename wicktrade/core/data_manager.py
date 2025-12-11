"""
Data manager for fetching and caching historical market data.
Uses Alpaca API with local parquet file caching for efficiency.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytz

from .broker import AlpacaBroker
from .types import Candle

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages historical market data with local caching.

    Fetches data from Alpaca and caches to parquet files to minimize
    API calls and enable faster backtesting.
    """

    def __init__(
        self,
        broker: Optional[AlpacaBroker] = None,
        cache_dir: str = "data/raw",
    ):
        """
        Initialize data manager.

        Args:
            broker: AlpacaBroker instance (creates new one if not provided)
            cache_dir: Directory for cached data files
        """
        self.broker = broker or AlpacaBroker()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataManager initialized with cache dir: {self.cache_dir}")

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path for symbol/timeframe combination."""
        # Replace / with _ for crypto symbols (e.g., BTC/USD -> BTC_USD)
        safe_symbol = symbol.replace("/", "_")
        return self.cache_dir / f"{safe_symbol}_{timeframe}.parquet"

    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is a crypto pair (contains /)."""
        return "/" in symbol

    def _load_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load cached data if available."""
        path = self._cache_path(symbol, timeframe)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                logger.debug(f"Loaded {len(df)} bars from cache: {path}")
                return df
            except Exception as e:
                logger.warning(f"Error loading cache {path}: {e}")
        return None

    def _save_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Save data to cache."""
        path = self._cache_path(symbol, timeframe)
        try:
            df.to_parquet(path)
            logger.debug(f"Saved {len(df)} bars to cache: {path}")
        except Exception as e:
            logger.warning(f"Error saving cache {path}: {e}")

    def _merge_data(
        self, cached: pd.DataFrame, new: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge cached and new data, removing duplicates."""
        if cached is None or cached.empty:
            return new
        if new is None or new.empty:
            return cached

        # Combine and remove duplicates based on index (timestamp)
        combined = pd.concat([cached, new])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        return combined

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get historical bar data for a symbol.

        Args:
            symbol: Stock symbol (e.g., "SPY")
            timeframe: Timeframe (1Min, 5Min, 15Min, 30Min, 1Hour, 4Hour, 1Day)
            start: Start datetime
            end: End datetime (default: now)
            use_cache: Whether to use/update local cache

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index is datetime
        """
        if end is None:
            end = datetime.now(pytz.UTC)
        elif end.tzinfo is None:
            end = end.replace(tzinfo=pytz.UTC)
        if start is None:
            # Default to ~6 months of data
            start = end - timedelta(days=180)
        elif start.tzinfo is None:
            start = start.replace(tzinfo=pytz.UTC)

        # Try to load from cache first
        cached_df = None
        if use_cache:
            cached_df = self._load_cache(symbol, timeframe)

        # Determine what data we need to fetch
        fetch_start = start
        if cached_df is not None and not cached_df.empty:
            cache_end = cached_df.index.max()
            if isinstance(cache_end, pd.Timestamp):
                cache_end = cache_end.to_pydatetime()

            # Only fetch data newer than cache
            if cache_end > start:
                fetch_start = cache_end

        # Fetch new data from Alpaca
        new_df = None
        if fetch_start < end:
            logger.info(
                f"Fetching {symbol} {timeframe} from {fetch_start} to {end}"
            )
            # Route crypto symbols to crypto client
            if self._is_crypto(symbol):
                new_df = self.broker.get_crypto_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=fetch_start,
                    end=end,
                    limit=10000,
                )
            else:
                new_df = self.broker.get_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=fetch_start,
                    end=end,
                    limit=10000,  # Max allowed
                )

        # Merge cached and new data
        if cached_df is not None and new_df is not None:
            result = self._merge_data(cached_df, new_df)
        elif cached_df is not None:
            result = cached_df
        elif new_df is not None:
            result = new_df
        else:
            logger.warning(f"No data available for {symbol} {timeframe}")
            return pd.DataFrame()

        # Save updated data to cache
        if use_cache and result is not None and not result.empty:
            self._save_cache(result, symbol, timeframe)

        # Filter to requested date range
        result = result[(result.index >= start) & (result.index <= end)]

        logger.info(
            f"Returning {len(result)} bars for {symbol} {timeframe} "
            f"from {result.index.min()} to {result.index.max()}"
        )

        return result

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> List[Candle]:
        """
        Get historical data as list of Candle objects.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe string
            start: Start datetime
            end: End datetime
            use_cache: Whether to use cache

        Returns:
            List of Candle objects
        """
        df = self.get_bars(symbol, timeframe, start, end, use_cache)

        if df.empty:
            return []

        candles = []
        for idx, row in df.iterrows():
            candles.append(Candle(
                timestamp=idx.to_pydatetime() if isinstance(idx, pd.Timestamp) else idx,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            ))

        return candles

    def download_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """
        Download and cache data for multiple symbols.

        Args:
            symbols: List of stock symbols
            timeframe: Timeframe to download
            start: Start datetime
            end: End datetime

        Returns:
            Dict mapping symbol to number of bars downloaded
        """
        results = {}
        for symbol in symbols:
            try:
                df = self.get_bars(symbol, timeframe, start, end, use_cache=True)
                results[symbol] = len(df)
                logger.info(f"Downloaded {len(df)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                results[symbol] = 0

        return results

    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> int:
        """
        Clear cached data files.

        Args:
            symbol: Specific symbol to clear (None for all)
            timeframe: Specific timeframe to clear (None for all)

        Returns:
            Number of files deleted
        """
        deleted = 0
        pattern = "*"
        if symbol and timeframe:
            pattern = f"{symbol}_{timeframe}.parquet"
        elif symbol:
            pattern = f"{symbol}_*.parquet"
        elif timeframe:
            pattern = f"*_{timeframe}.parquet"
        else:
            pattern = "*.parquet"

        for path in self.cache_dir.glob(pattern):
            try:
                path.unlink()
                deleted += 1
                logger.info(f"Deleted cache file: {path}")
            except Exception as e:
                logger.error(f"Error deleting {path}: {e}")

        return deleted

    def get_cache_info(self) -> List[Dict]:
        """
        Get information about cached data files.

        Returns:
            List of dicts with file info (symbol, timeframe, rows, size, modified)
        """
        info = []
        for path in self.cache_dir.glob("*.parquet"):
            try:
                parts = path.stem.split("_")
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = "_".join(parts[1:])
                else:
                    symbol = path.stem
                    timeframe = "unknown"

                df = pd.read_parquet(path)
                stat = path.stat()

                info.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "rows": len(df),
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "start": df.index.min() if not df.empty else None,
                    "end": df.index.max() if not df.empty else None,
                })
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")

        return info


# Type hint import for Dict
from typing import Dict
