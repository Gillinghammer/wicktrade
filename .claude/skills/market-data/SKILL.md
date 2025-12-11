---
name: market-data
description: Fetch and analyze market data using Alpaca API or MCP server. Use when downloading historical data, checking quotes, working with OHLCV candles, or debugging data issues.
---

# Market Data Skill

This skill provides patterns for fetching and managing market data in the wicktrade system.

## Two Ways to Access Data

### 1. Alpaca MCP Server (Recommended for Interactive Use)

The Alpaca MCP server is configured and provides direct access to market data tools:

**Available MCP Tools:**
- `get_stock_bars` - Historical OHLCV data
- `get_stock_quotes` - Real-time quotes
- `get_stock_snapshot` - Current market snapshot
- `get_stock_trades` - Trade-level data
- `get_crypto_bars` - Crypto historical data
- `get_crypto_quotes` - Crypto quotes
- `is_market_open` - Check market status
- `get_calendar` - Trading calendar

**Example MCP Usage:**
```
Use get_stock_bars to fetch SPY data for the last 30 days at 15Min timeframe
```

### 2. Python DataManager (For Backtesting)

```python
from wicktrade.core.data_manager import DataManager
from datetime import datetime, timedelta

dm = DataManager()
candles = dm.get_candles(
    symbol="SPY",
    timeframe="15Min",
    start=datetime.now() - timedelta(days=30),
)
```

## Supported Timeframes

| Timeframe | MCP Arg | Python Arg |
|-----------|---------|------------|
| 1 minute | 1Min | "1Min" |
| 5 minutes | 5Min | "5Min" |
| 15 minutes | 15Min | "15Min" |
| 30 minutes | 30Min | "30Min" |
| 1 hour | 1Hour | "1Hour" |
| 4 hours | 4Hour | "4Hour" |
| 1 day | 1Day | "1Day" |

## MCP Server Patterns

### Get Historical Bars
```
Use get_stock_bars for AAPL with:
- timeframe: 15Min
- start: 2024-01-01
- limit: 1000
```

### Get Real-Time Quote
```
Use get_stock_quotes for SPY to get current bid/ask
```

### Check Market Status
```
Use is_market_open to check if trading is available
```

### Get Trading Calendar
```
Use get_calendar to see upcoming market hours
```

## Python Patterns

### Download Data for Multiple Symbols

```python
dm = DataManager()
symbols = ["SPY", "QQQ", "AAPL", "MSFT"]
results = dm.download_symbols(symbols, timeframe="15Min")
```

### Check Cache Status

```python
cache_info = dm.get_cache_info()
for info in cache_info:
    print(f"{info['symbol']} {info['timeframe']}: {info['rows']} bars")
```

### Clear Cache

```python
dm.clear_cache(symbol="SPY")  # Specific symbol
dm.clear_cache()               # All cached data
```

## Data Validation

```python
# Check for gaps in data
candles = dm.get_candles("SPY", "15Min")
for i in range(1, len(candles)):
    gap = (candles[i].timestamp - candles[i-1].timestamp).total_seconds()
    expected = 15 * 60
    if gap > expected * 2:
        print(f"Gap detected at {candles[i].timestamp}")
```

## When to Use Each Method

| Use Case | Recommended Method |
|----------|-------------------|
| Quick data check | MCP Server |
| Backtesting | Python DataManager |
| Real-time quotes | MCP Server |
| Bulk historical download | Python DataManager |
| Market status check | MCP Server |
| Cached data access | Python DataManager |
