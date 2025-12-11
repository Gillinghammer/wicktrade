---
name: strategy
description: Develop and modify trading strategies. Use when implementing new entry/exit logic, adjusting risk parameters, debugging signals, creating strategy variations, or executing paper trades.
---

# Strategy Development Skill

This skill provides patterns for developing, testing, and deploying trading strategies.

## Strategy Architecture

All strategies inherit from `BaseStrategy`:

```python
from wicktrade.strategy.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def should_enter(self, candles, current_idx, account) -> Optional[Signal]:
        pass

    def should_exit(self, position, candles, current_idx, account) -> Optional[dict]:
        pass

    def position_size(self, signal, account) -> float:
        pass
```

## WickStrategy Configuration

```python
config = {
    # Trend Detection
    "min_trend_length": 5,           # Min candles for valid trend
    "swing_lookback": 3,             # argrelextrema order

    # Entry
    "wick_confirmation_ratio": 0.5,  # Min wick ratio to confirm

    # Exit
    "initial_target_ratio": 1.0,     # Exit 50% at avg_wick
    "trailing_ratio": 0.8,           # Trail at 80% of avg_wick
    "stop_loss_multiplier": 1.5,     # Stop at 1.5x max_wick

    # Risk
    "max_position_pct": 5.0,         # Max 5% per position
    "risk_per_trade_pct": 1.0,       # Risk 1% per trade
}
```

## Paper Trading with MCP

The Alpaca MCP server enables paper trading directly from Claude Code:

### Check Account Status
```
Use get_account_info to view current balance and buying power
```

### View Positions
```
Use get_all_positions to see open positions
Use get_open_position for SPY to check specific holding
```

### Place Orders
```
Use place_stock_order to buy 100 shares of SPY at market
Use place_stock_order to sell 50 shares of AAPL with limit price 180
```

### Order Management
```
Use get_all_orders to view pending orders
Use cancel_order to cancel order ID abc123
Use cancel_all_orders to clear all pending orders
```

### Close Positions
```
Use close_position for SPY to exit completely
Use close_all_positions to liquidate everything
```

## Entry Logic Deep Dive

The WickStrategy entry flow:

1. **Trend Detection**
   ```python
   trend = self.trend_detector.get_active_trend(
       candles, current_idx, trend_type="UPTREND"
   )
   ```

2. **Wick Statistics**
   ```python
   wick_stats = self.wick_analyzer.analyze_trend_wicks(candles, trend)
   ```

3. **Confirmation Check**
   ```python
   is_confirmed = self.wick_analyzer.is_wick_confirmation(
       current_candle, wick_stats, "UPTREND", confirmation_ratio
   )
   ```

4. **Signal Generation**
   ```python
   targets = self.wick_analyzer.get_entry_targets(
       entry_price, wick_stats, "UPTREND"
   )
   ```

## Exit Logic

Exit conditions (checked in order):

1. **Stop Loss** - Hard stop at 1.5x max_wick below entry
2. **Max Target** - Full exit at max_wick height
3. **Initial Target** - Partial (50%) exit at avg_wick height
4. **Trailing Stop** - After partial exit, trail at 80% of avg_wick
5. **Trend Broken** - Exit if trend structure breaks

## Risk Management

### Position Sizing Formula

```
shares = (portfolio × risk_pct) / stop_distance
```

Example:
```python
portfolio = 100000
risk_pct = 0.01   # 1%
entry = 150.00
stop = 148.50
stop_distance = 1.50

risk_amount = 100000 * 0.01  # $1000
shares = 1000 / 1.50         # 666 shares
```

### Using MCP to Check Risk

```
Use get_account_info to check buying_power before trading
Use get_all_positions to verify total exposure
```

## Creating Strategy Variations

### Stricter Entry

```python
class StrictWickStrategy(WickStrategy):
    def should_enter(self, candles, current_idx, account):
        signal = super().should_enter(candles, current_idx, account)
        if signal:
            candle = candles[current_idx]
            avg_vol = sum(c.volume for c in candles[current_idx-20:current_idx]) / 20
            if candle.volume < avg_vol * 1.5:
                return None
        return signal
```

### Multiple Timeframe

```python
class MTFWickStrategy(WickStrategy):
    def __init__(self, higher_tf_candles):
        super().__init__()
        self.htf_candles = higher_tf_candles

    def should_enter(self, candles, current_idx, account):
        htf_trend = self.trend_detector.detect_trends(
            self.htf_candles, "UPTREND"
        )
        if not htf_trend:
            return None
        return super().should_enter(candles, current_idx, account)
```

## Debugging Signals

### Why No Entry?

```python
def debug_entry(candles, idx):
    detector = TrendDetector()
    analyzer = WickAnalyzer()

    trend = detector.get_active_trend(candles, idx, "UPTREND")
    if not trend:
        print("No active uptrend")
        return

    if not trend.is_valid(min_length=5):
        print(f"Trend too short: {trend.length}")
        return

    stats = analyzer.analyze_trend_wicks(candles, trend)
    if not stats.is_valid():
        print(f"Invalid wick stats: count={stats.count}")
        return

    candle = candles[idx]
    current_wick = candle.lower_wick_pct
    required = stats.avg_wick * 0.5
    print(f"Current wick: {current_wick:.3f}%, required: {required:.3f}%")
```

## Live Trading Workflow

1. **Backtest first** - Ensure strategy passes success criteria
2. **Paper trade** - Use MCP server with paper account
3. **Monitor** - Check positions and orders regularly
4. **Scale up** - Gradually increase position sizes

### Paper Trading Commands

```
# Check account
Use get_account_info

# Get signal from analysis, then:
Use place_stock_order to buy 10 shares of SPY at market

# Monitor
Use get_all_positions
Use get_open_position for SPY

# Exit
Use close_position for SPY
```

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Too few trades | min_trend_length too high | Reduce to 3-5 |
| High drawdown | stop_loss_multiplier too high | Reduce to 1.0-1.5 |
| Low win rate | confirmation_ratio too low | Increase to 0.6-0.7 |
| Exits too early | initial_target_ratio too low | Increase to 1.2-1.5 |
