---
name: backtest
description: Run and analyze backtests for trading strategies. Use when testing strategies, optimizing parameters, comparing timeframes, or generating performance reports.
---

# Backtest Skill

This skill provides patterns for running backtests and analyzing results in the wicktrade system.

## Quick Start

```python
from wicktrade.core.data_manager import DataManager
from wicktrade.strategy.wick_strategy import WickStrategy
from wicktrade.backtest.engine import BacktestEngine
from wicktrade.metrics.reporter import Reporter

# Get data
dm = DataManager()
candles = dm.get_candles("SPY", "15Min", use_cache=True)

# Run backtest
strategy = WickStrategy()
engine = BacktestEngine(strategy, initial_capital=100000)
result = engine.run(candles, "SPY")

# Report
reporter = Reporter()
reporter.print_summary(result)
```

## Data Sources

### Using MCP Server for Quick Data Check

Before running a backtest, verify data availability:
```
Use get_stock_bars for SPY with timeframe 15Min and limit 10 to check data
```

### Using DataManager for Backtesting

```python
dm = DataManager()
candles = dm.get_candles("SPY", "15Min", use_cache=True)
```

## Running Backtests

### Single Symbol/Timeframe

```python
result = engine.run(candles, "SPY")
reporter.print_summary(result)
reporter.print_trades(result.trades)
```

### Multiple Timeframes

```python
from wicktrade.backtest.engine import run_backtest

timeframes = ["5Min", "15Min", "30Min", "1Hour"]
results = {}

for tf in timeframes:
    candles = dm.get_candles("SPY", tf)
    results[tf] = run_backtest("SPY", candles, WickStrategy())

reporter.compare_timeframes(results)
```

### Custom Strategy Config

```python
config = {
    "min_trend_length": 7,
    "wick_confirmation_ratio": 0.6,
    "stop_loss_multiplier": 2.0,
}
strategy = WickStrategy(config=config)
```

## Analyzing Results

### Key Metrics

```python
m = result.metrics

# Performance
print(f"Return: {m['total_return_pct']:.2f}%")
print(f"Sharpe: {m['sharpe_ratio']:.2f}")
print(f"Max DD: {m['max_drawdown_pct']:.2f}%")

# Trade quality
print(f"Win Rate: {m['win_rate']:.1f}%")
print(f"Profit Factor: {m['profit_factor']:.2f}")
```

### Success Criteria Check

```python
reporter.print_success_criteria(result)
```

### Trade Analysis

```python
winners = result.winning_trades
losers = result.losing_trades

for reason, count in m['exit_reasons'].items():
    print(f"{reason}: {count}")
```

## Parameter Optimization

### Grid Search Pattern

```python
from itertools import product

param_grid = {
    "min_trend_length": [3, 5, 7],
    "wick_confirmation_ratio": [0.4, 0.5, 0.6],
    "stop_loss_multiplier": [1.0, 1.5, 2.0],
}

best_sharpe = -float("inf")
best_params = None

for params in product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), params))
    strategy = WickStrategy(config=config)
    result = run_backtest("SPY", candles, strategy)

    if result.metrics["sharpe_ratio"] > best_sharpe:
        best_sharpe = result.metrics["sharpe_ratio"]
        best_params = config
```

## Walk-Forward Analysis

```python
train_end = int(len(candles) * 0.7)
train_candles = candles[:train_end]
test_candles = candles[train_end:]

train_result = run_backtest("SPY", train_candles, strategy)
test_result = run_backtest("SPY", test_candles, strategy)

print(f"In-sample Sharpe: {train_result.metrics['sharpe_ratio']:.2f}")
print(f"Out-of-sample Sharpe: {test_result.metrics['sharpe_ratio']:.2f}")
```

## CLI Scripts

```bash
# Run single backtest
python scripts/run_backtest.py --symbol SPY --timeframe 15Min

# Compare timeframes
python scripts/run_backtest.py --symbol SPY --compare-timeframes

# With custom parameters
python scripts/run_backtest.py --symbol SPY --min-trend-length 7 --show-trades
```

## Debugging Poor Results

| Issue | Cause | Fix |
|-------|-------|-----|
| Low win rate | Trend detection too loose | Increase min_trend_length |
| High drawdown | Stops too wide | Reduce stop_loss_multiplier |
| Few trades | Entry too strict | Lower confirmation_ratio |
| Negative Sharpe | Strategy not viable | Try different timeframe/symbol |

## Exporting Results

```python
import json

report_dict = reporter.generate_report_dict(result)
with open("backtest_result.json", "w") as f:
    json.dump(report_dict, f, indent=2)
```
