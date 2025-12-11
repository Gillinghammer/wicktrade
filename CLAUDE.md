# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wicktrade is a wick-based trading system for US stocks that exploits predictable lower-wick patterns in uptrend channels. The system identifies trends, measures wick statistics, and trades based on wick pattern confirmation.

## Alpaca MCP Server

An Alpaca MCP server is configured for this project, providing direct access to:
- Market data (bars, quotes, snapshots)
- Account info and positions
- Order placement and management
- Paper trading capabilities

**Status:** Connected to paper trading account.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run backtest
python scripts/run_backtest.py --symbol SPY --timeframe 15Min --start 2024-01-01

# Compare timeframes
python scripts/run_backtest.py --symbol SPY --compare-timeframes

# Download historical data
python scripts/download_data.py --symbols SPY,QQQ,AAPL --timeframe 15Min

# Run tests
pytest tests/ -v

# Check MCP server
claude mcp list
```

## Architecture

```
wicktrade/
├── core/           # Types, broker integration, data management
├── analysis/       # Trend detection, wick statistics
├── strategy/       # BaseStrategy ABC, WickStrategy implementation
├── backtest/       # Backtest engine, portfolio tracking, fees
└── metrics/        # Performance metrics, reporting
```

### Data Flow
1. `DataManager` fetches OHLCV from Alpaca, caches to parquet
2. `TrendDetector` identifies uptrend channels using scipy swing detection
3. `WickAnalyzer` calculates wick statistics within trends
4. `WickStrategy` generates signals based on wick confirmation
5. `BacktestEngine` simulates trades, tracks portfolio
6. `MetricsCalculator` computes Sharpe, drawdown, win rate

### Key Types
- `Candle`: OHLCV data point with wick calculations
- `WickStats`: min/max/avg/std of lower wick percentages
- `TrendChannel`: start/end indices, swing points, trend strength
- `Signal`: entry/exit signals with targets and stops

## Configuration

Strategy parameters in `config/settings.yaml`:
- `min_trend_length`: Minimum candles for valid trend (default: 5)
- `wick_confirmation_ratio`: Current wick >= this ratio of avg_wick (default: 0.5)
- `stop_loss_multiplier`: Stop at N x max_wick below entry (default: 1.5)

## Skills

Claude Code Skills in `.claude/skills/`:
- `market-data`: Data fetching via MCP or Python
- `backtest`: Backtesting workflows and analysis
- `strategy`: Strategy development and paper trading

## Success Criteria

Before paper trading, backtest should achieve:
- Sharpe Ratio > 1.0
- Max Drawdown < 25%
- Win Rate > 50%
- Profit Factor > 1.3
- Minimum 100 trades
