"""
Wick-based trading strategy implementation.
Trades based on lower-wick patterns in uptrend channels.
"""

import logging
from typing import Optional, Dict, Any, List

from wicktrade.core.types import (
    Signal, Position, Account, Candle, SignalType, WickStats, TrendChannel
)
from wicktrade.analysis.trend_detector import TrendDetector
from wicktrade.analysis.wick_analyzer import WickAnalyzer
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class WickStrategy(BaseStrategy):
    """
    Trading strategy based on wick patterns in trend channels.

    Entry Logic (LONG only):
    1. Detect active uptrend (higher highs + higher lows)
    2. Calculate wick statistics for the trend
    3. Enter when current candle shows confirming lower wick

    Exit Logic:
    1. Initial target: Exit 50% at avg_wick height
    2. Trailing: Trail remaining 50% toward max_wick
    3. Stop loss: 1.5x max_wick below entry
    """

    def __init__(self, name: str = "wick_strategy", config: Optional[Dict[str, Any]] = None):
        """
        Initialize wick strategy.

        Args:
            name: Strategy name
            config: Strategy configuration with keys:
                - min_trend_length: Min candles for valid trend (default: 5)
                - swing_lookback: Lookback for swing detection (default: 3)
                - wick_confirmation_ratio: Min wick ratio to confirm (default: 0.5)
                - initial_target_ratio: Exit ratio at avg_wick (default: 1.0)
                - trailing_ratio: Trailing stop ratio (default: 0.8)
                - stop_loss_multiplier: Stop loss multiplier (default: 1.5)
                - max_position_pct: Max position as % of portfolio (default: 5.0)
                - risk_per_trade_pct: Risk per trade as % (default: 1.0)
        """
        default_config = {
            "min_trend_length": 5,
            "swing_lookback": 3,
            "wick_confirmation_ratio": 0.5,
            "initial_target_ratio": 1.0,
            "trailing_ratio": 0.8,
            "stop_loss_multiplier": 1.5,
            "max_position_pct": 5.0,
            "risk_per_trade_pct": 1.0,
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config)

        # Initialize analysis components
        self.trend_detector = TrendDetector(
            lookback=self.config["swing_lookback"],
            min_trend_length=self.config["min_trend_length"],
        )
        self.wick_analyzer = WickAnalyzer()

        # Cache for active trend and wick stats
        self._active_trend: Optional[TrendChannel] = None
        self._wick_stats: Optional[WickStats] = None

    def should_enter(
        self,
        candles: List[Candle],
        current_idx: int,
        account: Account,
    ) -> Optional[Signal]:
        """
        Check if entry conditions are met.

        Entry requires:
        1. Active uptrend detected
        2. Valid wick statistics from trend
        3. Current candle has confirming lower wick
        """
        if not self.is_active:
            return None

        if current_idx < self.config["min_trend_length"]:
            return None

        current_candle = self.get_current_candle(candles, current_idx)
        symbol = "UNKNOWN"  # Will be set by caller

        # Check for already open position (handled by backtest engine)
        if account.positions:
            return None

        # Detect active uptrend
        trend = self.trend_detector.get_active_trend(
            candles,
            current_idx,
            trend_type="UPTREND",
        )

        if trend is None:
            return None

        if not trend.is_valid(self.config["min_trend_length"]):
            return None

        # Calculate wick statistics
        wick_stats = self.wick_analyzer.analyze_trend_wicks(candles, trend)

        if not wick_stats.is_valid():
            return None

        # Check for wick confirmation on current candle
        is_confirmed = self.wick_analyzer.is_wick_confirmation(
            current_candle,
            wick_stats,
            trend_type="UPTREND",
            confirmation_ratio=self.config["wick_confirmation_ratio"],
        )

        if not is_confirmed:
            return None

        # Calculate entry targets
        entry_price = current_candle.close  # Enter at next bar open (approx close)
        targets = self.wick_analyzer.get_entry_targets(
            entry_price,
            wick_stats,
            trend_type="UPTREND",
        )

        # Apply stop loss multiplier from config
        stop_distance = entry_price - targets["stop_loss"]
        adjusted_stop = entry_price - (stop_distance * self.config["stop_loss_multiplier"] / 1.5)

        # Calculate confidence based on trend strength and wick pattern
        confidence = self._calculate_confidence(trend, wick_stats, current_candle)

        # Cache trend and stats for exit logic
        self._active_trend = trend
        self._wick_stats = wick_stats

        signal = Signal(
            symbol=symbol,
            signal_type=SignalType.LONG,
            entry_price=entry_price,
            stop_loss=adjusted_stop,
            initial_target=targets["initial_target"],
            max_target=targets["max_target"],
            confidence=confidence,
            wick_stats=wick_stats,
            trend=trend,
            rationale=self._generate_rationale(trend, wick_stats, current_candle),
            strategy_name=self.name,
        )

        logger.info(
            f"Entry signal generated: {signal.symbol} @ {signal.entry_price:.2f} "
            f"SL={signal.stop_loss:.2f} T1={signal.initial_target:.2f} "
            f"T2={signal.max_target:.2f} conf={signal.confidence:.2f}"
        )

        return signal

    def should_exit(
        self,
        position: Position,
        candles: List[Candle],
        current_idx: int,
        account: Account,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if exit conditions are met.

        Exit conditions:
        1. Stop loss hit
        2. Initial target reached (partial exit)
        3. Trailing stop hit (after initial target)
        4. Max target reached
        5. Trend broken
        """
        if not position.is_open:
            return None

        current_candle = self.get_current_candle(candles, current_idx)
        current_price = current_candle.close

        # Get wick stats from position signal or cached
        wick_stats = position.signal.wick_stats if position.signal else self._wick_stats
        if wick_stats is None:
            logger.warning("No wick stats available for exit check")
            return None

        entry_price = position.avg_price
        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        # Check stop loss (using low for more accurate simulation)
        stop_loss = position.signal.stop_loss if position.signal else entry_price * 0.98
        if current_candle.low <= stop_loss:
            return {
                "exit_type": "FULL",
                "exit_pct": 1.0,
                "exit_price": stop_loss,
                "reason": "stop_loss",
            }

        # Check max target
        max_target = position.signal.max_target if position.signal else entry_price * 1.02
        if current_candle.high >= max_target:
            return {
                "exit_type": "FULL",
                "exit_pct": 1.0,
                "exit_price": max_target,
                "reason": "max_target",
            }

        # Check initial target (partial exit)
        if not position.partial_exit_taken:
            initial_target = position.signal.initial_target if position.signal else entry_price * 1.01
            if current_candle.high >= initial_target:
                return {
                    "exit_type": "PARTIAL",
                    "exit_pct": 0.5,
                    "exit_price": initial_target,
                    "reason": "initial_target",
                }

        # Check trailing stop (after partial exit)
        if position.partial_exit_taken:
            trailing_stop = self.wick_analyzer.calculate_trailing_stop(
                current_price,
                entry_price,
                wick_stats,
                trend_type="UPTREND",
                trailing_ratio=self.config["trailing_ratio"],
            )

            # Update trailing stop if it's higher than current
            if position.trailing_stop_price is None or trailing_stop > position.trailing_stop_price:
                position.trailing_stop_price = trailing_stop

            if current_candle.low <= position.trailing_stop_price:
                return {
                    "exit_type": "FULL",
                    "exit_pct": 1.0,
                    "exit_price": position.trailing_stop_price,
                    "reason": "trailing_stop",
                }

        # Check if trend is broken
        if self._active_trend:
            if not self.trend_detector.is_trend_intact(
                self._active_trend, current_candle, current_idx
            ):
                return {
                    "exit_type": "FULL",
                    "exit_pct": 1.0,
                    "exit_price": current_price,
                    "reason": "trend_broken",
                }

        return None

    def position_size(
        self,
        signal: Signal,
        account: Account,
    ) -> float:
        """
        Calculate position size based on risk parameters.

        Uses risk-based sizing: risk amount / stop distance
        Constrained by max position percentage.
        """
        # Risk amount based on portfolio
        risk_amount = account.portfolio_value * (self.config["risk_per_trade_pct"] / 100)

        # Stop distance
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        if stop_distance <= 0:
            return 0

        # Risk-based position size
        risk_based_size = risk_amount / stop_distance

        # Max position constraint
        max_position_value = account.portfolio_value * (self.config["max_position_pct"] / 100)
        max_size = max_position_value / signal.entry_price

        # Buying power constraint
        bp_size = account.buying_power / signal.entry_price

        # Return minimum of all constraints
        size = min(risk_based_size, max_size, bp_size)

        # Round to whole shares for stocks
        return int(size)

    def _calculate_confidence(
        self,
        trend: TrendChannel,
        wick_stats: WickStats,
        current_candle: Candle,
    ) -> float:
        """
        Calculate signal confidence based on multiple factors.

        Factors:
        - Trend strength (0.4 weight)
        - Wick consistency (0.3 weight)
        - Current wick quality (0.3 weight)
        """
        # Trend strength component
        trend_score = trend.strength

        # Wick consistency (lower std relative to mean is better)
        if wick_stats.avg_wick > 0:
            cv = wick_stats.std_wick / wick_stats.avg_wick  # Coefficient of variation
            wick_consistency = max(0, 1 - cv)  # Lower CV = higher score
        else:
            wick_consistency = 0

        # Current wick quality (closer to avg is better)
        current_wick = current_candle.lower_wick_pct
        if wick_stats.avg_wick > 0:
            wick_ratio = current_wick / wick_stats.avg_wick
            # Ideal ratio is around 1.0, penalize extremes
            wick_quality = max(0, 1 - abs(1 - wick_ratio))
        else:
            wick_quality = 0

        # Weighted average
        confidence = (
            0.4 * trend_score +
            0.3 * wick_consistency +
            0.3 * wick_quality
        )

        return min(1.0, max(0.0, confidence))

    def _generate_rationale(
        self,
        trend: TrendChannel,
        wick_stats: WickStats,
        current_candle: Candle,
    ) -> str:
        """Generate human-readable rationale for the signal."""
        current_wick = current_candle.lower_wick_pct

        return (
            f"Uptrend detected ({trend.length} candles, strength: {trend.strength:.2f}). "
            f"Lower wick pattern: avg={wick_stats.avg_wick:.3f}%, "
            f"current={current_wick:.3f}%. "
            f"Entry on wick confirmation."
        )

    def validate_signal(self, signal: Signal, account: Account) -> bool:
        """
        Validate signal with relaxed risk/reward for wick strategy.

        The wick strategy targets small, high-probability gains. Standard
        1:1 risk/reward requirements don't apply since we expect higher
        win rates to compensate.
        """
        if signal is None:
            return False

        # Confidence check
        if signal.confidence < 0.0 or signal.confidence > 1.0:
            return False

        # Relaxed risk/reward - require at least 0.3:1 (30% of risk as potential reward)
        if signal.risk_reward_ratio < 0.3:
            return False

        # Buying power check
        position_value = self.position_size(signal, account) * signal.entry_price
        if position_value > account.buying_power:
            return False

        return True
