"""
Wick analysis for calculating wick statistics within trends.
Analyzes lower wicks in uptrends and upper wicks in downtrends.
"""

import logging
from typing import List, Optional

import numpy as np

from wicktrade.core.types import Candle, TrendChannel, WickStats

logger = logging.getLogger(__name__)


class WickAnalyzer:
    """
    Analyzes candlestick wicks within trend channels.

    In uptrends, focuses on lower wicks (buyers stepping in at support).
    In downtrends, focuses on upper wicks (sellers stepping in at resistance).
    """

    def __init__(
        self,
        min_samples: int = 3,
        outlier_std: float = 3.0,
    ):
        """
        Initialize wick analyzer.

        Args:
            min_samples: Minimum candles required for valid statistics
            outlier_std: Number of standard deviations to filter outliers
        """
        self.min_samples = min_samples
        self.outlier_std = outlier_std

    def calculate_lower_wick_pct(self, candle: Candle) -> float:
        """
        Calculate lower wick as percentage of close price.

        Lower wick = body_bottom - low
        """
        return candle.lower_wick_pct

    def calculate_upper_wick_pct(self, candle: Candle) -> float:
        """
        Calculate upper wick as percentage of close price.

        Upper wick = high - body_top
        """
        return candle.upper_wick_pct

    def _filter_outliers(self, values: List[float]) -> List[float]:
        """
        Remove outlier values using standard deviation threshold.
        """
        if len(values) < 3:
            return values

        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return values

        # Keep values within outlier_std standard deviations
        filtered = [v for v in values if abs(v - mean) <= self.outlier_std * std]

        # Ensure we don't filter out too many values
        if len(filtered) < self.min_samples and len(values) >= self.min_samples:
            return values

        return filtered if filtered else values

    def analyze_trend_wicks(
        self,
        candles: List[Candle],
        trend: TrendChannel,
        filter_outliers: bool = True,
    ) -> WickStats:
        """
        Analyze wick statistics for candles within a trend.

        Args:
            candles: Full list of candles
            trend: TrendChannel defining the analysis range
            filter_outliers: Whether to filter extreme wick values

        Returns:
            WickStats with wick statistics for the trend
        """
        # Get candles within trend range
        trend_candles = candles[trend.start_idx:trend.end_idx + 1]

        if len(trend_candles) < self.min_samples:
            logger.warning(
                f"Insufficient candles for wick analysis: "
                f"{len(trend_candles)} < {self.min_samples}"
            )
            return WickStats(
                min_wick=0.0,
                max_wick=0.0,
                avg_wick=0.0,
                std_wick=0.0,
                median_wick=0.0,
                count=len(trend_candles),
            )

        # Calculate appropriate wicks based on trend type
        if trend.is_uptrend:
            wicks = [self.calculate_lower_wick_pct(c) for c in trend_candles]
        else:
            wicks = [self.calculate_upper_wick_pct(c) for c in trend_candles]

        # Filter outliers if requested
        if filter_outliers:
            wicks = self._filter_outliers(wicks)

        # Calculate statistics
        arr = np.array(wicks)

        return WickStats(
            min_wick=float(np.min(arr)),
            max_wick=float(np.max(arr)),
            avg_wick=float(np.mean(arr)),
            std_wick=float(np.std(arr)),
            median_wick=float(np.median(arr)),
            count=len(wicks),
        )

    def analyze_candle_wick(
        self,
        candle: Candle,
        trend_type: str = "UPTREND",
    ) -> float:
        """
        Analyze the relevant wick for a single candle.

        Args:
            candle: Candle to analyze
            trend_type: "UPTREND" or "DOWNTREND"

        Returns:
            Wick percentage (lower for uptrend, upper for downtrend)
        """
        if trend_type == "UPTREND":
            return self.calculate_lower_wick_pct(candle)
        else:
            return self.calculate_upper_wick_pct(candle)

    def is_wick_confirmation(
        self,
        candle: Candle,
        wick_stats: WickStats,
        trend_type: str = "UPTREND",
        confirmation_ratio: float = 0.5,
    ) -> bool:
        """
        Check if a candle's wick confirms the trend pattern.

        A wick is considered confirming if it's meaningful (above ratio of avg)
        but not extreme (below max * 1.5).

        Args:
            candle: Candle to check
            wick_stats: WickStats from the trend
            trend_type: Type of trend
            confirmation_ratio: Minimum ratio of avg_wick required

        Returns:
            True if wick confirms the pattern
        """
        if not wick_stats.is_valid(self.min_samples):
            return False

        current_wick = self.analyze_candle_wick(candle, trend_type)

        # Wick should be at least confirmation_ratio of average
        min_wick = wick_stats.avg_wick * confirmation_ratio

        # Wick shouldn't be too extreme (panic selling/buying)
        max_wick = wick_stats.max_wick * 1.5

        return min_wick <= current_wick <= max_wick

    def get_entry_targets(
        self,
        entry_price: float,
        wick_stats: WickStats,
        trend_type: str = "UPTREND",
    ) -> dict:
        """
        Calculate entry targets based on wick statistics.

        Args:
            entry_price: Expected entry price
            wick_stats: WickStats from the trend
            trend_type: Type of trend

        Returns:
            Dict with initial_target, max_target, and stop_loss prices
        """
        if trend_type == "UPTREND":
            # Long position targets
            initial_target = entry_price * (1 + wick_stats.avg_wick / 100)
            max_target = entry_price * (1 + wick_stats.max_wick / 100)
            stop_loss = entry_price * (1 - wick_stats.max_wick * 1.5 / 100)
        else:
            # Short position targets
            initial_target = entry_price * (1 - wick_stats.avg_wick / 100)
            max_target = entry_price * (1 - wick_stats.max_wick / 100)
            stop_loss = entry_price * (1 + wick_stats.max_wick * 1.5 / 100)

        return {
            "initial_target": initial_target,
            "max_target": max_target,
            "stop_loss": stop_loss,
        }

    def calculate_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        wick_stats: WickStats,
        trend_type: str = "UPTREND",
        trailing_ratio: float = 0.8,
    ) -> float:
        """
        Calculate trailing stop price based on wick statistics.

        Args:
            current_price: Current market price
            entry_price: Position entry price
            wick_stats: WickStats from the trend
            trend_type: Type of trend
            trailing_ratio: Ratio of avg_wick for trailing stop

        Returns:
            Trailing stop price
        """
        trail_distance_pct = wick_stats.avg_wick * trailing_ratio

        if trend_type == "UPTREND":
            # Long position - stop below current price
            return current_price * (1 - trail_distance_pct / 100)
        else:
            # Short position - stop above current price
            return current_price * (1 + trail_distance_pct / 100)

    def summarize_wick_pattern(
        self,
        candles: List[Candle],
        trend: TrendChannel,
    ) -> str:
        """
        Generate a human-readable summary of the wick pattern.

        Args:
            candles: Full list of candles
            trend: TrendChannel to summarize

        Returns:
            String summary of the wick pattern
        """
        stats = self.analyze_trend_wicks(candles, trend)

        wick_type = "lower" if trend.is_uptrend else "upper"
        direction = "uptrend" if trend.is_uptrend else "downtrend"

        return (
            f"{direction.title()} detected ({trend.length} candles, "
            f"strength: {trend.strength:.2f})\n"
            f"{wick_type.title()} wick stats:\n"
            f"  - Average: {stats.avg_wick:.3f}%\n"
            f"  - Min: {stats.min_wick:.3f}%\n"
            f"  - Max: {stats.max_wick:.3f}%\n"
            f"  - Std Dev: {stats.std_wick:.3f}%\n"
            f"  - Samples: {stats.count}"
        )
