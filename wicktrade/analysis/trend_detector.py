"""
Trend detection using swing highs and lows.
Identifies uptrend and downtrend channels in price data.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from wicktrade.core.types import Candle, TrendChannel

logger = logging.getLogger(__name__)


class TrendDetector:
    """
    Detects trend channels using swing high/low analysis.

    Uses scipy's argrelextrema to find local maxima (swing highs) and
    local minima (swing lows), then identifies sequences that form
    valid uptrend or downtrend channels.
    """

    def __init__(
        self,
        lookback: int = 3,
        min_trend_length: int = 5,
        min_swings: int = 2,
    ):
        """
        Initialize trend detector.

        Args:
            lookback: Order parameter for argrelextrema (how many bars on each
                     side to compare for determining local extrema)
            min_trend_length: Minimum number of candles for a valid trend
            min_swings: Minimum number of swing points required
        """
        self.lookback = lookback
        self.min_trend_length = min_trend_length
        self.min_swings = min_swings

    def find_swing_highs(self, highs: np.ndarray) -> np.ndarray:
        """
        Find indices of swing highs (local maxima).

        Args:
            highs: Array of high prices

        Returns:
            Array of indices where swing highs occur
        """
        indices = argrelextrema(highs, np.greater_equal, order=self.lookback)[0]
        return indices

    def find_swing_lows(self, lows: np.ndarray) -> np.ndarray:
        """
        Find indices of swing lows (local minima).

        Args:
            lows: Array of low prices

        Returns:
            Array of indices where swing lows occur
        """
        indices = argrelextrema(lows, np.less_equal, order=self.lookback)[0]
        return indices

    def _is_higher_high(self, highs: List[Tuple[int, float]]) -> bool:
        """Check if swing highs are making higher highs."""
        if len(highs) < 2:
            return False
        return highs[-1][1] > highs[-2][1]

    def _is_higher_low(self, lows: List[Tuple[int, float]]) -> bool:
        """Check if swing lows are making higher lows."""
        if len(lows) < 2:
            return False
        return lows[-1][1] > lows[-2][1]

    def _is_lower_high(self, highs: List[Tuple[int, float]]) -> bool:
        """Check if swing highs are making lower highs."""
        if len(highs) < 2:
            return False
        return highs[-1][1] < highs[-2][1]

    def _is_lower_low(self, lows: List[Tuple[int, float]]) -> bool:
        """Check if swing lows are making lower lows."""
        if len(lows) < 2:
            return False
        return lows[-1][1] < lows[-2][1]

    def _calculate_trend_strength(
        self,
        swing_points: List[Tuple[int, float]],
        trend_type: str,
    ) -> float:
        """
        Calculate trend strength based on consistency of swing points.

        Returns value between 0 and 1, where 1 is a perfect trend.
        """
        if len(swing_points) < 2:
            return 0.0

        # Calculate how consistently points follow the trend
        consistent = 0
        total = len(swing_points) - 1

        for i in range(1, len(swing_points)):
            prev_price = swing_points[i - 1][1]
            curr_price = swing_points[i][1]

            if trend_type == "UPTREND":
                if curr_price > prev_price:
                    consistent += 1
            else:  # DOWNTREND
                if curr_price < prev_price:
                    consistent += 1

        return consistent / total if total > 0 else 0.0

    def detect_trends(
        self,
        candles: List[Candle],
        trend_type: str = "UPTREND",
    ) -> List[TrendChannel]:
        """
        Detect trend channels in candle data.

        Args:
            candles: List of Candle objects
            trend_type: "UPTREND" or "DOWNTREND"

        Returns:
            List of detected TrendChannel objects
        """
        if len(candles) < self.min_trend_length:
            return []

        # Extract price arrays
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])

        # Find swing points
        swing_high_indices = self.find_swing_highs(highs)
        swing_low_indices = self.find_swing_lows(lows)

        if len(swing_high_indices) < self.min_swings or len(swing_low_indices) < self.min_swings:
            return []

        # Convert to (index, price) tuples
        swing_highs = [(int(i), float(highs[i])) for i in swing_high_indices]
        swing_lows = [(int(i), float(lows[i])) for i in swing_low_indices]

        trends = []

        if trend_type == "UPTREND":
            trends = self._detect_uptrends(swing_highs, swing_lows, len(candles))
        else:
            trends = self._detect_downtrends(swing_highs, swing_lows, len(candles))

        return trends

    def _detect_uptrends(
        self,
        swing_highs: List[Tuple[int, float]],
        swing_lows: List[Tuple[int, float]],
        total_candles: int,
    ) -> List[TrendChannel]:
        """
        Detect uptrend channels (higher highs + higher lows).
        """
        trends = []

        # Start from each potential trend beginning
        for start_idx in range(len(swing_lows) - 1):
            # Build a sequence of higher lows
            trend_lows = [swing_lows[start_idx]]

            for i in range(start_idx + 1, len(swing_lows)):
                if swing_lows[i][1] > trend_lows[-1][1]:  # Higher low
                    trend_lows.append(swing_lows[i])
                else:
                    break  # Trend broken

            if len(trend_lows) < self.min_swings:
                continue

            # Find corresponding highs within this range
            start_candle_idx = trend_lows[0][0]
            end_candle_idx = trend_lows[-1][0]

            trend_highs = [
                sh for sh in swing_highs
                if start_candle_idx <= sh[0] <= end_candle_idx
            ]

            # Check if highs are also making higher highs
            valid_highs = []
            for i, sh in enumerate(trend_highs):
                if i == 0 or sh[1] > valid_highs[-1][1]:
                    valid_highs.append(sh)

            if len(valid_highs) < self.min_swings:
                continue

            # Calculate trend length
            trend_length = end_candle_idx - start_candle_idx + 1

            if trend_length < self.min_trend_length:
                continue

            # Calculate strength
            low_strength = self._calculate_trend_strength(trend_lows, "UPTREND")
            high_strength = self._calculate_trend_strength(valid_highs, "UPTREND")
            strength = (low_strength + high_strength) / 2

            trend = TrendChannel(
                trend_type="UPTREND",
                start_idx=start_candle_idx,
                end_idx=end_candle_idx,
                swing_lows=trend_lows,
                swing_highs=valid_highs,
                strength=strength,
            )

            trends.append(trend)

        # Remove overlapping trends, keeping strongest
        trends = self._remove_overlapping_trends(trends)

        return trends

    def _detect_downtrends(
        self,
        swing_highs: List[Tuple[int, float]],
        swing_lows: List[Tuple[int, float]],
        total_candles: int,
    ) -> List[TrendChannel]:
        """
        Detect downtrend channels (lower highs + lower lows).
        """
        trends = []

        # Start from each potential trend beginning
        for start_idx in range(len(swing_highs) - 1):
            # Build a sequence of lower highs
            trend_highs = [swing_highs[start_idx]]

            for i in range(start_idx + 1, len(swing_highs)):
                if swing_highs[i][1] < trend_highs[-1][1]:  # Lower high
                    trend_highs.append(swing_highs[i])
                else:
                    break  # Trend broken

            if len(trend_highs) < self.min_swings:
                continue

            # Find corresponding lows within this range
            start_candle_idx = trend_highs[0][0]
            end_candle_idx = trend_highs[-1][0]

            trend_lows = [
                sl for sl in swing_lows
                if start_candle_idx <= sl[0] <= end_candle_idx
            ]

            # Check if lows are also making lower lows
            valid_lows = []
            for i, sl in enumerate(trend_lows):
                if i == 0 or sl[1] < valid_lows[-1][1]:
                    valid_lows.append(sl)

            if len(valid_lows) < self.min_swings:
                continue

            # Calculate trend length
            trend_length = end_candle_idx - start_candle_idx + 1

            if trend_length < self.min_trend_length:
                continue

            # Calculate strength
            low_strength = self._calculate_trend_strength(valid_lows, "DOWNTREND")
            high_strength = self._calculate_trend_strength(trend_highs, "DOWNTREND")
            strength = (low_strength + high_strength) / 2

            trend = TrendChannel(
                trend_type="DOWNTREND",
                start_idx=start_candle_idx,
                end_idx=end_candle_idx,
                swing_lows=valid_lows,
                swing_highs=trend_highs,
                strength=strength,
            )

            trends.append(trend)

        # Remove overlapping trends, keeping strongest
        trends = self._remove_overlapping_trends(trends)

        return trends

    def _remove_overlapping_trends(
        self, trends: List[TrendChannel]
    ) -> List[TrendChannel]:
        """
        Remove overlapping trends, keeping the strongest ones.
        """
        if not trends:
            return []

        # Sort by strength descending
        sorted_trends = sorted(trends, key=lambda t: t.strength, reverse=True)

        result = []
        used_ranges = []

        for trend in sorted_trends:
            # Check if this trend overlaps with any already selected
            overlaps = False
            for start, end in used_ranges:
                # Check for overlap
                if not (trend.end_idx < start or trend.start_idx > end):
                    overlaps = True
                    break

            if not overlaps:
                result.append(trend)
                used_ranges.append((trend.start_idx, trend.end_idx))

        return result

    def get_active_trend(
        self,
        candles: List[Candle],
        current_idx: int,
        trend_type: str = "UPTREND",
    ) -> Optional[TrendChannel]:
        """
        Get the most recent active trend at a specific candle index.

        Args:
            candles: List of all candles
            current_idx: Current candle index
            trend_type: Type of trend to look for

        Returns:
            Most recent TrendChannel that includes current_idx, or None
        """
        # Only analyze candles up to current index
        analysis_candles = candles[:current_idx + 1]

        trends = self.detect_trends(analysis_candles, trend_type)

        # Find trend that includes current candle
        for trend in reversed(trends):  # Start from most recent
            if trend.start_idx <= current_idx <= trend.end_idx:
                return trend

        return None

    def is_trend_intact(
        self,
        trend: TrendChannel,
        candle: Candle,
        candle_idx: int,
    ) -> bool:
        """
        Check if a trend is still intact given a new candle.

        Args:
            trend: The trend to check
            candle: New candle to evaluate
            candle_idx: Index of the new candle

        Returns:
            True if trend appears intact, False if broken
        """
        if trend.is_uptrend:
            # In uptrend, check if price made a lower low
            last_swing_low = trend.swing_lows[-1][1]
            if candle.low < last_swing_low * 0.99:  # 1% buffer
                return False
        else:
            # In downtrend, check if price made a higher high
            last_swing_high = trend.swing_highs[-1][1]
            if candle.high > last_swing_high * 1.01:  # 1% buffer
                return False

        return True
