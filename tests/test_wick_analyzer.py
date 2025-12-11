"""Tests for wick analysis."""

import pytest
from datetime import datetime, timedelta

from wicktrade.core.types import Candle, TrendChannel, WickStats
from wicktrade.analysis.wick_analyzer import WickAnalyzer


def create_test_candles_with_wicks(n: int = 10) -> list:
    """Create candles with consistent lower wicks."""
    candles = []
    base_price = 100.0

    for i in range(n):
        # Create bullish candles with lower wicks
        open_price = base_price + i * 0.3
        close_price = open_price + 0.5
        high_price = close_price + 0.1
        # Lower wick of about 0.8% of close price
        low_price = open_price - 0.8

        candles.append(Candle(
            timestamp=datetime.now() + timedelta(minutes=i * 15),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000,
        ))

    return candles


def create_test_trend() -> TrendChannel:
    """Create a test uptrend channel."""
    return TrendChannel(
        trend_type="UPTREND",
        start_idx=0,
        end_idx=9,
        swing_lows=[(1, 99.5), (5, 100.5)],
        swing_highs=[(3, 101.0), (7, 102.0)],
        strength=0.8,
    )


class TestWickAnalyzer:
    """Tests for WickAnalyzer class."""

    def test_calculate_lower_wick(self):
        """Test lower wick calculation."""
        analyzer = WickAnalyzer()

        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=98.0,
            close=100.5,
            volume=1000,
        )

        wick_pct = analyzer.calculate_lower_wick_pct(candle)

        # Lower wick = 100 - 98 = 2
        # As % of close (100.5) = 1.99%
        assert pytest.approx(wick_pct, rel=0.01) == 1.99

    def test_calculate_upper_wick(self):
        """Test upper wick calculation."""
        analyzer = WickAnalyzer()

        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=103.0,
            low=99.0,
            close=100.5,
            volume=1000,
        )

        wick_pct = analyzer.calculate_upper_wick_pct(candle)

        # Upper wick = 103 - 100.5 = 2.5
        # As % of close (100.5) = 2.49%
        assert pytest.approx(wick_pct, rel=0.01) == 2.49

    def test_analyze_trend_wicks(self):
        """Test wick statistics calculation for a trend."""
        candles = create_test_candles_with_wicks(10)
        trend = create_test_trend()
        analyzer = WickAnalyzer()

        stats = analyzer.analyze_trend_wicks(candles, trend)

        assert stats.count == 10
        assert stats.min_wick > 0
        assert stats.max_wick >= stats.min_wick
        assert stats.min_wick <= stats.avg_wick <= stats.max_wick
        assert stats.is_valid()

    def test_wick_confirmation_true(self):
        """Test that good wick is confirmed."""
        analyzer = WickAnalyzer()

        # Create stats with avg_wick of 1.0%
        stats = WickStats(
            min_wick=0.5,
            max_wick=2.0,
            avg_wick=1.0,
            std_wick=0.3,
            median_wick=0.9,
            count=10,
        )

        # Candle with 0.8% lower wick (>= 50% of avg)
        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=100.5,
            low=99.2,
            close=100.3,
            volume=1000,
        )

        is_confirmed = analyzer.is_wick_confirmation(
            candle, stats, "UPTREND", confirmation_ratio=0.5
        )

        assert is_confirmed

    def test_wick_confirmation_false_too_small(self):
        """Test that small wick is not confirmed."""
        analyzer = WickAnalyzer()

        stats = WickStats(
            min_wick=0.5,
            max_wick=2.0,
            avg_wick=1.0,
            std_wick=0.3,
            median_wick=0.9,
            count=10,
        )

        # Candle with very small lower wick (< 50% of avg)
        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=100.5,
            low=99.8,  # Very small wick
            close=100.3,
            volume=1000,
        )

        is_confirmed = analyzer.is_wick_confirmation(
            candle, stats, "UPTREND", confirmation_ratio=0.5
        )

        assert not is_confirmed

    def test_get_entry_targets_uptrend(self):
        """Test entry target calculation for uptrend."""
        analyzer = WickAnalyzer()

        stats = WickStats(
            min_wick=0.5,
            max_wick=2.0,
            avg_wick=1.0,
            std_wick=0.3,
            median_wick=0.9,
            count=10,
        )

        entry_price = 100.0
        targets = analyzer.get_entry_targets(entry_price, stats, "UPTREND")

        # Initial target at avg_wick (1%) above entry
        assert pytest.approx(targets["initial_target"], rel=0.01) == 101.0

        # Max target at max_wick (2%) above entry
        assert pytest.approx(targets["max_target"], rel=0.01) == 102.0

        # Stop loss at 1.5 * max_wick (3%) below entry
        assert pytest.approx(targets["stop_loss"], rel=0.01) == 97.0

    def test_calculate_trailing_stop(self):
        """Test trailing stop calculation."""
        analyzer = WickAnalyzer()

        stats = WickStats(
            min_wick=0.5,
            max_wick=2.0,
            avg_wick=1.0,
            std_wick=0.3,
            median_wick=0.9,
            count=10,
        )

        current_price = 102.0
        entry_price = 100.0

        trailing_stop = analyzer.calculate_trailing_stop(
            current_price, entry_price, stats, "UPTREND", trailing_ratio=0.8
        )

        # Trail at 80% of avg_wick (0.8%) below current
        # 102 * (1 - 0.008) = 101.18
        assert pytest.approx(trailing_stop, rel=0.01) == 101.18

    def test_outlier_filtering(self):
        """Test that outliers are filtered from wick stats."""
        analyzer = WickAnalyzer(outlier_std=2.0)

        # Create candles with one extreme outlier
        candles = create_test_candles_with_wicks(10)

        # Add extreme outlier
        candles[5] = Candle(
            timestamp=candles[5].timestamp,
            open=candles[5].open,
            high=candles[5].high,
            low=candles[5].low - 10.0,  # Extreme lower wick
            close=candles[5].close,
            volume=1000,
        )

        trend = create_test_trend()
        stats = analyzer.analyze_trend_wicks(candles, trend, filter_outliers=True)

        # Stats should still be reasonable despite outlier
        assert stats.avg_wick < 5.0  # Should not be skewed by outlier
