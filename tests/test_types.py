"""Tests for core types."""

import pytest
from datetime import datetime

from wicktrade.core.types import Candle, WickStats, TrendChannel


class TestCandle:
    """Tests for Candle dataclass."""

    def test_bullish_candle(self):
        """Test bullish candle properties."""
        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000,
        )

        assert candle.is_bullish
        assert not candle.is_bearish
        assert candle.body_top == 104.0
        assert candle.body_bottom == 100.0
        assert candle.body_size == 4.0
        assert candle.upper_wick == 1.0  # 105 - 104
        assert candle.lower_wick == 1.0  # 100 - 99

    def test_bearish_candle(self):
        """Test bearish candle properties."""
        candle = Candle(
            timestamp=datetime.now(),
            open=104.0,
            high=105.0,
            low=99.0,
            close=100.0,
            volume=1000,
        )

        assert candle.is_bearish
        assert not candle.is_bullish
        assert candle.body_top == 104.0
        assert candle.body_bottom == 100.0

    def test_wick_percentages(self):
        """Test wick percentage calculations."""
        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=102.0,
            low=98.0,
            close=101.0,
            volume=1000,
        )

        # Upper wick: 102 - 101 = 1, as % of 101 = 0.99%
        assert pytest.approx(candle.upper_wick_pct, rel=0.01) == 0.99

        # Lower wick: 100 - 98 = 2, as % of 101 = 1.98%
        assert pytest.approx(candle.lower_wick_pct, rel=0.01) == 1.98

    def test_doji_candle(self):
        """Test doji (open == close) candle."""
        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000,
        )

        assert candle.body_size == 0.0
        assert candle.body_top == 100.0
        assert candle.body_bottom == 100.0


class TestWickStats:
    """Tests for WickStats dataclass."""

    def test_valid_stats(self):
        """Test valid wick stats."""
        stats = WickStats(
            min_wick=0.5,
            max_wick=2.0,
            avg_wick=1.0,
            std_wick=0.3,
            median_wick=0.9,
            count=10,
        )

        assert stats.is_valid(min_count=3)
        assert stats.is_valid(min_count=10)
        assert not stats.is_valid(min_count=11)

    def test_invalid_stats(self):
        """Test invalid wick stats."""
        stats = WickStats(
            min_wick=0.0,
            max_wick=0.0,
            avg_wick=0.0,
            std_wick=0.0,
            median_wick=0.0,
            count=2,
        )

        assert not stats.is_valid(min_count=3)


class TestTrendChannel:
    """Tests for TrendChannel dataclass."""

    def test_uptrend(self):
        """Test uptrend channel properties."""
        trend = TrendChannel(
            trend_type="UPTREND",
            start_idx=0,
            end_idx=9,
            swing_lows=[(1, 100.0), (5, 102.0)],
            swing_highs=[(3, 104.0), (7, 106.0)],
            strength=0.8,
        )

        assert trend.is_uptrend
        assert not trend.is_downtrend
        assert trend.length == 10
        assert trend.is_valid(min_length=5)
        assert not trend.is_valid(min_length=15)

    def test_downtrend(self):
        """Test downtrend channel properties."""
        trend = TrendChannel(
            trend_type="DOWNTREND",
            start_idx=0,
            end_idx=9,
            swing_lows=[(3, 98.0), (7, 96.0)],
            swing_highs=[(1, 104.0), (5, 102.0)],
            strength=0.7,
        )

        assert trend.is_downtrend
        assert not trend.is_uptrend
