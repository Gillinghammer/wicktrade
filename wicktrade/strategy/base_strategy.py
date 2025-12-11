"""
Base strategy interface that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from wicktrade.core.types import Signal, Position, Account, Candle


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            config: Strategy configuration parameters
        """
        self.name = name
        self.config = config
        self.is_active = True

    @abstractmethod
    def should_enter(
        self,
        candles: List[Candle],
        current_idx: int,
        account: Account,
    ) -> Optional[Signal]:
        """
        Determine if we should enter a new position.

        Args:
            candles: List of historical candles
            current_idx: Index of current candle in the list
            account: Current account state

        Returns:
            Signal object if entry conditions are met, None otherwise
        """
        pass

    @abstractmethod
    def should_exit(
        self,
        position: Position,
        candles: List[Candle],
        current_idx: int,
        account: Account,
    ) -> Optional[Dict[str, Any]]:
        """
        Determine if we should exit an existing position.

        Args:
            position: Current position
            candles: List of historical candles
            current_idx: Index of current candle
            account: Current account state

        Returns:
            Dict with exit info if should exit, None otherwise
            Expected keys: exit_type, exit_pct (partial exit %), reason
        """
        pass

    @abstractmethod
    def position_size(
        self,
        signal: Signal,
        account: Account,
    ) -> float:
        """
        Calculate position size for a signal.

        Args:
            signal: Trading signal
            account: Current account state

        Returns:
            Position size (number of shares)
        """
        pass

    def get_current_candle(
        self,
        candles: List[Candle],
        current_idx: int,
    ) -> Candle:
        """Get the current candle."""
        return candles[current_idx]

    def get_lookback_candles(
        self,
        candles: List[Candle],
        current_idx: int,
        lookback: int,
    ) -> List[Candle]:
        """Get candles for lookback period ending at current index."""
        start = max(0, current_idx - lookback + 1)
        return candles[start:current_idx + 1]

    def validate_signal(self, signal: Signal, account: Account) -> bool:
        """
        Validate that a signal meets basic requirements.

        Args:
            signal: Signal to validate
            account: Current account state

        Returns:
            True if signal is valid, False otherwise
        """
        if signal is None:
            return False

        # Confidence check
        if signal.confidence < 0.0 or signal.confidence > 1.0:
            return False

        # Risk/reward check
        if signal.risk_reward_ratio < 1.0:
            return False

        # Buying power check
        position_value = self.position_size(signal, account) * signal.entry_price
        if position_value > account.buying_power:
            return False

        return True

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update strategy configuration."""
        self.config.update(new_config)

    def activate(self) -> None:
        """Activate the strategy."""
        self.is_active = True

    def deactivate(self) -> None:
        """Deactivate the strategy."""
        self.is_active = False

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, active={self.is_active})"
