"""
Fee models for backtesting.
Implements realistic fee structures for different brokers.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AlpacaFeeModel:
    """
    Alpaca fee structure for US equities.

    Alpaca offers commission-free trading but passes through
    regulatory fees:
    - SEC fee: $0.0000278 per dollar of sale proceeds (sells only)
    - FINRA TAF: $0.000166 per share (max $8.30 per trade)
    - ECN/Exchange fees: ~$0.003 per share (varies)
    """

    sec_fee_rate: float = 0.0000278      # Per dollar sold
    finra_taf_rate: float = 0.000166     # Per share
    finra_taf_max: float = 8.30          # Max TAF per trade
    ecn_fee_per_share: float = 0.003     # ECN/exchange fee per share
    commission: float = 0.0               # Commission per trade

    def calculate_fees(
        self,
        side: str,
        shares: float,
        price: float,
        notional: Optional[float] = None,
    ) -> dict:
        """
        Calculate all fees for a trade.

        Args:
            side: "buy" or "sell"
            shares: Number of shares
            price: Price per share
            notional: Total trade value (calculated if not provided)

        Returns:
            Dict with breakdown of fees and total
        """
        if notional is None:
            notional = shares * price

        fees = {
            "commission": self.commission,
            "sec_fee": 0.0,
            "finra_taf": 0.0,
            "ecn_fee": 0.0,
            "total": 0.0,
        }

        # Commission
        fees["commission"] = self.commission

        # FINRA TAF (applies to both buys and sells)
        taf = min(shares * self.finra_taf_rate, self.finra_taf_max)
        fees["finra_taf"] = taf

        # ECN/Exchange fees
        fees["ecn_fee"] = shares * self.ecn_fee_per_share

        # SEC fee (sells only)
        if side.lower() == "sell":
            fees["sec_fee"] = notional * self.sec_fee_rate

        # Total
        fees["total"] = (
            fees["commission"] +
            fees["sec_fee"] +
            fees["finra_taf"] +
            fees["ecn_fee"]
        )

        return fees

    def total_round_trip_fees(
        self,
        shares: float,
        entry_price: float,
        exit_price: float,
    ) -> float:
        """
        Calculate total fees for a round-trip trade (buy + sell).

        Args:
            shares: Number of shares
            entry_price: Entry price per share
            exit_price: Exit price per share

        Returns:
            Total fees for the round trip
        """
        buy_fees = self.calculate_fees("buy", shares, entry_price)
        sell_fees = self.calculate_fees("sell", shares, exit_price)

        return buy_fees["total"] + sell_fees["total"]


@dataclass
class ZeroFeeModel:
    """
    Zero fee model for simplified backtesting.
    Useful for initial strategy development without fee friction.
    """

    def calculate_fees(
        self,
        side: str,
        shares: float,
        price: float,
        notional: Optional[float] = None,
    ) -> dict:
        """Always returns zero fees."""
        return {
            "commission": 0.0,
            "sec_fee": 0.0,
            "finra_taf": 0.0,
            "ecn_fee": 0.0,
            "total": 0.0,
        }

    def total_round_trip_fees(
        self,
        shares: float,
        entry_price: float,
        exit_price: float,
    ) -> float:
        """Always returns zero."""
        return 0.0


@dataclass
class PercentageFeeModel:
    """
    Simple percentage-based fee model.
    Useful for modeling brokers with percentage commissions.
    """

    fee_pct: float = 0.001  # 0.1% default

    def calculate_fees(
        self,
        side: str,
        shares: float,
        price: float,
        notional: Optional[float] = None,
    ) -> dict:
        """Calculate percentage-based fees."""
        if notional is None:
            notional = shares * price

        fee = notional * self.fee_pct

        return {
            "commission": fee,
            "sec_fee": 0.0,
            "finra_taf": 0.0,
            "ecn_fee": 0.0,
            "total": fee,
        }

    def total_round_trip_fees(
        self,
        shares: float,
        entry_price: float,
        exit_price: float,
    ) -> float:
        """Calculate round-trip percentage fees."""
        entry_notional = shares * entry_price
        exit_notional = shares * exit_price

        return (entry_notional + exit_notional) * self.fee_pct
