"""
Enhanced execution simulation for backtesting.
Includes session-based spreads, partial fills, re-quotes, swaps, and gap-through behavior.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for execution simulation."""
    # Session-based spreads (in pips)
    spread_asia: float = 0.0003
    spread_london: float = 0.0002
    spread_ny: float = 0.00025
    spread_overlap: float = 0.00015  # London-NY overlap
    
    # Partial fill parameters
    partial_fill_probability: float = 0.15  # Base probability of partial fill
    partial_fill_min_ratio: float = 0.5  # Minimum fill ratio when partial
    
    # Re-quote parameters
    requote_probability: float = 0.05  # Base probability of re-quote
    requote_spike_multiplier: float = 3.0  # Multiply during high volatility
    
    # Swap/financing
    swap_long_rate: float = -0.00005  # Daily swap for long positions (negative = cost)
    swap_short_rate: float = -0.00003  # Daily swap for short positions
    commission_rate: float = 0.0  # Commission per unit traded (0 = no commission)
    
    # Gap-through behavior
    gap_through_enabled: bool = True
    gap_slippage_multiplier: float = 2.0  # Additional slippage during gaps
    gap_threshold_multiplier: float = 0.5  # Threshold for gap detection (as multiple of volatility)
    
    # Volatility normalization
    max_volatility_ratio: float = 0.01  # Maximum volatility/price ratio for normalization
    

class TradingSession:
    """Detect and manage trading sessions."""
    
    @staticmethod
    def get_session(timestamp: datetime) -> str:
        """
        Determine trading session based on UTC time.
        
        Sessions (approximate UTC times):
        - Asia: 23:00 - 08:00 UTC (Tokyo 08:00-17:00 JST)
        - London: 08:00 - 16:00 UTC
        - NY: 13:00 - 22:00 UTC
        - Overlap (London-NY): 13:00 - 16:00 UTC
        """
        hour = timestamp.hour
        
        # London-NY overlap (highest liquidity)
        if 13 <= hour < 16:
            return "overlap"
        # London session
        elif 8 <= hour < 16:
            return "london"
        # NY session (after London close)
        elif 16 <= hour < 22:
            return "ny"
        # Asia session
        else:
            return "asia"
    
    @staticmethod
    def get_spread(session: str, config: ExecutionConfig) -> float:
        """Get spread for a given trading session."""
        spreads = {
            "asia": config.spread_asia,
            "london": config.spread_london,
            "ny": config.spread_ny,
            "overlap": config.spread_overlap,
        }
        return spreads.get(session, config.spread_london)


@dataclass
class ExecutionResult:
    """Result of an execution attempt."""
    filled: bool
    fill_ratio: float  # Percentage of order filled (1.0 = full, 0.5 = half)
    fill_price: float
    requoted: bool
    commission: float
    spread_cost: float
    details: str


class ExecutionSimulator:
    """Simulates realistic order execution with market microstructure effects."""
    
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
    
    def execute_market_order(
        self,
        direction: str,
        entry_price: float,
        size: float,
        timestamp: datetime,
        volatility: float,
        next_high: float,
        next_low: float,
    ) -> ExecutionResult:
        """
        Simulate execution of a market order with partial fills and re-quotes.
        
        Args:
            direction: "long" or "short"
            entry_price: Current market price
            size: Position size
            timestamp: Order timestamp
            volatility: Current market volatility (ATR or similar)
            next_high: Next bar high price
            next_low: Next bar low price
            
        Returns:
            ExecutionResult with fill details
        """
        session = TradingSession.get_session(timestamp)
        spread = TradingSession.get_spread(session, self.config)
        
        # Check for re-quote based on volatility
        vol_normalized = min(volatility / entry_price, 0.1)  # Cap at 10%
        requote_prob = self.config.requote_probability * (1 + vol_normalized * self.config.requote_spike_multiplier)
        requoted = np.random.random() < requote_prob
        
        if requoted:
            return ExecutionResult(
                filled=False,
                fill_ratio=0.0,
                fill_price=entry_price,
                requoted=True,
                commission=0.0,
                spread_cost=0.0,
                details=f"Re-quoted in {session} session due to volatility",
            )
        
        # Check for partial fill
        partial_prob = self.config.partial_fill_probability * (1 + vol_normalized)
        is_partial = np.random.random() < partial_prob
        fill_ratio = 1.0
        
        if is_partial:
            fill_ratio = np.random.uniform(
                self.config.partial_fill_min_ratio, 1.0
            )
        
        # Apply spread
        if direction == "long":
            fill_price = entry_price + spread
        else:
            fill_price = entry_price - spread
        
        # Calculate commission
        commission = size * fill_ratio * self.config.commission_rate
        spread_cost = size * fill_ratio * spread
        
        details = f"Filled {fill_ratio*100:.1f}% in {session} session"
        if is_partial:
            details += " (partial)"
        
        return ExecutionResult(
            filled=True,
            fill_ratio=fill_ratio,
            fill_price=fill_price,
            requoted=False,
            commission=commission,
            spread_cost=spread_cost,
            details=details,
        )
    
    def execute_stop_loss(
        self,
        direction: str,
        stop_price: float,
        current_price: float,
        bar_high: float,
        bar_low: float,
        volatility: float,
        slippage_pips: float,
    ) -> Tuple[bool, float]:
        """
        Execute stop-loss with gap-through behavior.
        
        Returns:
            (hit, fill_price) tuple
        """
        hit = False
        fill_price = stop_price
        
        if direction == "long":
            # Long stop is below entry, triggered when price goes down
            hit = bar_low <= stop_price
            if hit and self.config.gap_through_enabled:
                # Check for gap
                gap_size = max(0, stop_price - bar_low)
                
                # Larger gaps in high volatility
                gap_threshold = volatility * self.config.gap_threshold_multiplier
                if gap_size > gap_threshold:
                    # Gap detected - fill at worse price
                    fill_price = stop_price - slippage_pips * self.config.gap_slippage_multiplier
                    fill_price = max(fill_price, bar_low)  # Can't fill below low
                else:
                    fill_price = stop_price - slippage_pips
        else:
            # Short stop is above entry, triggered when price goes up
            hit = bar_high >= stop_price
            if hit and self.config.gap_through_enabled:
                gap_size = max(0, bar_high - stop_price)
                
                gap_threshold = volatility * self.config.gap_threshold_multiplier
                if gap_size > gap_threshold:
                    # Gap detected - fill at worse price
                    fill_price = stop_price + slippage_pips * self.config.gap_slippage_multiplier
                    fill_price = min(fill_price, bar_high)  # Can't fill above high
                else:
                    fill_price = stop_price + slippage_pips
        
        return hit, fill_price
    
    def execute_take_profit(
        self,
        direction: str,
        tp_price: float,
        bar_high: float,
        bar_low: float,
    ) -> Tuple[bool, float]:
        """
        Execute take-profit order (limit order).
        Assumes favorable fill at limit price if touched.
        
        Returns:
            (hit, fill_price) tuple
        """
        hit = False
        fill_price = tp_price
        
        if direction == "long":
            # Long TP is above entry, triggered when price goes up
            hit = bar_high >= tp_price
        else:
            # Short TP is below entry, triggered when price goes down
            hit = bar_low <= tp_price
        
        return hit, fill_price
    
    def calculate_swap(
        self,
        direction: str,
        position_size: float,
        days_held: float,
    ) -> float:
        """
        Calculate swap/financing costs.
        
        Args:
            direction: "long" or "short"
            position_size: Size of position
            days_held: Number of days position was held
            
        Returns:
            Swap cost (negative = cost, positive = credit)
        """
        if direction == "long":
            swap_rate = self.config.swap_long_rate
        else:
            swap_rate = self.config.swap_short_rate
        
        return position_size * swap_rate * days_held


__all__ = ["ExecutionConfig", "ExecutionSimulator", "ExecutionResult", "TradingSession"]
