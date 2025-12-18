"""Tests for enhanced execution simulation features."""
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.backtest.execution import (
    ExecutionConfig,
    ExecutionSimulator,
    TradingSession,
)


def test_trading_session_detection():
    """Test that trading sessions are correctly detected."""
    # Asia session (23:00 - 08:00 UTC)
    asia_time = datetime(2020, 1, 1, 2, 0, tzinfo=timezone.utc)
    assert TradingSession.get_session(asia_time) == "asia"
    
    # London session (08:00 - 16:00 UTC)
    london_time = datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc)
    assert TradingSession.get_session(london_time) == "london"
    
    # NY session (16:00 - 22:00 UTC, after London close)
    ny_time = datetime(2020, 1, 1, 18, 0, tzinfo=timezone.utc)
    assert TradingSession.get_session(ny_time) == "ny"
    
    # Overlap session (13:00 - 16:00 UTC)
    overlap_time = datetime(2020, 1, 1, 14, 0, tzinfo=timezone.utc)
    assert TradingSession.get_session(overlap_time) == "overlap"


def test_session_based_spreads():
    """Test that spreads vary by session."""
    config = ExecutionConfig()
    
    # Different spreads for different sessions
    asia_spread = TradingSession.get_spread("asia", config)
    london_spread = TradingSession.get_spread("london", config)
    ny_spread = TradingSession.get_spread("ny", config)
    overlap_spread = TradingSession.get_spread("overlap", config)
    
    # Verify spreads are different
    assert asia_spread > london_spread  # Asia has wider spreads
    assert overlap_spread < london_spread  # Overlap has tightest spreads
    assert ny_spread > overlap_spread  # NY wider than overlap


def test_market_order_execution():
    """Test basic market order execution."""
    config = ExecutionConfig(
        spread_london=0.0002,
        partial_fill_probability=0.0,  # Disable partials for this test
        requote_probability=0.0,  # Disable re-quotes
    )
    sim = ExecutionSimulator(config)
    
    timestamp = datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc)  # London session
    
    # Long order
    result = sim.execute_market_order(
        direction="long",
        entry_price=1.0,
        size=10000,
        timestamp=timestamp,
        volatility=0.001,
        next_high=1.01,
        next_low=0.99,
    )
    
    assert result.filled
    assert result.fill_ratio == 1.0
    assert result.fill_price == 1.0 + 0.0002  # Entry + spread
    assert not result.requoted


def test_partial_fills():
    """Test partial fill simulation."""
    np.random.seed(42)
    config = ExecutionConfig(
        partial_fill_probability=1.0,  # Always partial
        partial_fill_min_ratio=0.5,
        requote_probability=0.0,
    )
    sim = ExecutionSimulator(config)
    
    timestamp = datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc)
    
    result = sim.execute_market_order(
        direction="long",
        entry_price=1.0,
        size=10000,
        timestamp=timestamp,
        volatility=0.001,
        next_high=1.01,
        next_low=0.99,
    )
    
    assert result.filled
    assert 0.5 <= result.fill_ratio < 1.0  # Partial fill


def test_requotes():
    """Test re-quote simulation."""
    np.random.seed(42)
    config = ExecutionConfig(
        requote_probability=1.0,  # Always re-quote
    )
    sim = ExecutionSimulator(config)
    
    timestamp = datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc)
    
    result = sim.execute_market_order(
        direction="long",
        entry_price=1.0,
        size=10000,
        timestamp=timestamp,
        volatility=0.001,
        next_high=1.01,
        next_low=0.99,
    )
    
    assert not result.filled
    assert result.requoted


def test_stop_loss_gap_through():
    """Test stop-loss execution with gap-through behavior."""
    config = ExecutionConfig(
        gap_through_enabled=True,
        gap_slippage_multiplier=2.0,
    )
    sim = ExecutionSimulator(config)
    
    # Long position with stop at 0.99
    stop_price = 0.99
    current_price = 1.0
    bar_high = 1.01
    bar_low = 0.97  # Gap down past stop
    volatility = 0.005
    slippage = 0.0001
    
    hit, fill_price = sim.execute_stop_loss(
        direction="long",
        stop_price=stop_price,
        current_price=current_price,
        bar_high=bar_high,
        bar_low=bar_low,
        volatility=volatility,
        slippage_pips=slippage,
    )
    
    assert hit
    # Should fill worse than stop due to gap
    assert fill_price < stop_price
    assert fill_price >= bar_low  # Can't fill below bar low


def test_stop_loss_normal():
    """Test stop-loss execution without gap."""
    config = ExecutionConfig(
        gap_through_enabled=True,
    )
    sim = ExecutionSimulator(config)
    
    # Long position with stop at 0.99
    stop_price = 0.99
    current_price = 1.0
    bar_high = 1.01
    bar_low = 0.985  # Just touches stop, no big gap
    volatility = 0.002
    slippage = 0.0001
    
    hit, fill_price = sim.execute_stop_loss(
        direction="long",
        stop_price=stop_price,
        current_price=current_price,
        bar_high=bar_high,
        bar_low=bar_low,
        volatility=volatility,
        slippage_pips=slippage,
    )
    
    assert hit
    # Should fill at approximately stop minus slippage
    assert abs(fill_price - (stop_price - slippage)) < 0.001


def test_take_profit_execution():
    """Test take-profit execution."""
    config = ExecutionConfig()
    sim = ExecutionSimulator(config)
    
    # Long position with TP at 1.01
    tp_price = 1.01
    bar_high = 1.015
    bar_low = 0.99
    
    hit, fill_price = sim.execute_take_profit(
        direction="long",
        tp_price=tp_price,
        bar_high=bar_high,
        bar_low=bar_low,
    )
    
    assert hit
    assert fill_price == tp_price


def test_swap_calculation():
    """Test swap/financing calculation."""
    config = ExecutionConfig(
        swap_long_rate=-0.00005,
        swap_short_rate=-0.00003,
    )
    sim = ExecutionSimulator(config)
    
    # Long position swap
    swap_long = sim.calculate_swap(
        direction="long",
        position_size=10000,
        days_held=1.0,
    )
    assert swap_long == 10000 * -0.00005 * 1.0
    assert swap_long < 0  # Cost
    
    # Short position swap
    swap_short = sim.calculate_swap(
        direction="short",
        position_size=10000,
        days_held=1.0,
    )
    assert swap_short == 10000 * -0.00003 * 1.0
    assert swap_short < 0  # Cost


def test_commission_calculation():
    """Test commission calculation."""
    config = ExecutionConfig(
        commission_rate=0.00002,
        requote_probability=0.0,
        partial_fill_probability=0.0,
    )
    sim = ExecutionSimulator(config)
    
    timestamp = datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc)
    
    result = sim.execute_market_order(
        direction="long",
        entry_price=1.0,
        size=10000,
        timestamp=timestamp,
        volatility=0.001,
        next_high=1.01,
        next_low=0.99,
    )
    
    assert result.filled
    expected_commission = 10000 * 1.0 * 0.00002
    assert result.commission == expected_commission


def test_short_position_execution():
    """Test short position execution."""
    config = ExecutionConfig(
        spread_london=0.0002,
        partial_fill_probability=0.0,
        requote_probability=0.0,
    )
    sim = ExecutionSimulator(config)
    
    timestamp = datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc)
    
    # Short order
    result = sim.execute_market_order(
        direction="short",
        entry_price=1.0,
        size=10000,
        timestamp=timestamp,
        volatility=0.001,
        next_high=1.01,
        next_low=0.99,
    )
    
    assert result.filled
    assert result.fill_ratio == 1.0
    assert result.fill_price == 1.0 - 0.0002  # Entry - spread for short
