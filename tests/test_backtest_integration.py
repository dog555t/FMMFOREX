"""Integration test for backtest engine with enhanced execution."""
import pandas as pd
import numpy as np

from src.backtest.engine import BacktestEngine
from src.backtest.execution import ExecutionConfig
from src.models.regime import RegimeModel, RegimeModelConfig
from src.models.fakeout_classifier import FakeoutClassifier, ClassifierConfig
from src.models.rl_policy import RLPolicy
from src.risk.controls import RiskEngine, RiskConfig


def sample_df():
    """Create sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="15min")
    
    # Create synthetic price data with some trends
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    
    data = {
        "time": dates,
        "open": close_prices + np.random.randn(100) * 0.05,
        "high": close_prices + abs(np.random.randn(100)) * 0.1,
        "low": close_prices - abs(np.random.randn(100)) * 0.1,
        "close": close_prices,
        "volume": np.random.randint(1000, 10000, 100),
    }
    return pd.DataFrame(data)


def test_backtest_with_execution_config():
    """Test that backtest engine works with execution configuration."""
    df = sample_df()
    
    # Build features matching backtest engine approach
    from src.features.build_features import FeatureConfig, build_breakout_features
    feat_cfg = FeatureConfig()
    breakout_feat = build_breakout_features(df, feat_cfg)
    
    # Build regime features matching the backtest engine's expectations
    regime_features = breakout_feat[["atr", "vol_expansion"]].copy()
    regime_features["vol"] = df["close"].pct_change().rolling(10).std().fillna(0)
    regime_features["drawdown"] = (df["close"] - df["close"].cummax()) / df["close"].cummax()
    
    # Train models
    regime_model = RegimeModel(RegimeModelConfig())
    regime_model.fit(regime_features.fillna(0))
    
    # Simple classifier
    labels = (breakout_feat["breakout_up"] | breakout_feat["breakout_down"]).astype(int)
    classifier = FakeoutClassifier(ClassifierConfig())
    classifier.fit(breakout_feat.fillna(0), labels)
    
    # Setup execution config
    exec_config = ExecutionConfig(
        spread_asia=0.0003,
        spread_london=0.0002,
        spread_ny=0.00025,
        spread_overlap=0.00015,
        partial_fill_probability=0.1,
        requote_probability=0.05,
        swap_long_rate=-0.00005,
        swap_short_rate=-0.00003,
        commission_rate=0.0001,
        gap_through_enabled=True,
    )
    
    # Run backtest
    risk_config = RiskConfig(
        daily_drawdown_limit=0.05,
        rolling_drawdown_limit=0.1,
        spread_limit=0.5,
        volatility_percentile=2.0,
        max_leverage=10,
        max_positions=3,
    )
    risk_engine = RiskEngine(risk_config)
    policy = RLPolicy(max_leverage=10, mode="heuristic")
    
    engine = BacktestEngine(
        regime_model=regime_model,
        classifier=classifier,
        policy=policy,
        risk_engine=risk_engine,
        slippage_pips=0.0001,
        spread_pips=0.0002,
        initial_balance=100000,
        execution_config=exec_config,
    )
    
    result = engine.run(df)
    
    # Verify result structure
    assert result.equity_curve is not None
    assert len(result.equity_curve) > 0
    assert result.trades is not None
    assert result.metrics is not None
    
    # Check that new fields are present in trades
    if not result.trades.empty:
        assert "filled_ratio" in result.trades.columns
        assert "commission" in result.trades.columns
        assert "swap" in result.trades.columns
        assert "exit_reason" in result.trades.columns


def test_backtest_without_execution_config():
    """Test that backtest engine works without execution config (backward compatible)."""
    df = sample_df()
    
    # Build features matching backtest engine approach
    from src.features.build_features import FeatureConfig, build_breakout_features
    feat_cfg = FeatureConfig()
    breakout_feat = build_breakout_features(df, feat_cfg)
    
    # Build regime features matching the backtest engine's expectations
    regime_features = breakout_feat[["atr", "vol_expansion"]].copy()
    regime_features["vol"] = df["close"].pct_change().rolling(10).std().fillna(0)
    regime_features["drawdown"] = (df["close"] - df["close"].cummax()) / df["close"].cummax()
    
    # Train models
    regime_model = RegimeModel(RegimeModelConfig())
    regime_model.fit(regime_features.fillna(0))
    
    labels = (breakout_feat["breakout_up"] | breakout_feat["breakout_down"]).astype(int)
    classifier = FakeoutClassifier(ClassifierConfig())
    classifier.fit(breakout_feat.fillna(0), labels)
    
    # Run backtest without execution config
    risk_config = RiskConfig(
        daily_drawdown_limit=0.05,
        rolling_drawdown_limit=0.1,
        spread_limit=0.5,
        volatility_percentile=2.0,
        max_leverage=10,
        max_positions=3,
    )
    risk_engine = RiskEngine(risk_config)
    policy = RLPolicy(max_leverage=10, mode="heuristic")
    
    engine = BacktestEngine(
        regime_model=regime_model,
        classifier=classifier,
        policy=policy,
        risk_engine=risk_engine,
        slippage_pips=0.0001,
        spread_pips=0.0002,
        initial_balance=100000,
    )
    
    result = engine.run(df)
    
    # Should still work with default execution config
    assert result.equity_curve is not None
    assert len(result.equity_curve) > 0
