from src.risk.controls import RiskConfig, RiskEngine


def test_kill_switch_drawdown_triggers():
    cfg = RiskConfig(
        daily_drawdown_limit=0.05,
        rolling_drawdown_limit=0.1,
        spread_limit=0.5,
        volatility_percentile=2.0,
        max_leverage=5,
        max_positions=2,
    )
    engine = RiskEngine(cfg)
    engine.register_trade(100)
    engine.register_trade(90)
    assert engine.kill_switch(spread=0.1, realized_vol=0.5)


def test_position_risk_in_trend_higher():
    cfg = RiskConfig(
        daily_drawdown_limit=0.05,
        rolling_drawdown_limit=0.1,
        spread_limit=0.5,
        volatility_percentile=2.0,
        max_leverage=5,
        max_positions=2,
        risky_mode_risk=0.03,
        base_risk=0.01,
    )
    engine = RiskEngine(cfg)
    base = engine.position_risk(10000, "calm")
    trend = engine.position_risk(10000, "trend")
    assert trend > base
