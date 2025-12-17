from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    daily_drawdown_limit: float
    rolling_drawdown_limit: float
    spread_limit: float
    volatility_percentile: float
    max_leverage: float
    max_positions: int
    risky_mode_risk: float = 0.03
    base_risk: float = 0.01


class RiskEngine:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.equity_curve: List[float] = []
        self.open_positions = 0

    def can_open(self, position_size: float) -> bool:
        within_limits = self.open_positions < self.config.max_positions and position_size <= self.config.max_leverage
        logger.debug("Can open? %s (open %s max %s)", within_limits, self.open_positions, self.config.max_positions)
        return within_limits

    def position_risk(self, equity: float, regime: str) -> float:
        risk = self.config.base_risk if regime != "trend" else self.config.risky_mode_risk
        return equity * risk

    def register_trade(self, new_equity: float) -> None:
        self.equity_curve.append(new_equity)
        self.open_positions = min(self.open_positions + 1, self.config.max_positions)

    def close_trade(self) -> None:
        self.open_positions = max(0, self.open_positions - 1)

    def kill_switch(self, spread: float, realized_vol: float) -> bool:
        if not self.equity_curve:
            return False
        equity_series = np.array(self.equity_curve)
        daily_dd = (equity_series[-1] - equity_series.max()) / equity_series.max()
        rolling_dd = (equity_series[-1] - np.maximum.accumulate(equity_series)).min() if len(equity_series) > 1 else 0
        triggers = [
            daily_dd <= -self.config.daily_drawdown_limit,
            rolling_dd <= -self.config.rolling_drawdown_limit,
            spread > self.config.spread_limit,
            realized_vol > self.config.volatility_percentile,
        ]
        if any(triggers):
            logger.warning("Kill switch activated with triggers %s", triggers)
            return True
        return False


__all__ = ["RiskEngine", "RiskConfig"]
