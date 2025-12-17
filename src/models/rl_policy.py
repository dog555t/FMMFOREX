from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PolicyOutput:
    position_size: float
    stop_distance: float
    take_profit_distance: float
    scale_in: bool
    scale_out: bool


class RLPolicy:
    """Placeholder PPO/SAC-like policy controlling leverage and scaling."""

    def __init__(self, max_leverage: float, mode: Literal["heuristic", "rl"] = "heuristic") -> None:
        self.max_leverage = max_leverage
        self.mode = mode
        # learned weights for rl mode stub
        self.weights = np.array([0.2, 0.4, -0.3, -0.1])

    def act(self, regime: str, direction: Literal["long", "short"], p_fakeout: float, volatility: float) -> PolicyOutput:
        if regime == "panic":
            return PolicyOutput(0.0, 0.0, 0.0, False, True)
        if self.mode == "heuristic":
            risk_scale = 0.5 if regime == "calm" else 1.0
            fakeout_penalty = max(0.2, 1 - p_fakeout)
            position = min(self.max_leverage, risk_scale * fakeout_penalty * (1 / max(volatility, 1e-5)))
            stop = max(0.0005, volatility * 1.5)
            tp = stop * 2.2
            return PolicyOutput(position_size=position, stop_distance=stop, take_profit_distance=tp, scale_in=False, scale_out=True)

        features = np.array([1.0 if regime == "trend" else 0.0, p_fakeout, volatility, 1.0 if direction == "long" else -1.0])
        logit = float(self.weights @ features)
        leverage = min(self.max_leverage, max(0.0, 1 + logit))
        stop = max(0.0004, volatility * (1.2 - 0.3 * p_fakeout))
        tp = stop * (2.5 - p_fakeout)
        logger.debug("RL mode leverage %.3f stop %.5f tp %.5f", leverage, stop, tp)
        return PolicyOutput(position_size=leverage, stop_distance=stop, take_profit_distance=tp, scale_in=True, scale_out=True)


__all__ = ["RLPolicy", "PolicyOutput"]
