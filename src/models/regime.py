from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


@dataclass
class RegimeModelConfig:
    n_states: int = 3
    random_state: int = 42


class RegimeModel:
    """HMM-like regime detector using Gaussian Mixture as a proxy."""

    def __init__(self, config: RegimeModelConfig) -> None:
        self.config = config
        self.model = GaussianMixture(n_components=config.n_states, random_state=config.random_state)
        self.state_order: List[str] = []

    def fit(self, features: pd.DataFrame) -> None:
        logger.info("Training regime detector on %s samples", len(features))
        self.model.fit(features)
        labels = self.model.predict(features)
        self.state_order = self._rank_states(features, labels)
        logger.info("State ordering inferred as %s", self.state_order)

    def predict(self, features: pd.DataFrame) -> List[str]:
        if not self.state_order:
            raise RuntimeError("Model not trained")
        labels = self.model.predict(features)
        return [self.state_order[label] for label in labels]

    def _rank_states(self, features: pd.DataFrame, labels: np.ndarray) -> List[str]:
        ranking = []
        for state in range(self.config.n_states):
            mask = labels == state
            vol = features.loc[mask, "vol"].mean()
            dd = features.loc[mask, "drawdown"].mean()
            ranking.append((state, vol, dd))
        # higher vol or drawdown -> panic, lowest vol -> calm, mid -> trend
        sorted_states = sorted(ranking, key=lambda x: (x[1], x[2]))
        if len(sorted_states) < 3:
            # fallback simple
            return ["calm" if i == 0 else "trend" for i in range(self.config.n_states)]
        calm_state = sorted_states[0][0]
        panic_state = sorted_states[-1][0]
        trend_state = [s[0] for s in sorted_states if s[0] not in (calm_state, panic_state)][0]
        mapping = {calm_state: "calm", trend_state: "trend", panic_state: "panic"}
        ordered = [mapping[i] for i in range(self.config.n_states)]
        return ordered


__all__ = ["RegimeModel", "RegimeModelConfig"]
