from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    threshold: float = 0.15


def feature_drift(train_features: pd.DataFrame, live_features: pd.DataFrame, config: DriftConfig) -> Tuple[bool, float]:
    if train_features.empty or live_features.empty:
        return False, 0.0
    common_cols = [c for c in train_features.columns if c in live_features.columns]
    if not common_cols:
        return False, 0.0
    diffs = []
    for col in common_cols:
        mu_train, sigma_train = train_features[col].mean(), train_features[col].std() + 1e-6
        mu_live, sigma_live = live_features[col].mean(), live_features[col].std() + 1e-6
        diff = abs(mu_train - mu_live) / sigma_train + abs(sigma_train - sigma_live) / sigma_train
        diffs.append(diff)
    score = float(np.mean(diffs))
    drift = score > config.threshold
    if drift:
        logger.warning("Feature drift detected: score %.3f > threshold %.3f", score, config.threshold)
    return drift, score


__all__ = ["DriftConfig", "feature_drift"]
