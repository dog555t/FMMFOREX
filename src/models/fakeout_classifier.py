from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
    learning_rate: float = 0.05
    n_estimators: int = 150
    max_depth: int = 3


class FakeoutClassifier:
    def __init__(self, config: ClassifierConfig) -> None:
        self.config = config
        if LGBMClassifier is not None:
            self.model: any = LGBMClassifier(
                learning_rate=config.learning_rate,
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
            )
            logger.info("Using LightGBM classifier")
        else:
            self.model = GradientBoostingClassifier(
                learning_rate=config.learning_rate, n_estimators=config.n_estimators, max_depth=config.max_depth
            )
            logger.info("Using sklearn GradientBoosting classifier")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.info("Training fakeout classifier on %s samples", len(X))
        self.model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


__all__ = ["FakeoutClassifier", "ClassifierConfig"]
