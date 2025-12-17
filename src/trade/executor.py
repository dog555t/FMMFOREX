from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from src.data.oanda_client import OandaClient
from src.risk.controls import RiskEngine

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    pair: str
    direction: str
    units: float
    stop_loss: float
    take_profit: float


class TradeExecutor:
    def __init__(self, client: OandaClient, risk_engine: RiskEngine) -> None:
        self.client = client
        self.risk_engine = risk_engine

    def paper_trade(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Running paper trading over dataframe with %s rows", len(df))
        records: Dict[str, float] = {}
        # In a real system, streaming prices would be used.
        return pd.DataFrame([records])


__all__ = ["TradeExecutor", "OrderRequest"]
