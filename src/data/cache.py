from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CandleCache:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, pair: str, timeframe: str, df: pd.DataFrame) -> None:
        logger.info("Caching %s candles for %s", len(df), pair)
        with sqlite3.connect(self.db_path) as conn:
            table = f"candles_{pair}_{timeframe}"
            df.to_sql(table, conn, if_exists="replace", index=False)

    def load(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        table = f"candles_{pair}_{timeframe}"
        with sqlite3.connect(self.db_path) as conn:
            try:
                df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["time"])
                logger.info("Loaded %s cached candles for %s", len(df), pair)
                return df
            except Exception:
                logger.warning("No cache found for %s %s", pair, timeframe)
                return None


__all__ = ["CandleCache"]
