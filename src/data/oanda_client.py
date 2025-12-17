from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class OandaConfig:
    api_url: str
    access_token: str
    account_id: str
    practice: bool = True


class OandaClient:
    """Lightweight OANDA REST client focused on candle retrieval."""

    def __init__(self, config: OandaConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {config.access_token}"})

    def fetch_candles(
        self,
        pair: str,
        granularity: str,
        count: int = 500,
        price: str = "M",
        synthetic: bool = False,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        if synthetic:
            logger.info("Generating synthetic GBM candles for %s", pair)
            return self._generate_synthetic(pair=pair, granularity=granularity, count=count, seed=seed)

        params = {"granularity": granularity, "count": count, "price": price}
        endpoint = f"{self.config.api_url}/instruments/{pair}/candles"
        logger.info("Requesting candles from OANDA: %s", endpoint)
        resp = self.session.get(endpoint, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("candles", [])
        records = []
        for candle in data:
            mid = candle.get("mid", {})
            records.append(
                {
                    "time": pd.to_datetime(candle["time"]),
                    "open": float(mid.get("o")),
                    "high": float(mid.get("h")),
                    "low": float(mid.get("l")),
                    "close": float(mid.get("c")),
                    "volume": float(candle.get("volume", 0)),
                    "complete": candle.get("complete", True),
                }
            )
        df = pd.DataFrame.from_records(records).sort_values("time").reset_index(drop=True)
        return df

    def _generate_synthetic(self, pair: str, granularity: str, count: int, seed: Optional[int]) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dt = 1 / 24
        mu, sigma = 0.05, 0.1
        prices = [1.0]
        for _ in range(count - 1):
            drift = (mu - 0.5 * sigma ** 2) * dt
            shock = sigma * np.sqrt(dt) * rng.standard_normal()
            prices.append(prices[-1] * np.exp(drift + shock))
        prices_arr = np.array(prices)
        highs = prices_arr * (1 + rng.uniform(0, 0.001, size=count))
        lows = prices_arr * (1 - rng.uniform(0, 0.001, size=count))
        opens = prices_arr * (1 + rng.uniform(-0.0005, 0.0005, size=count))
        closes = prices_arr
        now = int(time.time())
        index = pd.date_range(pd.to_datetime(now, unit="s"), periods=count, freq="15min")
        df = pd.DataFrame(
            {
                "time": index,
                "open": opens,
                "high": np.maximum(highs, opens),
                "low": np.minimum(lows, opens),
                "close": closes,
                "volume": rng.uniform(100, 1000, size=count),
                "complete": True,
            }
        )
        logger.debug("Synthetic candles generated for %s with granularity %s", pair, granularity)
        return df


__all__ = ["OandaClient", "OandaConfig"]
