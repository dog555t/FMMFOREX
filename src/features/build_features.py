from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    breakout_lookback: int = 20
    atr_period: int = 14


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift(1))
    low_close = np.abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    logger.debug("Computed ATR with period %s", period)
    return atr


def build_regime_features(df: pd.DataFrame, risk_proxy: Optional[pd.Series] = None) -> pd.DataFrame:
    returns = np.log(df["close"]).diff().fillna(0)
    vol = returns.rolling(20, min_periods=5).std().fillna(0)
    atr = compute_atr(df, period=14)
    rolling_max = df["close"].cummax()
    drawdown = (df["close"] - rolling_max) / rolling_max
    features = pd.DataFrame({"ret": returns, "vol": vol, "atr": atr, "drawdown": drawdown})
    if risk_proxy is not None and len(risk_proxy) == len(df):
        features["proxy_corr"] = returns.rolling(30, min_periods=5).corr(risk_proxy)
    else:
        features["proxy_corr"] = 0.0
    logger.debug("Built regime feature matrix with shape %s", features.shape)
    return features.fillna(0)


def build_breakout_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    atr = compute_atr(df, period=config.atr_period)
    highest_high = df["high"].rolling(config.breakout_lookback, min_periods=5).max()
    lowest_low = df["low"].rolling(config.breakout_lookback, min_periods=5).min()
    breakout_up = (df["close"] > highest_high.shift(1)) & (df["close"] > df["close"].rolling(3).mean())
    breakout_down = (df["close"] < lowest_low.shift(1)) & (df["close"] < df["close"].rolling(3).mean())
    body = np.abs(df["close"] - df["open"])
    wick = (df["high"] - df["low"]) - body
    vwap_proxy = df["close"].rolling(10, min_periods=1).mean()
    features = pd.DataFrame(
        {
            "atr": atr,
            "highest_high": highest_high,
            "lowest_low": lowest_low,
            "breakout_up": breakout_up.astype(int),
            "breakout_down": breakout_down.astype(int),
            "wick_body_ratio": (wick / body.replace(0, np.nan)).fillna(0),
            "momentum": df["close"].pct_change().rolling(5).mean().fillna(0),
            "vol_expansion": df["close"].pct_change().rolling(5).std().fillna(0),
            "dist_to_vwap": (df["close"] - vwap_proxy) / atr.replace(0, np.nan),
        }
    )
    features["dist_to_vwap"] = features["dist_to_vwap"].fillna(0)
    features["time_of_day"] = df["time"].dt.hour + df["time"].dt.minute / 60.0
    logger.debug("Built breakout features with shape %s", features.shape)
    return features


__all__ = ["FeatureConfig", "compute_atr", "build_regime_features", "build_breakout_features"]
