"""Multi-currency backtest runner for comparison."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.audit.logger import AuditConfig, AuditLogger
from src.backtest.engine import BacktestEngine, BacktestResult
from src.data.cache import CandleCache
from src.data.oanda_client import OandaClient, OandaConfig
from src.features.build_features import FeatureConfig, build_breakout_features, build_regime_features
from src.models.fakeout_classifier import ClassifierConfig, FakeoutClassifier
from src.models.regime import RegimeModel, RegimeModelConfig
from src.models.rl_policy import RLPolicy
from src.risk.controls import RiskConfig, RiskEngine

logger = logging.getLogger(__name__)


@dataclass
class CurrencyPairResult:
    """Result for a single currency pair backtest."""
    pair: str
    result: BacktestResult
    metrics: Dict[str, float]


class MultiCurrencyRunner:
    """Run backtests on multiple currency pairs and aggregate results."""
    
    def __init__(self, config: dict):
        self.config = config
        self.oanda_config = OandaConfig(**config["oanda"])
        self.cache = CandleCache(Path(config["data"]["cache_path"]))
        
    def run_pair(self, pair: str, timeframe: str, policy: str = "heuristic") -> CurrencyPairResult:
        """Run backtest for a single currency pair."""
        logger.info("Running backtest for %s on %s timeframe", pair, timeframe)
        
        # Load or fetch data
        df = self.cache.load(pair, timeframe)
        if df is None:
            logger.info("No cached data for %s, fetching...", pair)
            client = OandaClient(self.oanda_config)
            candles = client.fetch_candles(
                pair, 
                timeframe, 
                count=1000, 
                synthetic=self.config["data"].get("synthetic", False)
            )
            self.cache.save(pair, timeframe, candles)
            df = self.cache.load(pair, timeframe)
        
        if df is None:
            raise RuntimeError(f"Failed to load data for {pair}")
        
        # Build features and models
        feat_cfg = FeatureConfig(
            breakout_lookback=self.config["trading"]["breakout_lookback"],
            atr_period=self.config["trading"]["atr_period"],
        )
        breakout_feat = build_breakout_features(df, feat_cfg)
        labels = (breakout_feat["breakout_up"] | breakout_feat["breakout_down"]).astype(int)
        
        regime_model = RegimeModel(RegimeModelConfig())
        regime_model.fit(build_regime_features(df))
        
        classifier = FakeoutClassifier(ClassifierConfig())
        classifier.fit(breakout_feat.fillna(0), labels)
        
        # Setup audit logger
        audit_cfg = AuditConfig(
            log_path=f"data/audit_{pair.replace('/', '_')}_{timeframe}.jsonl",
            enabled=self.config.get("audit", {}).get("enabled", True)
        )
        audit_logger = AuditLogger(audit_cfg)
        
        # Run backtest
        risk_engine = RiskEngine(RiskConfig(**self.config["risk"]), audit_logger=audit_logger)
        policy_obj = RLPolicy(max_leverage=self.config["risk"]["max_leverage"], mode=policy)
        
        engine = BacktestEngine(
            regime_model=regime_model,
            classifier=classifier,
            policy=policy_obj,
            risk_engine=risk_engine,
            slippage_pips=self.config["backtest"]["slippage_pips"],
            spread_pips=self.config["backtest"]["spread_pips"],
            initial_balance=self.config["backtest"]["initial_balance"],
            audit_logger=audit_logger,
        )
        
        result = engine.run(df)
        logger.info("Backtest complete for %s: %s", pair, result.metrics)
        
        return CurrencyPairResult(
            pair=pair,
            result=result,
            metrics=result.metrics,
        )
    
    def run_multiple(
        self, 
        pairs: List[str], 
        timeframe: str = "M15", 
        policy: str = "heuristic"
    ) -> List[CurrencyPairResult]:
        """Run backtests for multiple currency pairs."""
        results = []
        for pair in pairs:
            try:
                result = self.run_pair(pair, timeframe, policy)
                results.append(result)
            except Exception as e:
                logger.error("Failed to run backtest for %s: %s", pair, e, exc_info=True)
        
        return results
    
    def compare_results(self, results: List[CurrencyPairResult]) -> pd.DataFrame:
        """Compare results across currency pairs."""
        if not results:
            return pd.DataFrame()
        
        comparison_data = []
        for result in results:
            comparison_data.append({
                "pair": result.pair,
                "final_balance": result.metrics.get("final_balance", 0),
                "max_drawdown": result.metrics.get("max_drawdown", 0),
                "sharpe": result.metrics.get("sharpe", 0),
                "win_rate": result.metrics.get("win_rate", 0),
                "avg_r": result.metrics.get("avg_r", 0),
                "total_trades": len(result.result.trades) if result.result.trades is not None else 0,
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values("final_balance", ascending=False)
        return df


__all__ = ["MultiCurrencyRunner", "CurrencyPairResult"]
