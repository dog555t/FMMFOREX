from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import yaml

from src.audit.logger import AuditConfig, AuditLogger
from src.backtest.engine import BacktestEngine
from src.backtest.execution import ExecutionConfig
from src.comparison.charts import create_comparison_charts
from src.comparison.runner import MultiCurrencyRunner
from src.data.cache import CandleCache
from src.data.oanda_client import OandaClient, OandaConfig
from src.features.build_features import FeatureConfig, build_breakout_features, build_regime_features
from src.models.fakeout_classifier import ClassifierConfig, FakeoutClassifier
from src.models.regime import RegimeModel, RegimeModelConfig
from src.models.rl_policy import RLPolicy
from src.risk.controls import RiskConfig, RiskEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Override with environment variables for Docker support
    if "TRADING_PAIR" in os.environ:
        cfg["trading"]["pair"] = os.environ["TRADING_PAIR"]
    if "TIMEFRAME" in os.environ:
        cfg["trading"]["timeframe"] = os.environ["TIMEFRAME"]
    if "WEB_PORT" in os.environ:
        cfg.setdefault("web", {})["port"] = int(os.environ["WEB_PORT"])
    
    return cfg


def download_data(args: argparse.Namespace) -> None:
    cfg = load_config()
    oanda_cfg = OandaConfig(**cfg["oanda"])
    client = OandaClient(oanda_cfg)
    cache = CandleCache(Path(cfg["data"]["cache_path"]))
    candles = client.fetch_candles(cfg["trading"]["pair"], cfg["trading"]["timeframe"], count=1000, synthetic=cfg["data"].get("synthetic", False))
    cache.save(cfg["trading"]["pair"], cfg["trading"]["timeframe"], candles)
    logger.info("Downloaded and cached data")


def train_regime(args: argparse.Namespace) -> None:
    cfg = load_config()
    cache = CandleCache(Path(cfg["data"]["cache_path"]))
    df = cache.load(cfg["trading"]["pair"], cfg["trading"]["timeframe"])
    if df is None:
        raise RuntimeError("No data cached. Run download-data first")
    feat = build_regime_features(df)
    model = RegimeModel(RegimeModelConfig())
    model.fit(feat)
    logger.info("Regime model trained on cached data")


def train_classifier(args: argparse.Namespace) -> None:
    cfg = load_config()
    cache = CandleCache(Path(cfg["data"]["cache_path"]))
    df = cache.load(cfg["trading"]["pair"], cfg["trading"]["timeframe"])
    if df is None:
        raise RuntimeError("No data cached. Run download-data first")
    feat_cfg = FeatureConfig(breakout_lookback=cfg["trading"]["breakout_lookback"], atr_period=cfg["trading"]["atr_period"])
    feat = build_breakout_features(df, feat_cfg)
    # Synthetic labels for prototype
    labels = (feat["breakout_up"] | feat["breakout_down"]).astype(int)
    clf = FakeoutClassifier(ClassifierConfig())
    clf.fit(feat.fillna(0), labels)
    logger.info("Fakeout classifier trained")


def run_backtest(args: argparse.Namespace) -> None:
    cfg = load_config()
    cache = CandleCache(Path(cfg["data"]["cache_path"]))
    df = cache.load(cfg["trading"]["pair"], cfg["trading"]["timeframe"])
    if df is None:
        raise RuntimeError("No data cached. Run download-data first")
    feat_cfg = FeatureConfig(breakout_lookback=cfg["trading"]["breakout_lookback"], atr_period=cfg["trading"]["atr_period"])
    breakout_feat = build_breakout_features(df, feat_cfg)
    labels = (breakout_feat["breakout_up"] | breakout_feat["breakout_down"]).astype(int)

    regime_model = RegimeModel(RegimeModelConfig())
    regime_model.fit(build_regime_features(df))

    classifier = FakeoutClassifier(ClassifierConfig())
    classifier.fit(breakout_feat.fillna(0), labels)

    # Setup audit logger
    audit_cfg = AuditConfig(**cfg.get("audit", {}))
    audit_logger = AuditLogger(audit_cfg)

    risk_engine = RiskEngine(RiskConfig(**cfg["risk"]), audit_logger=audit_logger)
    policy = RLPolicy(max_leverage=cfg["risk"]["max_leverage"], mode=args.policy)
    
    # Load execution config if available
    execution_config = None
    if "execution" in cfg["backtest"]:
        execution_config = ExecutionConfig(**cfg["backtest"]["execution"])
    
    engine = BacktestEngine(
        regime_model=regime_model,
        classifier=classifier,
        policy=policy,
        risk_engine=risk_engine,
        slippage_pips=cfg["backtest"]["slippage_pips"],
        spread_pips=cfg["backtest"]["spread_pips"],
        initial_balance=cfg["backtest"]["initial_balance"],
        audit_logger=audit_logger,
        execution_config=execution_config,
    )
    result = engine.run(df)
    logger.info("Backtest complete. Metrics: %s", result.metrics)
    logger.info("Regime distribution: %s", result.regime_distribution)
    logger.info("Audit logs saved to: %s", audit_cfg.log_path)


def paper_trade(args: argparse.Namespace) -> None:
    cfg = load_config()
    cache = CandleCache(Path(cfg["data"]["cache_path"]))
    df = cache.load(cfg["trading"]["pair"], cfg["trading"]["timeframe"])
    if df is None:
        raise RuntimeError("No data cached. Run download-data first")
    logger.info("Paper trading stub - integrate with TradeExecutor for live demo")


def compare_pairs(args: argparse.Namespace) -> None:
    cfg = load_config()
    pairs = args.pairs.split(",")
    timeframe = args.timeframe or cfg["trading"]["timeframe"]
    policy = args.policy
    
    logger.info("Running multi-currency comparison for pairs: %s", pairs)
    runner = MultiCurrencyRunner(cfg)
    results = runner.run_multiple(pairs, timeframe, policy)
    
    # Generate comparison table
    comparison_df = runner.compare_results(results)
    logger.info("\n=== Multi-Currency Comparison Results ===\n%s", comparison_df.to_string(index=False))
    
    # Save results
    comparison_df.to_csv("data/comparison_results.csv", index=False)
    logger.info("Comparison results saved to data/comparison_results.csv")
    
    # Generate charts if requested
    if args.charts:
        logger.info("Generating comparison charts...")
        charts = create_comparison_charts(results)
        logger.info("Generated %d comparison charts", len(charts))


def start_web_server(args: argparse.Namespace) -> None:
    cfg = load_config()
    web_cfg = cfg.get("web", {})
    
    from src.web.app import create_app
    
    app = create_app()
    host = web_cfg.get("host", "0.0.0.0")
    port = web_cfg.get("port", 5000)
    debug = web_cfg.get("debug", False)
    
    logger.info("Starting web server on http://%s:%s", host, port)
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    parser = argparse.ArgumentParser(description="FMMFOREX prototype CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("download-data")
    sub.add_parser("train-regime")
    sub.add_parser("train-classifier")
    backtest_parser = sub.add_parser("backtest")
    backtest_parser.add_argument("--policy", choices=["heuristic", "rl"], default="heuristic")
    sub.add_parser("paper-trade")
    sub.add_parser("web", help="Start web interface")
    
    # Multi-currency comparison command
    compare_parser = sub.add_parser("compare-pairs", help="Compare multiple currency pairs")
    compare_parser.add_argument("--pairs", required=True, help="Comma-separated list of currency pairs (e.g., USD_JPY,EUR_USD,GBP_USD)")
    compare_parser.add_argument("--timeframe", help="Timeframe (default: from config)")
    compare_parser.add_argument("--policy", choices=["heuristic", "rl"], default="heuristic")
    compare_parser.add_argument("--charts", action="store_true", help="Generate comparison charts")

    args = parser.parse_args()
    if args.command == "download-data":
        download_data(args)
    elif args.command == "train-regime":
        train_regime(args)
    elif args.command == "train-classifier":
        train_classifier(args)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "paper-trade":
        paper_trade(args)
    elif args.command == "compare-pairs":
        compare_pairs(args)
    elif args.command == "web":
        start_web_server(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
