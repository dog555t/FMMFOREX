from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from flask import Flask, jsonify, render_template, request

from src.audit.logger import AuditConfig, AuditLogger
from src.backtest.engine import BacktestEngine
from src.data.cache import CandleCache
from src.features.build_features import FeatureConfig, build_breakout_features, build_regime_features
from src.models.fakeout_classifier import ClassifierConfig, FakeoutClassifier
from src.models.regime import RegimeModel, RegimeModelConfig
from src.models.rl_policy import RLPolicy
from src.risk.controls import RiskConfig, RiskEngine

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_app(config_path: str = "config.yaml") -> Flask:
    app = Flask(__name__)
    app.config["CONFIG_PATH"] = config_path
    
    @app.route("/")
    def index() -> str:
        """Home page with system status."""
        cfg = load_config(app.config["CONFIG_PATH"])
        cache = CandleCache(Path(cfg["data"]["cache_path"]))
        df = cache.load(cfg["trading"]["pair"], cfg["trading"]["timeframe"])
        
        status = {
            "data_available": df is not None,
            "data_rows": len(df) if df is not None else 0,
            "trading_pair": cfg["trading"]["pair"],
            "timeframe": cfg["trading"]["timeframe"],
            "risk_limits": cfg["risk"],
        }
        
        return render_template("index.html", status=status)
    
    @app.route("/config")
    def config_view() -> str:
        """View current configuration."""
        cfg = load_config(app.config["CONFIG_PATH"])
        return render_template("config.html", config=cfg)
    
    @app.route("/backtest")
    def backtest_view() -> str:
        """Backtest results page."""
        return render_template("backtest.html")
    
    @app.route("/api/backtest", methods=["POST"])
    def run_backtest() -> Dict[str, Any]:
        """API endpoint to run backtest."""
        try:
            data = request.get_json() or {}
            policy = data.get("policy", "heuristic")
            
            cfg = load_config(app.config["CONFIG_PATH"])
            cache = CandleCache(Path(cfg["data"]["cache_path"]))
            df = cache.load(cfg["trading"]["pair"], cfg["trading"]["timeframe"])
            
            if df is None:
                return jsonify({"error": "No data available. Run download-data first"}), 400
            
            # Setup audit logger
            audit_cfg = AuditConfig(log_path=cfg.get("audit", {}).get("log_path", "data/audit.jsonl"))
            audit_logger = AuditLogger(audit_cfg)
            
            # Build features and models
            feat_cfg = FeatureConfig(
                breakout_lookback=cfg["trading"]["breakout_lookback"],
                atr_period=cfg["trading"]["atr_period"],
            )
            breakout_feat = build_breakout_features(df, feat_cfg)
            labels = (breakout_feat["breakout_up"] | breakout_feat["breakout_down"]).astype(int)
            
            regime_model = RegimeModel(RegimeModelConfig())
            regime_model.fit(build_regime_features(df))
            
            classifier = FakeoutClassifier(ClassifierConfig())
            classifier.fit(breakout_feat.fillna(0), labels)
            
            risk_engine = RiskEngine(RiskConfig(**cfg["risk"]), audit_logger=audit_logger)
            policy_obj = RLPolicy(max_leverage=cfg["risk"]["max_leverage"], mode=policy)
            
            engine = BacktestEngine(
                regime_model=regime_model,
                classifier=classifier,
                policy=policy_obj,
                risk_engine=risk_engine,
                slippage_pips=cfg["backtest"]["slippage_pips"],
                spread_pips=cfg["backtest"]["spread_pips"],
                initial_balance=cfg["backtest"]["initial_balance"],
                audit_logger=audit_logger,
            )
            
            result = engine.run(df)
            
            return jsonify({
                "success": True,
                "metrics": result.metrics,
                "regime_distribution": result.regime_distribution,
                "trades_count": len(result.trades),
            })
        except Exception as e:
            logger.error("Backtest failed: %s", e, exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    @app.route("/audit")
    def audit_view() -> str:
        """Audit logs page."""
        cfg = load_config(app.config["CONFIG_PATH"])
        audit_cfg = AuditConfig(log_path=cfg.get("audit", {}).get("log_path", "data/audit.jsonl"))
        audit_logger = AuditLogger(audit_cfg)
        
        # Get recent logs
        logs = audit_logger.read_logs(limit=100)
        logs.reverse()  # Most recent first
        
        # Generate report
        report = audit_logger.generate_report()
        
        return render_template("audit.html", logs=logs, report=report)
    
    @app.route("/api/audit/report")
    def audit_report_api() -> Dict[str, Any]:
        """API endpoint for audit report."""
        try:
            cfg = load_config(app.config["CONFIG_PATH"])
            audit_cfg = AuditConfig(log_path=cfg.get("audit", {}).get("log_path", "data/audit.jsonl"))
            audit_logger = AuditLogger(audit_cfg)
            report = audit_logger.generate_report()
            return jsonify(report)
        except Exception as e:
            logger.error("Failed to generate audit report: %s", e)
            return jsonify({"error": str(e)}), 500
    
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
