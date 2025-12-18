from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.audit.logger import AuditLogger
from src.backtest.execution import ExecutionConfig, ExecutionSimulator
from src.features.build_features import FeatureConfig, build_breakout_features, compute_atr
from src.models.fakeout_classifier import FakeoutClassifier
from src.models.regime import RegimeModel
from src.models.rl_policy import RLPolicy, PolicyOutput
from src.risk.controls import RiskEngine

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    regime_distribution: Dict[str, float]


class BacktestEngine:
    def __init__(
        self,
        regime_model: RegimeModel,
        classifier: FakeoutClassifier,
        policy: RLPolicy,
        risk_engine: RiskEngine,
        slippage_pips: float,
        spread_pips: float,
        initial_balance: float,
        audit_logger: Optional[AuditLogger] = None,
        execution_config: Optional[ExecutionConfig] = None,
    ) -> None:
        self.regime_model = regime_model
        self.classifier = classifier
        self.policy = policy
        self.risk_engine = risk_engine
        self.slippage_pips = slippage_pips
        self.spread_pips = spread_pips
        self.balance = initial_balance
        self.audit_logger = audit_logger
        self.execution_config = execution_config or ExecutionConfig()
        self.execution_sim = ExecutionSimulator(self.execution_config)

    def run(self, df: pd.DataFrame) -> BacktestResult:
        logger.info("Starting backtest on %s bars", len(df))
        feature_cfg = FeatureConfig()
        breakout_features = build_breakout_features(df, feature_cfg)
        regime_features = breakout_features[["atr", "vol_expansion"]].copy()
        regime_features["vol"] = df["close"].pct_change().rolling(10).std().fillna(0)
        regime_features["drawdown"] = (df["close"] - df["close"].cummax()) / df["close"].cummax()
        regimes = self.regime_model.predict(regime_features.fillna(0))
        fakeout_prob = self.classifier.predict_proba(breakout_features.fillna(0))

        equity = [self.balance]
        trades: List[Dict[str, float]] = []
        atr = compute_atr(df, period=feature_cfg.atr_period)

        for i in range(1, len(df)):
            regime = regimes[i]
            if regime == "panic":
                equity.append(equity[-1])
                continue
            direction: str | None = None
            if breakout_features.loc[i, "breakout_up"]:
                direction = "long"
            elif breakout_features.loc[i, "breakout_down"]:
                direction = "short"
            if direction is None:
                equity.append(equity[-1])
                continue
            p_fakeout = float(fakeout_prob[i])
            atr_val = float(atr.iloc[i])
            policy_out: PolicyOutput = self.policy.act(regime, direction, p_fakeout, volatility=atr_val)
            price = df.loc[i, "close"]
            stop = price - policy_out.stop_distance if direction == "long" else price + policy_out.stop_distance
            tp = price + policy_out.take_profit_distance if direction == "long" else price - policy_out.take_profit_distance

            # Log model prediction
            if self.audit_logger:
                self.audit_logger.log_model_prediction(
                    model_type="fakeout_classifier",
                    prediction="fakeout",
                    confidence=p_fakeout,
                    regime=regime,
                )

            risk_budget = self.risk_engine.position_risk(equity[-1], regime)
            position_size = min(policy_out.position_size, risk_budget / max(policy_out.stop_distance, 1e-6))
            if not self.risk_engine.can_open(position_size):
                equity.append(equity[-1])
                continue

            # simulate next bar outcome
            next_high, next_low, next_close = df.loc[i + 1, ["high", "low", "close"]] if i + 1 < len(df) else (
                df.loc[i, "high"],
                df.loc[i, "low"],
                df.loc[i, "close"],
            )
            
            # Get timestamp for session detection
            timestamp = pd.to_datetime(df.loc[i, "time"])
            
            # Execute market order with new simulator
            exec_result = self.execution_sim.execute_market_order(
                direction=direction,
                entry_price=price,
                size=position_size,
                timestamp=timestamp,
                volatility=atr_val,
                next_high=next_high,
                next_low=next_low,
            )
            
            # Handle re-quotes - skip trade
            if exec_result.requoted:
                equity.append(equity[-1])
                if self.audit_logger:
                    self.audit_logger.log_trade(
                        direction=direction,
                        regime=regime,
                        price=price,
                        position_size=0.0,
                        stop_distance=policy_out.stop_distance,
                        take_profit_distance=policy_out.take_profit_distance,
                        fakeout_probability=p_fakeout,
                        time=str(df.loc[i, "time"]),
                    )
                continue
            
            # Adjust position size for partial fills
            actual_position_size = position_size * exec_result.fill_ratio
            filled = exec_result.fill_price
            
            # Log trade entry
            if self.audit_logger:
                self.audit_logger.log_trade(
                    direction=direction,
                    regime=regime,
                    price=filled,
                    position_size=actual_position_size,
                    stop_distance=policy_out.stop_distance,
                    take_profit_distance=policy_out.take_profit_distance,
                    fakeout_probability=p_fakeout,
                    time=str(df.loc[i, "time"]),
                )

            # Simulate exit with gap-through behavior
            pnl = 0.0
            hit_tp, tp_fill = self.execution_sim.execute_take_profit(direction, tp, next_high, next_low)
            hit_sl, sl_fill = self.execution_sim.execute_stop_loss(
                direction, stop, price, next_high, next_low, atr_val, self.slippage_pips
            )
            
            exit_reason = "unknown"
            exit_price = next_close
            
            if hit_tp and not hit_sl:
                pnl = (tp_fill - filled) if direction == "long" else (filled - tp_fill)
                exit_reason = "take_profit"
                exit_price = tp_fill
            elif hit_sl and not hit_tp:
                pnl = (sl_fill - filled) if direction == "long" else (filled - sl_fill)
                exit_reason = "stop_loss"
                exit_price = sl_fill
            else:
                pnl = (next_close - filled) if direction == "long" else (filled - next_close)
                exit_reason = "market"
                exit_price = next_close
            
            # Calculate days held for swap (assuming intraday = 0, overnight = 1)
            # For simplicity, assume each bar is intraday unless specified
            days_held = 0.0
            if i + 1 < len(df):
                time_diff = pd.to_datetime(df.loc[i + 1, "time"]) - timestamp
                days_held = time_diff.total_seconds() / (24 * 3600)
            
            # Apply swap/financing
            swap_cost = self.execution_sim.calculate_swap(direction, actual_position_size, days_held)
            
            # Total costs
            total_costs = exec_result.commission + swap_cost
            
            trade_return = pnl / max(policy_out.stop_distance, 1e-6)
            trade_pnl = actual_position_size * pnl - total_costs
            new_balance = equity[-1] + trade_pnl
            
            # Log trade exit
            if self.audit_logger:
                self.audit_logger.log_trade_exit(
                    pnl=trade_pnl,
                    exit_reason=exit_reason,
                    exit_price=next_close,
                    time=str(df.loc[i + 1, "time"]) if i + 1 < len(df) else str(df.loc[i, "time"]),
                )
            
            self.risk_engine.register_trade(new_balance)
            equity.append(new_balance)
            trades.append(
                {
                    "time": df.loc[i, "time"],
                    "direction": direction,
                    "regime": regime,
                    "p_fakeout": p_fakeout,
                    "position_size": actual_position_size,
                    "filled_ratio": exec_result.fill_ratio,
                    "pnl": trade_pnl,
                    "r_multiple": trade_return,
                    "commission": exec_result.commission,
                    "swap": swap_cost,
                    "exit_reason": exit_reason,
                }
            )

        equity_curve = pd.Series(equity, index=df.index[: len(equity)])
        trades_df = pd.DataFrame(trades)
        metrics = self._metrics(equity_curve, trades_df)
        regime_dist = dict(trades_df["regime"].value_counts(normalize=True)) if not trades_df.empty else {}
        if metrics.get("plot"):
            self._plot(equity_curve)
        return BacktestResult(equity_curve=equity_curve, trades=trades_df, metrics=metrics, regime_distribution=regime_dist)

    def _metrics(self, equity: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
        returns = equity.pct_change().fillna(0)
        sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)) if returns.std() > 0 else 0.0
        dd = (equity / equity.cummax() - 1).min()
        win_rate = float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0
        avg_r = float(trades["r_multiple"].mean()) if not trades.empty else 0.0
        metrics = {
            "final_balance": float(equity.iloc[-1]),
            "max_drawdown": float(dd),
            "sharpe": float(sharpe),
            "win_rate": win_rate,
            "avg_r": avg_r,
            "plot": False,
        }
        logger.info("Backtest metrics %s", metrics)
        return metrics

    def _plot(self, equity: pd.Series) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        equity.plot(ax=ax, title="Equity Curve")
        ax.set_ylabel("Balance")
        fig.tight_layout()
        fig.savefig("backtest_equity.png")
        plt.close(fig)


__all__ = ["BacktestEngine", "BacktestResult"]
