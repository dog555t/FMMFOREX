from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditConfig:
    """Configuration for audit logging."""
    log_path: str = "data/audit.jsonl"
    enabled: bool = True


@dataclass
class AuditEntry:
    """Single audit log entry for FTC compliance tracking."""
    timestamp: str
    event_type: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """
    Audit logger for FTC compliance and trade decision tracking.
    Logs all trading decisions, risk controls, and model predictions.
    """
    
    def __init__(self, config: AuditConfig) -> None:
        self.config = config
        self.log_path = Path(config.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("AuditLogger initialized with path: %s", self.log_path)
    
    def log_trade(
        self,
        direction: str,
        regime: str,
        price: float,
        position_size: float,
        stop_distance: float,
        take_profit_distance: float,
        fakeout_probability: float,
        **kwargs: Any,
    ) -> None:
        """Log a trade entry decision."""
        if not self.config.enabled:
            return
        
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="trade_entry",
            details={
                "direction": direction,
                "regime": regime,
                "price": price,
                "position_size": position_size,
                "stop_distance": stop_distance,
                "take_profit_distance": take_profit_distance,
                "fakeout_probability": fakeout_probability,
                **kwargs,
            },
        )
        self._write(entry)
    
    def log_trade_exit(
        self,
        pnl: float,
        exit_reason: str,
        exit_price: float,
        **kwargs: Any,
    ) -> None:
        """Log a trade exit."""
        if not self.config.enabled:
            return
        
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="trade_exit",
            details={
                "pnl": pnl,
                "exit_reason": exit_reason,
                "exit_price": exit_price,
                **kwargs,
            },
        )
        self._write(entry)
    
    def log_risk_control(
        self,
        control_type: str,
        triggered: bool,
        details: Dict[str, Any],
    ) -> None:
        """Log risk control decision (kill switch, position limits, etc.)."""
        if not self.config.enabled:
            return
        
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="risk_control",
            details={
                "control_type": control_type,
                "triggered": triggered,
                **details,
            },
        )
        self._write(entry)
    
    def log_model_prediction(
        self,
        model_type: str,
        prediction: Any,
        confidence: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Log model prediction for audit trail."""
        if not self.config.enabled:
            return
        
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="model_prediction",
            details={
                "model_type": model_type,
                "prediction": prediction,
                "confidence": confidence,
                **kwargs,
            },
        )
        self._write(entry)
    
    def log_config_change(
        self,
        config_section: str,
        old_value: Any,
        new_value: Any,
        changed_by: str = "system",
    ) -> None:
        """Log configuration changes."""
        if not self.config.enabled:
            return
        
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type="config_change",
            details={
                "config_section": config_section,
                "old_value": old_value,
                "new_value": new_value,
                "changed_by": changed_by,
            },
        )
        self._write(entry)
    
    def _write(self, entry: AuditEntry) -> None:
        """Write audit entry to JSONL file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error("Failed to write audit log: %s", e)
    
    def read_logs(
        self,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEntry]:
        """Read audit logs with optional filtering."""
        if not self.log_path.exists():
            return []
        
        entries: List[AuditEntry] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    entry = AuditEntry(**data)
                    if event_type is None or entry.event_type == event_type:
                        entries.append(entry)
                    if limit and len(entries) >= limit:
                        break
        except Exception as e:
            logger.error("Failed to read audit log: %s", e)
        
        return entries
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate compliance report from audit logs."""
        logs = self.read_logs()
        
        report = {
            "total_entries": len(logs),
            "by_type": {},
            "trades": {
                "total_entries": 0,
                "total_exits": 0,
            },
            "risk_controls": {
                "total_checks": 0,
                "total_triggered": 0,
            },
            "model_predictions": {
                "total": 0,
            },
        }
        
        for entry in logs:
            event_type = entry.event_type
            report["by_type"][event_type] = report["by_type"].get(event_type, 0) + 1
            
            if event_type == "trade_entry":
                report["trades"]["total_entries"] += 1
            elif event_type == "trade_exit":
                report["trades"]["total_exits"] += 1
            elif event_type == "risk_control":
                report["risk_controls"]["total_checks"] += 1
                if entry.details.get("triggered"):
                    report["risk_controls"]["total_triggered"] += 1
            elif event_type == "model_prediction":
                report["model_predictions"]["total"] += 1
        
        return report


__all__ = ["AuditLogger", "AuditConfig", "AuditEntry"]
