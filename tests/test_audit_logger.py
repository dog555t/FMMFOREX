import tempfile
from pathlib import Path

from src.audit.logger import AuditConfig, AuditLogger


def test_audit_logger_trade_logging():
    """Test that trade entries are logged correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_audit.jsonl"
        cfg = AuditConfig(log_path=str(log_path), enabled=True)
        logger = AuditLogger(cfg)
        
        logger.log_trade(
            direction="long",
            regime="trend",
            price=100.0,
            position_size=1000.0,
            stop_distance=0.5,
            take_profit_distance=1.0,
            fakeout_probability=0.2,
        )
        
        logs = logger.read_logs()
        assert len(logs) == 1
        assert logs[0].event_type == "trade_entry"
        assert logs[0].details["direction"] == "long"
        assert logs[0].details["regime"] == "trend"


def test_audit_logger_risk_control_logging():
    """Test that risk control decisions are logged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_audit.jsonl"
        cfg = AuditConfig(log_path=str(log_path), enabled=True)
        logger = AuditLogger(cfg)
        
        logger.log_risk_control(
            control_type="kill_switch",
            triggered=True,
            details={"reason": "drawdown_limit"},
        )
        
        logs = logger.read_logs()
        assert len(logs) == 1
        assert logs[0].event_type == "risk_control"
        assert logs[0].details["control_type"] == "kill_switch"
        assert logs[0].details["triggered"] is True


def test_audit_logger_disabled():
    """Test that logging can be disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_audit.jsonl"
        cfg = AuditConfig(log_path=str(log_path), enabled=False)
        logger = AuditLogger(cfg)
        
        logger.log_trade(
            direction="long",
            regime="trend",
            price=100.0,
            position_size=1000.0,
            stop_distance=0.5,
            take_profit_distance=1.0,
            fakeout_probability=0.2,
        )
        
        logs = logger.read_logs()
        assert len(logs) == 0


def test_audit_report_generation():
    """Test that audit report is generated correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_audit.jsonl"
        cfg = AuditConfig(log_path=str(log_path), enabled=True)
        logger = AuditLogger(cfg)
        
        # Log various events
        logger.log_trade(
            direction="long",
            regime="trend",
            price=100.0,
            position_size=1000.0,
            stop_distance=0.5,
            take_profit_distance=1.0,
            fakeout_probability=0.2,
        )
        logger.log_trade_exit(pnl=50.0, exit_reason="take_profit", exit_price=101.0)
        logger.log_risk_control(
            control_type="kill_switch",
            triggered=False,
            details={},
        )
        logger.log_model_prediction(
            model_type="fakeout_classifier",
            prediction="fakeout",
            confidence=0.8,
        )
        
        report = logger.generate_report()
        assert report["total_entries"] == 4
        assert report["trades"]["total_entries"] == 1
        assert report["trades"]["total_exits"] == 1
        assert report["risk_controls"]["total_checks"] == 1
        assert report["risk_controls"]["total_triggered"] == 0
        assert report["model_predictions"]["total"] == 1


def test_audit_logger_filtering():
    """Test that logs can be filtered by event type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_audit.jsonl"
        cfg = AuditConfig(log_path=str(log_path), enabled=True)
        logger = AuditLogger(cfg)
        
        logger.log_trade(
            direction="long",
            regime="trend",
            price=100.0,
            position_size=1000.0,
            stop_distance=0.5,
            take_profit_distance=1.0,
            fakeout_probability=0.2,
        )
        logger.log_risk_control(
            control_type="kill_switch",
            triggered=False,
            details={},
        )
        
        trade_logs = logger.read_logs(event_type="trade_entry")
        assert len(trade_logs) == 1
        assert trade_logs[0].event_type == "trade_entry"
        
        risk_logs = logger.read_logs(event_type="risk_control")
        assert len(risk_logs) == 1
        assert risk_logs[0].event_type == "risk_control"
