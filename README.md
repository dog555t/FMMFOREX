# FMMFOREX

Prototype of a high-risk / high-reward FX trading system with strict risk controls. The project focuses on OANDA integration (practice by default), regime detection, breakout entries, and defensive risk management.

## WARNING
- This code is for research only. Live trading of leveraged FX is extremely risky.
- Always use OANDA practice accounts first and validate results with independent backtests.

## Features
- **Data**: OANDA REST downloader with SQLite caching and synthetic GBM data mode for offline tests.
- **Regime detection**: Hidden-state proxy via GaussianMixture to label calm / trend / panic regimes using volatility, ATR, drawdown, and proxy correlation.
- **Breakout signal**: High/low breakout with ATR filter and engineered features (wick/body, momentum, VWAP distance, time of day).
- **Fakeout classifier**: LightGBM (fallback GradientBoosting) binary classifier returning probability of fake breakout.
- **Policy**: Heuristic or RL-style (stub) policy that translates signals into leverage, stops, targets, and scaling decisions.
- **Risk controls**: Kill-switch on drawdown, spread, volatility spikes, feature drift; max leverage and position caps.
- **Backtest**: Slippage + spread model, equity curve metrics, per-regime stats, plots.
- **Monitoring**: Simple feature drift detector to prevent model drift.
- **FTC Audit Trail**: Comprehensive logging of all trading decisions, risk controls, and model predictions for compliance.
- **Web Interface**: Flask-based web UI for running backtests, viewing audit logs, and monitoring system status.

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Edit `config.yaml` with your OANDA practice credentials and preferences.
3. Download data (synthetic mode can be enabled in `config.yaml`):
   ```bash
   python -m src.cli download-data
   ```
4. Train models:
   ```bash
   python -m src.cli train-regime
   python -m src.cli train-classifier
   ```
5. Run backtest:
   ```bash
   python -m src.cli backtest --policy heuristic
   ```
6. **Start web interface** (recommended):
   ```bash
   python -m src.cli web
   ```
   Then open your browser to `http://localhost:5000` to access the web dashboard.
7. Paper trade stub (extend `TradeExecutor` for live streaming):
   ```bash
   python -m src.cli paper-trade
   ```

## Configuration
See `config.yaml` for pairs, timeframes, risk limits, slippage/spread, and monitoring thresholds.

## Docker Deployment

Run multiple currency pair instances simultaneously using Docker Compose:

### Quick Start with Docker
```bash
# Build and start all currency pair instances
docker-compose up -d

# Access individual instances:
# USD/JPY: http://localhost:5000
# EUR/USD: http://localhost:5001
# GBP/USD: http://localhost:5002
# AUD/USD: http://localhost:5003

# View logs
docker-compose logs -f

# Stop all instances
docker-compose down
```

### Single Currency Pair with Docker
```bash
# Build the Docker image
docker build -t fmmforex .

# Run a single instance
docker run -p 5000:5000 \
  -e TRADING_PAIR=USD_JPY \
  -e TIMEFRAME=M15 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  fmmforex
```

### Multi-Currency Comparison
Use the CLI or web interface to compare multiple currency pairs:

```bash
# CLI comparison
python -m src.cli compare-pairs --pairs USD_JPY,EUR_USD,GBP_USD --policy heuristic --charts

# Or use the web interface at http://localhost:5000/compare
```

The multi-currency comparison feature allows you to:
- Run backtests on multiple currency pairs simultaneously
- Compare equity curves side-by-side
- Analyze performance metrics across pairs
- View interactive comparison charts
- Identify the best performing currency pairs for your strategy

## Testing
Run unit tests for features and risk controls:
```bash
pytest
```

## Web Interface

The web interface provides:
- **Dashboard**: System status and configuration overview
- **Backtest Runner**: Execute backtests with different policies via the UI
- **Multi-Currency Comparison**: Compare performance across different currency pairs with interactive charts
- **Audit Logs**: View comprehensive FTC compliance audit trail
- **Configuration Viewer**: Inspect current system settings

Access at `http://localhost:5000` after running `python -m src.cli web`.

## FTC Audit Trail

All trading decisions, risk controls, and model predictions are logged for compliance:
- Trade entries and exits with timestamps
- Risk control decisions (kill-switch activations, position limits)
- Model predictions with confidence scores
- Configuration changes

Audit logs are stored in JSONL format at `data/audit.jsonl` (configurable in `config.yaml`).

## Files
- `src/data/oanda_client.py` – OANDA client + synthetic data.
- `src/data/cache.py` – SQLite caching layer.
- `src/features/build_features.py` – ATR, regime and breakout features.
- `src/models/regime.py` – Regime detector.
- `src/models/fakeout_classifier.py` – Fakeout probability model.
- `src/models/rl_policy.py` – Heuristic and RL-style policy outputs.
- `src/backtest/engine.py` – Backtest with slippage, metrics, plots.
- `src/comparison/runner.py` – Multi-currency backtest runner.
- `src/comparison/charts.py` – Comparison chart generation.
- `src/trade/executor.py` – Paper/live execution stub.
- `src/risk/controls.py` – Risk engine and kill switch.
- `src/monitoring/drift.py` – Feature drift monitor.
- `src/audit/logger.py` – FTC compliance audit logger.
- `src/web/app.py` – Flask web application.
- `src/cli.py` – CLI entrypoints.
- `Dockerfile` – Docker container definition.
- `docker-compose.yml` – Multi-instance orchestration.
