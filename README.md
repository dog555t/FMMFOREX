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
6. Paper trade stub (extend `TradeExecutor` for live streaming):
   ```bash
   python -m src.cli paper-trade
   ```

## Configuration
See `config.yaml` for pairs, timeframes, risk limits, slippage/spread, and monitoring thresholds.

## Testing
Run unit tests for features and risk controls:
```bash
pytest
```

## Files
- `src/data/oanda_client.py` – OANDA client + synthetic data.
- `src/data/cache.py` – SQLite caching layer.
- `src/features/build_features.py` – ATR, regime and breakout features.
- `src/models/regime.py` – Regime detector.
- `src/models/fakeout_classifier.py` – Fakeout probability model.
- `src/models/rl_policy.py` – Heuristic and RL-style policy outputs.
- `src/backtest/engine.py` – Backtest with slippage, metrics, plots.
- `src/trade/executor.py` – Paper/live execution stub.
- `src/risk/controls.py` – Risk engine and kill switch.
- `src/monitoring/drift.py` – Feature drift monitor.
- `src/cli.py` – CLI entrypoints.
