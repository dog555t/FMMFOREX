"""Chart generation for multi-currency comparison."""
from __future__ import annotations

import base64
from io import BytesIO
from typing import List

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from src.comparison.runner import CurrencyPairResult


def create_comparison_charts(results: List[CurrencyPairResult]) -> dict:
    """Create comparison charts for multiple currency pairs.
    
    Returns:
        Dictionary with chart names as keys and base64-encoded images as values.
    """
    if not results:
        return {}
    
    charts = {}
    
    # 1. Equity Curves Comparison
    charts["equity_curves"] = _create_equity_curves_chart(results)
    
    # 2. Metrics Comparison Bar Chart
    charts["metrics_comparison"] = _create_metrics_comparison(results)
    
    # 3. Drawdown Comparison
    charts["drawdown_comparison"] = _create_drawdown_comparison(results)
    
    # 4. Win Rate and Sharpe Comparison
    charts["performance_comparison"] = _create_performance_comparison(results)
    
    return charts


def _create_equity_curves_chart(results: List[CurrencyPairResult]) -> str:
    """Create overlaid equity curves for all pairs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for result in results:
        equity = result.result.equity_curve
        # Normalize to percentage return from initial balance
        normalized = (equity / equity.iloc[0] - 1) * 100
        ax.plot(range(len(normalized)), normalized, label=result.pair, linewidth=2)
    
    ax.set_title("Equity Curves Comparison", fontsize=16, fontweight='bold')
    ax.set_xlabel("Time (bars)", fontsize=12)
    ax.set_ylabel("Return (%)", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    return _fig_to_base64(fig)


def _create_metrics_comparison(results: List[CurrencyPairResult]) -> str:
    """Create bar chart comparing key metrics."""
    pairs = [r.pair for r in results]
    final_balances = [r.metrics.get("final_balance", 0) for r in results]
    initial_balance = results[0].metrics.get("final_balance", 100000) / (1 + (results[0].metrics.get("final_balance", 100000) - 100000) / 100000)
    
    # Calculate returns as percentage
    returns = [(fb - initial_balance) / initial_balance * 100 for fb in final_balances]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if r > 0 else 'red' for r in returns]
    bars = ax.bar(pairs, returns, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{ret:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    ax.set_title("Return Comparison by Currency Pair", fontsize=16, fontweight='bold')
    ax.set_xlabel("Currency Pair", fontsize=12)
    ax.set_ylabel("Return (%)", fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return _fig_to_base64(fig)


def _create_drawdown_comparison(results: List[CurrencyPairResult]) -> str:
    """Create bar chart comparing max drawdowns."""
    pairs = [r.pair for r in results]
    drawdowns = [r.metrics.get("max_drawdown", 0) * 100 for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(pairs, drawdowns, color='darkred', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, dd in zip(bars, drawdowns):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{dd:.2f}%',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    ax.set_title("Maximum Drawdown Comparison", fontsize=16, fontweight='bold')
    ax.set_xlabel("Currency Pair", fontsize=12)
    ax.set_ylabel("Max Drawdown (%)", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return _fig_to_base64(fig)


def _create_performance_comparison(results: List[CurrencyPairResult]) -> str:
    """Create comparison of win rate and Sharpe ratio."""
    pairs = [r.pair for r in results]
    win_rates = [r.metrics.get("win_rate", 0) * 100 for r in results]
    sharpes = [r.metrics.get("sharpe", 0) for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Win Rate
    bars1 = ax1.bar(pairs, win_rates, color='steelblue', alpha=0.7, edgecolor='black')
    for bar, wr in zip(bars1, win_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{wr:.1f}%',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    ax1.set_title("Win Rate Comparison", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Currency Pair", fontsize=11)
    ax1.set_ylabel("Win Rate (%)", fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Sharpe Ratio
    colors = ['green' if s > 0 else 'red' for s in sharpes]
    bars2 = ax2.bar(pairs, sharpes, color=colors, alpha=0.7, edgecolor='black')
    for bar, sr in zip(bars2, sharpes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                f'{sr:.2f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    ax2.set_title("Sharpe Ratio Comparison", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Currency Pair", fontsize=11)
    ax2.set_ylabel("Sharpe Ratio", fontsize=11)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return image_base64


__all__ = ["create_comparison_charts"]
