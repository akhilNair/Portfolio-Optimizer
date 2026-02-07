"""Result display formatting utilities."""

from __future__ import annotations

from src.models.schemas import BlackLittermanOutput, PortfolioWeights


def format_weights_table(
    step1: PortfolioWeights,
    step2: PortfolioWeights,
    final: BlackLittermanOutput,
) -> str:
    """Format a comparison table of weights across all three steps."""
    tickers = sorted(final.posterior_weights.weights.keys())

    rows = []
    for ticker in tickers:
        w1 = step1.weights.get(ticker, 0)
        w2 = step2.weights.get(ticker, 0)
        wf = final.posterior_weights.weights.get(ticker, 0)
        rows.append(f"| {ticker} | {w1:.2%} | {w2:.2%} | {wf:.2%} |")

    header = "| Ticker | Historical | Real-time | Blended |\n|--------|-----------|-----------|---------|"
    return header + "\n" + "\n".join(rows)


def format_metrics_summary(
    step1: PortfolioWeights,
    step2: PortfolioWeights,
    final: BlackLittermanOutput,
) -> str:
    """Format performance metrics comparison."""
    bl = final.posterior_weights
    lines = [
        "| Metric | Historical | Real-time | Blended |",
        "|--------|-----------|-----------|---------|",
        f"| Return | {step1.expected_return:.2%} | {step2.expected_return:.2%} | {bl.expected_return:.2%} |",
        f"| Volatility | {step1.expected_volatility:.2%} | {step2.expected_volatility:.2%} | {bl.expected_volatility:.2%} |",
        f"| Sharpe | {step1.sharpe_ratio:.3f} | {step2.sharpe_ratio:.3f} | {bl.sharpe_ratio:.3f} |",
    ]
    return "\n".join(lines)
