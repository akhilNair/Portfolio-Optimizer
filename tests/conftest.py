"""Shared test fixtures."""

import numpy as np
import pytest

from src.models.schemas import (
    HistoricalReturnsData,
    OptimizationObjective,
    PortfolioWeights,
    RealTimeMarketSnapshot,
    SentimentScores,
)
from datetime import datetime


@pytest.fixture
def sample_tickers():
    return ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]


@pytest.fixture
def sample_returns(sample_tickers):
    """Synthetic annualized mean returns."""
    return [0.15, 0.12, 0.10, 0.20, 0.08]


@pytest.fixture
def sample_cov_matrix(sample_tickers):
    """Synthetic positive-definite covariance matrix."""
    n = len(sample_tickers)
    # Create a correlation matrix with reasonable values
    rng = np.random.default_rng(42)
    A = rng.normal(size=(n, n)) * 0.1
    Sigma = A @ A.T + np.eye(n) * 0.04  # Ensure PSD
    return Sigma.tolist()


@pytest.fixture
def sample_hist_data(sample_tickers, sample_returns, sample_cov_matrix):
    return HistoricalReturnsData(
        tickers=sample_tickers,
        mean_returns=sample_returns,
        cov_matrix=sample_cov_matrix,
        lookback_days=252,
    )


@pytest.fixture
def sample_snapshot(sample_tickers):
    n = len(sample_tickers)
    rng = np.random.default_rng(42)
    daily_returns = (rng.normal(0.0005, 0.02, size=(30, n))).tolist()
    cov = (np.cov(np.array(daily_returns).T)).tolist()
    mean_ret = [float(np.mean([row[i] for row in daily_returns])) for i in range(n)]

    return RealTimeMarketSnapshot(
        tickers=sample_tickers,
        current_prices={"AAPL": 180.0, "MSFT": 410.0, "GOOGL": 140.0, "NVDA": 800.0, "AMZN": 185.0},
        daily_returns_30d=daily_returns,
        cov_matrix_30d=cov,
        mean_returns_30d=mean_ret,
        fetch_timestamp=datetime.now(),
    )


@pytest.fixture
def sample_sentiment(sample_tickers):
    return SentimentScores(
        scores={"AAPL": 0.3, "MSFT": 0.2, "GOOGL": -0.1, "NVDA": 0.5, "AMZN": 0.0},
        rationale={
            "AAPL": "Strong iPhone demand",
            "MSFT": "Azure growth stable",
            "GOOGL": "Regulatory concerns",
            "NVDA": "AI demand surge",
            "AMZN": "Mixed retail outlook",
        },
        overall_market_sentiment=0.2,
        source_summary="Test fixture sentiment",
    )


@pytest.fixture
def sample_step1_weights():
    return PortfolioWeights(
        weights={"AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.15, "NVDA": 0.30, "AMZN": 0.10},
        expected_return=0.15,
        expected_volatility=0.18,
        sharpe_ratio=0.556,
        objective_used=OptimizationObjective.MAX_SHARPE,
    )


@pytest.fixture
def sample_step2_weights():
    return PortfolioWeights(
        weights={"AAPL": 0.20, "MSFT": 0.25, "GOOGL": 0.10, "NVDA": 0.35, "AMZN": 0.10},
        expected_return=0.16,
        expected_volatility=0.20,
        sharpe_ratio=0.550,
        objective_used=OptimizationObjective.MAX_SHARPE,
    )
