"""Tests for historical and real-time optimizers."""

import numpy as np
import pytest

from src.models.schemas import OptimizationObjective
from src.optimization.historical import HistoricalOptimizer
from src.optimization.realtime import RealTimeOptimizer


class TestHistoricalOptimizer:
    def test_max_sharpe(self, sample_tickers, sample_returns, sample_cov_matrix):
        optimizer = HistoricalOptimizer(risk_free_rate=0.05)
        result = optimizer.optimize(
            tickers=sample_tickers,
            mean_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            objective=OptimizationObjective.MAX_SHARPE,
        )

        assert abs(sum(result.weights.values()) - 1.0) < 0.05
        assert all(w >= 0 for w in result.weights.values())
        assert result.sharpe_ratio is not None
        assert result.expected_return > 0

    def test_min_variance(self, sample_tickers, sample_returns, sample_cov_matrix):
        optimizer = HistoricalOptimizer()
        result = optimizer.optimize(
            tickers=sample_tickers,
            mean_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            objective=OptimizationObjective.MIN_VARIANCE,
        )

        assert abs(sum(result.weights.values()) - 1.0) < 0.05
        assert result.expected_volatility > 0

    def test_max_return(self, sample_tickers, sample_returns, sample_cov_matrix):
        optimizer = HistoricalOptimizer()
        result = optimizer.optimize(
            tickers=sample_tickers,
            mean_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            objective=OptimizationObjective.MAX_RETURN,
        )

        assert abs(sum(result.weights.values()) - 1.0) < 0.05

    def test_max_weight_constraint(self, sample_tickers, sample_returns, sample_cov_matrix):
        optimizer = HistoricalOptimizer()
        result = optimizer.optimize(
            tickers=sample_tickers,
            mean_returns=sample_returns,
            cov_matrix=sample_cov_matrix,
            objective=OptimizationObjective.MIN_VARIANCE,
            max_weight=0.30,
        )

        assert all(w <= 0.31 for w in result.weights.values())  # small tolerance


class TestRealTimeOptimizer:
    def test_optimize_with_sentiment(self, sample_tickers, sample_snapshot, sample_sentiment):
        optimizer = RealTimeOptimizer(risk_free_rate=0.05, sentiment_alpha=0.15)
        result = optimizer.optimize(
            tickers=sample_tickers,
            snapshot=sample_snapshot,
            sentiment=sample_sentiment,
            objective=OptimizationObjective.MAX_SHARPE,
        )

        assert abs(sum(result.weights.values()) - 1.0) < 0.05
        assert all(w >= 0 for w in result.weights.values())
        assert result.metadata.get("sentiment_adjustment_alpha") == 0.15

    def test_neutral_sentiment_same_as_no_sentiment(
        self, sample_tickers, sample_snapshot
    ):
        """With zero sentiment scores, result should be close to no-sentiment optimization."""
        from src.models.schemas import SentimentScores

        neutral = SentimentScores(
            scores={t: 0.0 for t in sample_tickers},
            rationale={t: "" for t in sample_tickers},
            overall_market_sentiment=0.0,
            source_summary="neutral",
        )

        optimizer = RealTimeOptimizer()
        result = optimizer.optimize(
            tickers=sample_tickers,
            snapshot=sample_snapshot,
            sentiment=neutral,
        )

        assert abs(sum(result.weights.values()) - 1.0) < 0.05
