"""Step 2: CVXPY optimization with live market data and sentiment adjustment."""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from src.models.schemas import (
    PortfolioWeights,
    OptimizationObjective,
    RealTimeMarketSnapshot,
    SentimentScores,
)


class RealTimeOptimizer:
    """
    Same core CVXPY formulations as HistoricalOptimizer, but with:
    1. 30-day rolling window covariance (more responsive to recent conditions)
    2. Sentiment-adjusted expected returns:
       mu_adjusted[i] = mu_raw[i] * (1 + alpha * sentiment[i])
    """

    def __init__(self, risk_free_rate: float = 0.05, sentiment_alpha: float = 0.15):
        self.rf = risk_free_rate
        self.alpha = sentiment_alpha

    def optimize(
        self,
        tickers: list[str],
        snapshot: RealTimeMarketSnapshot,
        sentiment: SentimentScores,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        max_weight: float = 0.40,
    ) -> PortfolioWeights:
        n = len(tickers)
        mu_raw = np.array(snapshot.mean_returns_30d)
        Sigma = np.array(snapshot.cov_matrix_30d)

        # Sentiment adjustment
        sentiment_vec = np.array([sentiment.scores.get(t, 0.0) for t in tickers])
        mu_adjusted = mu_raw * (1.0 + self.alpha * sentiment_vec)

        # Annualize from daily to yearly
        mu_annual = mu_adjusted * 252
        Sigma_annual = Sigma * 252

        # Ensure PSD
        Sigma_annual = (Sigma_annual + Sigma_annual.T) / 2
        eigvals = np.linalg.eigvalsh(Sigma_annual)
        if eigvals.min() < 0:
            Sigma_annual += (-eigvals.min() + 1e-8) * np.eye(n)

        if objective == OptimizationObjective.MAX_SHARPE:
            weights = self._max_sharpe(n, mu_annual, Sigma_annual, max_weight)
        elif objective == OptimizationObjective.MIN_VARIANCE:
            weights = self._min_variance(n, Sigma_annual, max_weight)
        elif objective == OptimizationObjective.MAX_RETURN:
            weights = self._max_return(n, mu_annual, Sigma_annual, max_weight)
        else:
            weights = np.ones(n) / n

        weights = np.clip(weights, 0, max_weight)
        weights /= weights.sum()

        exp_ret = float(mu_annual @ weights)
        exp_vol = float(np.sqrt(weights @ Sigma_annual @ weights))
        sharpe = (exp_ret - self.rf) / exp_vol if exp_vol > 0 else 0.0

        return PortfolioWeights(
            weights={t: round(float(w_i), 6) for t, w_i in zip(tickers, weights)},
            expected_return=round(exp_ret, 6),
            expected_volatility=round(exp_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            objective_used=objective,
            metadata={
                "sentiment_adjustment_alpha": self.alpha,
                "sentiment_scores": sentiment.scores,
            },
        )

    def _max_sharpe(self, n, mu, Sigma, max_weight) -> np.ndarray:
        """Efficient frontier sweep to find max-Sharpe portfolio."""
        best_sharpe = -np.inf
        best_weights = np.ones(n) / n

        targets = np.linspace(mu.min(), mu.max(), 20)
        for target_ret in targets:
            w = cp.Variable(n)
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                w <= max_weight,
                mu @ w >= target_ret,
            ]
            prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
            prob.solve(solver=cp.OSQP, warm_start=True)

            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                ret = float(mu @ w.value)
                vol = float(np.sqrt(w.value @ Sigma @ w.value))
                sharpe = (ret - self.rf) / vol if vol > 0 else 0
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = w.value.copy()

        return best_weights

    def _min_variance(self, n, Sigma, max_weight) -> np.ndarray:
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        prob.solve(solver=cp.OSQP)
        return w.value if w.value is not None else np.ones(n) / n

    def _max_return(self, n, mu, Sigma, max_weight, max_vol: float = 0.25) -> np.ndarray:
        w = cp.Variable(n)
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_weight,
            cp.quad_form(w, Sigma) <= max_vol**2,
        ]
        prob = cp.Problem(cp.Maximize(mu @ w), constraints)
        prob.solve(solver=cp.OSQP)
        return w.value if w.value is not None else np.ones(n) / n
