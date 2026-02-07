"""Step 1: CVXPY portfolio optimization on historical returns."""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from src.models.schemas import PortfolioWeights, OptimizationObjective


class HistoricalOptimizer:
    """
    Markowitz mean-variance optimization using CVXPY.

    Formulations:
    - MAX_SHARPE: Cornuejols-Tutuncu transformation to convex QP
    - MIN_VARIANCE: min w^T Sigma w, s.t. sum(w)=1, w>=0
    - MAX_RETURN: max mu^T w, s.t. volatility constraint
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.rf = risk_free_rate

    def optimize(
        self,
        tickers: list[str],
        mean_returns: list[float],
        cov_matrix: list[list[float]],
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        max_weight: float = 0.40,
        min_weight: float = 0.0,
    ) -> PortfolioWeights:
        n = len(tickers)
        mu = np.array(mean_returns)
        Sigma = np.array(cov_matrix)

        # Ensure Sigma is positive semi-definite
        Sigma = (Sigma + Sigma.T) / 2
        eigvals = np.linalg.eigvalsh(Sigma)
        if eigvals.min() < 0:
            Sigma += (-eigvals.min() + 1e-8) * np.eye(n)

        if objective == OptimizationObjective.MAX_SHARPE:
            return self._max_sharpe(tickers, mu, Sigma, n, max_weight)
        elif objective == OptimizationObjective.MIN_VARIANCE:
            return self._min_variance(tickers, mu, Sigma, n, max_weight, min_weight)
        elif objective == OptimizationObjective.MAX_RETURN:
            return self._max_return(tickers, mu, Sigma, n, max_weight, min_weight)
        else:
            return self._min_variance(tickers, mu, Sigma, n, max_weight, min_weight)

    def _max_sharpe(self, tickers, mu, Sigma, n, max_weight) -> PortfolioWeights:
        """
        Cornuejols-Tutuncu transformation for max-Sharpe:
        Introduce y such that (mu - rf)^T y = 1, then minimize y^T Sigma y.
        Recover w = y / 1^T y.
        """
        y = cp.Variable(n)
        excess_mu = mu - self.rf

        # If all excess returns are non-positive, fall back to min-variance
        if np.all(excess_mu <= 0):
            return self._min_variance(tickers, mu, Sigma, n, max_weight, 0.0)

        constraints = [
            excess_mu @ y == 1,
            y >= 0,
        ]

        objective = cp.Minimize(cp.quad_form(y, Sigma))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            return self._min_variance(tickers, mu, Sigma, n, max_weight, 0.0)

        raw_weights = y.value / np.sum(y.value)
        raw_weights = np.clip(raw_weights, 0, max_weight)
        raw_weights /= raw_weights.sum()

        return self._build_result(
            tickers, raw_weights, mu, Sigma, OptimizationObjective.MAX_SHARPE
        )

    def _min_variance(self, tickers, mu, Sigma, n, max_weight, min_weight) -> PortfolioWeights:
        w = cp.Variable(n)
        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight,
        ]
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        weights = w.value if w.value is not None else np.ones(n) / n
        return self._build_result(
            tickers, weights, mu, Sigma, OptimizationObjective.MIN_VARIANCE
        )

    def _max_return(
        self, tickers, mu, Sigma, n, max_weight, min_weight, max_volatility: float = 0.25
    ) -> PortfolioWeights:
        w = cp.Variable(n)
        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight,
            cp.quad_form(w, Sigma) <= max_volatility**2,
        ]
        objective = cp.Maximize(mu @ w)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        weights = w.value if w.value is not None else np.ones(n) / n
        return self._build_result(
            tickers, weights, mu, Sigma, OptimizationObjective.MAX_RETURN
        )

    def _build_result(self, tickers, weights, mu, Sigma, objective) -> PortfolioWeights:
        w = np.array(weights)
        w = np.clip(w, 0, None)
        w /= w.sum()
        exp_ret = float(mu @ w)
        exp_vol = float(np.sqrt(w @ Sigma @ w))
        sharpe = (exp_ret - self.rf) / exp_vol if exp_vol > 0 else 0.0

        return PortfolioWeights(
            weights={t: round(float(w_i), 6) for t, w_i in zip(tickers, w)},
            expected_return=round(exp_ret, 6),
            expected_volatility=round(exp_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            objective_used=objective,
        )
