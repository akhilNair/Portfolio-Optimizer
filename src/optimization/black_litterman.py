"""Step 3: Black-Litterman model for blending historical and real-time optimizations."""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from src.models.schemas import (
    PortfolioWeights,
    BlackLittermanOutput,
    OptimizationObjective,
)


class BlackLittermanBlender:
    """
    Black-Litterman model blending Step 1 (historical) and Step 2 (real-time) results.

    Prior (equilibrium) returns:    pi = delta * Sigma * w_mkt
    Views from weight tilts:        P, Q derived from Step 1/Step 2 deviations
    Posterior returns:              mu_BL = [(tau*Sigma)^-1 + P^T Omega^-1 P]^-1
                                           * [(tau*Sigma)^-1 pi + P^T Omega^-1 Q]
    Final weights:                  CVXPY optimization with mu_BL, Sigma_BL
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.05,
        step1_confidence: float = 0.6,
        step2_confidence: float = 0.4,
    ):
        self.delta = risk_aversion
        self.tau = tau
        self.rf = risk_free_rate
        self.step1_conf = step1_confidence
        self.step2_conf = step2_confidence

    def blend(
        self,
        step1_weights: PortfolioWeights,
        step2_weights: PortfolioWeights,
        cov_matrix: list[list[float]],
        tickers: list[str],
        market_cap_weights: dict[str, float] | None = None,
    ) -> BlackLittermanOutput:
        n = len(tickers)
        Sigma = np.array(cov_matrix)

        # Ensure PSD
        Sigma = (Sigma + Sigma.T) / 2
        eigvals = np.linalg.eigvalsh(Sigma)
        if eigvals.min() < 0:
            Sigma += (-eigvals.min() + 1e-8) * np.eye(n)

        # Market weights: use provided or average of Step 1 and Step 2
        if market_cap_weights:
            w_mkt = np.array([market_cap_weights.get(t, 1.0 / n) for t in tickers])
        else:
            w1 = np.array([step1_weights.weights.get(t, 0) for t in tickers])
            w2 = np.array([step2_weights.weights.get(t, 0) for t in tickers])
            w_mkt = (w1 + w2) / 2.0
            w_mkt /= w_mkt.sum()

        # Equilibrium excess returns
        pi = self.delta * Sigma @ w_mkt

        # Construct views from weight tilts
        P, Q, omega_diag = self._construct_views(
            tickers, step1_weights, step2_weights, pi, Sigma
        )

        k = P.shape[0]
        Omega = np.diag(omega_diag)

        # Posterior returns (BL formula)
        tau_Sigma = self.tau * Sigma
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        Omega_inv = np.linalg.inv(Omega)

        M = tau_Sigma_inv + P.T @ Omega_inv @ P
        M_inv = np.linalg.inv(M)

        mu_BL = M_inv @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)

        # Posterior covariance
        Sigma_BL = Sigma + M_inv

        # Final optimization with posterior estimates
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, w >= 0, w <= 0.40]
        objective = cp.Maximize(mu_BL @ w - 0.5 * self.delta * cp.quad_form(w, Sigma_BL))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            final_w = np.clip(w.value, 0, 0.40)
            final_w /= final_w.sum()
        else:
            final_w = w_mkt

        exp_ret = float(mu_BL @ final_w)
        exp_vol = float(np.sqrt(final_w @ Sigma_BL @ final_w))
        sharpe = (exp_ret - self.rf) / exp_vol if exp_vol > 0 else 0.0

        posterior_weights = PortfolioWeights(
            weights={t: round(float(wi), 6) for t, wi in zip(tickers, final_w)},
            expected_return=round(exp_ret, 6),
            expected_volatility=round(exp_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            objective_used=OptimizationObjective.MAX_SHARPE,
            metadata={"blending_method": "black_litterman", "tau": self.tau, "delta": self.delta},
        )

        return BlackLittermanOutput(
            posterior_weights=posterior_weights,
            prior_returns={t: round(float(p), 6) for t, p in zip(tickers, pi)},
            posterior_returns={t: round(float(m), 6) for t, m in zip(tickers, mu_BL)},
            blend_diagnostics={
                "num_views": k,
                "step1_confidence": self.step1_conf,
                "step2_confidence": self.step2_conf,
                "tracking_error_vs_prior": float(
                    np.sqrt((final_w - w_mkt) @ Sigma @ (final_w - w_mkt))
                ),
            },
        )

    def _construct_views(
        self,
        tickers: list[str],
        step1_weights: PortfolioWeights,
        step2_weights: PortfolioWeights,
        pi: np.ndarray,
        Sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build P (views matrix), Q (view returns), and omega diagonal
        from Step 1 and Step 2 weight tilts relative to equal weight.
        """
        n = len(tickers)
        equal_w = 1.0 / n

        views_P = []
        views_Q = []
        views_omega = []

        for step_weights, confidence in [
            (step1_weights, self.step1_conf),
            (step2_weights, self.step2_conf),
        ]:
            w_arr = np.array([step_weights.weights.get(t, equal_w) for t in tickers])
            deviations = w_arr - equal_w

            overweight_idx = np.where(deviations > 0.02)[0]
            underweight_idx = np.where(deviations < -0.02)[0]

            if len(overweight_idx) == 0 or len(underweight_idx) == 0:
                continue

            most_underweight = underweight_idx[np.argmin(deviations[underweight_idx])]

            for idx in overweight_idx:
                p = np.zeros(n)
                p[idx] = 1.0
                p[most_underweight] = -1.0
                views_P.append(p)

                # View magnitude proportional to deviation and equilibrium return
                q_val = deviations[idx] * abs(pi[idx]) * 2.0
                views_Q.append(q_val)

                # Uncertainty inversely proportional to confidence
                omega_val = (1.0 - confidence) * (p @ (self.tau * Sigma) @ p)
                views_omega.append(max(omega_val, 1e-8))

        if len(views_P) == 0:
            # No significant tilts: single neutral view
            views_P.append(np.ones(n) / n)
            views_Q.append(float(np.mean(pi)))
            views_omega.append(self.tau * float(np.mean(np.diag(Sigma))))

        return np.array(views_P), np.array(views_Q), np.array(views_omega)
