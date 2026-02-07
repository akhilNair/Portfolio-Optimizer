"""Tests for the Black-Litterman blender."""

import numpy as np
import pytest

from src.optimization.black_litterman import BlackLittermanBlender


class TestBlackLittermanBlender:
    def test_blend_produces_valid_weights(
        self, sample_tickers, sample_step1_weights, sample_step2_weights, sample_cov_matrix
    ):
        blender = BlackLittermanBlender()
        result = blender.blend(
            step1_weights=sample_step1_weights,
            step2_weights=sample_step2_weights,
            cov_matrix=sample_cov_matrix,
            tickers=sample_tickers,
        )

        weights = result.posterior_weights.weights
        assert abs(sum(weights.values()) - 1.0) < 0.05
        assert all(w >= 0 for w in weights.values())
        assert all(w <= 0.41 for w in weights.values())

    def test_posterior_returns_differ_from_prior(
        self, sample_tickers, sample_step1_weights, sample_step2_weights, sample_cov_matrix
    ):
        blender = BlackLittermanBlender()
        result = blender.blend(
            step1_weights=sample_step1_weights,
            step2_weights=sample_step2_weights,
            cov_matrix=sample_cov_matrix,
            tickers=sample_tickers,
        )

        # Posterior returns should differ from prior (views have impact)
        prior = list(result.prior_returns.values())
        posterior = list(result.posterior_returns.values())
        assert prior != posterior

    def test_diagnostics_populated(
        self, sample_tickers, sample_step1_weights, sample_step2_weights, sample_cov_matrix
    ):
        blender = BlackLittermanBlender()
        result = blender.blend(
            step1_weights=sample_step1_weights,
            step2_weights=sample_step2_weights,
            cov_matrix=sample_cov_matrix,
            tickers=sample_tickers,
        )

        diag = result.blend_diagnostics
        assert "num_views" in diag
        assert "step1_confidence" in diag
        assert "step2_confidence" in diag
        assert "tracking_error_vs_prior" in diag

    def test_confidence_affects_blending(
        self, sample_tickers, sample_step1_weights, sample_step2_weights, sample_cov_matrix
    ):
        """Higher step1 confidence should tilt result toward step1 weights."""
        blender_high_s1 = BlackLittermanBlender(step1_confidence=0.9, step2_confidence=0.1)
        blender_high_s2 = BlackLittermanBlender(step1_confidence=0.1, step2_confidence=0.9)

        result_s1 = blender_high_s1.blend(
            step1_weights=sample_step1_weights,
            step2_weights=sample_step2_weights,
            cov_matrix=sample_cov_matrix,
            tickers=sample_tickers,
        )

        result_s2 = blender_high_s2.blend(
            step1_weights=sample_step1_weights,
            step2_weights=sample_step2_weights,
            cov_matrix=sample_cov_matrix,
            tickers=sample_tickers,
        )

        # The two results should differ (different confidence parameters)
        w_s1 = list(result_s1.posterior_weights.weights.values())
        w_s2 = list(result_s2.posterior_weights.weights.values())
        assert w_s1 != w_s2
