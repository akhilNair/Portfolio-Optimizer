"""Tests for Pydantic model validation."""

import pytest

from src.models.schemas import (
    AnalystRequest,
    AssetInBasket,
    BarrierCondition,
    BarrierDirection,
    PayoffStructure,
    PayoffType,
    PortfolioWeights,
    OptimizationObjective,
    StructuredNote,
)


class TestStructuredNote:
    def test_create_valid_note(self):
        note = StructuredNote(
            note_id="TEST001",
            issuer="Test Bank",
            currency="USD",
            underlying_basket=[
                AssetInBasket(ticker="AAPL", name="Apple Inc.", weight_in_note=0.5),
                AssetInBasket(ticker="MSFT", name="Microsoft Corp.", weight_in_note=0.5),
            ],
            payoff=PayoffStructure(
                payoff_type=PayoffType.AUTOCALL,
                coupon_rate_annual=0.08,
                autocall_trigger_pct=1.0,
                barrier=BarrierCondition(
                    direction=BarrierDirection.DOWN_IN,
                    level_pct=0.60,
                ),
                maturity_years=2.0,
                observation_frequency="quarterly",
            ),
            source_pdf="test.pdf",
        )

        assert note.note_id == "TEST001"
        assert len(note.underlying_basket) == 2
        assert note.payoff.payoff_type == PayoffType.AUTOCALL

    def test_all_payoff_types(self):
        for pt in PayoffType:
            payoff = PayoffStructure(payoff_type=pt)
            assert payoff.payoff_type == pt


class TestPortfolioWeights:
    def test_valid_weights(self):
        pw = PortfolioWeights(
            weights={"AAPL": 0.5, "MSFT": 0.5},
            expected_return=0.12,
            expected_volatility=0.15,
            sharpe_ratio=0.467,
            objective_used=OptimizationObjective.MAX_SHARPE,
        )
        assert sum(pw.weights.values()) == 1.0

    def test_invalid_weights_rejected(self):
        with pytest.raises(ValueError, match="Weights sum"):
            PortfolioWeights(
                weights={"AAPL": 0.3, "MSFT": 0.3},
                expected_return=0.12,
                expected_volatility=0.15,
                objective_used=OptimizationObjective.MAX_SHARPE,
            )


class TestAnalystRequest:
    def test_minimal_request(self):
        req = AnalystRequest()
        assert req.risk_tolerance == "moderate"

    def test_full_request(self):
        req = AnalystRequest(
            desired_payoff_type=PayoffType.AUTOCALL,
            desired_underlyings=["AAPL", "NVDA"],
            sector_preferences=["tech"],
            min_coupon=0.08,
            max_maturity_years=3.0,
            risk_tolerance="aggressive",
            geopolitical_concerns=["US-China trade war"],
        )
        assert req.desired_payoff_type == PayoffType.AUTOCALL
        assert len(req.desired_underlyings) == 2
