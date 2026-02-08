"""Canonical Pydantic data models for the portfolio optimization engine."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────


class PayoffType(str, Enum):
    AUTOCALL = "autocall"
    BARRIER_REVERSE_CONVERTIBLE = "barrier_reverse_convertible"
    PHOENIX = "phoenix"
    CAPITAL_PROTECTED = "capital_protected"
    VANILLA_BASKET = "vanilla_basket"
    RANGE_ACCRUAL = "range_accrual"


class BarrierDirection(str, Enum):
    DOWN_IN = "down_in"
    DOWN_OUT = "down_out"
    UP_IN = "up_in"
    UP_OUT = "up_out"


class OptimizationObjective(str, Enum):
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"


# ── Structured Note Models ───────────────────────────────────────────


class AssetInBasket(BaseModel):
    """Single asset within a structured note's underlying basket."""

    ticker: Optional[str] = None
    name: str
    isin: Optional[str] = None
    ticker_bloomberg: Optional[str] = None
    weight_in_note: Optional[float] = None
    initial_fixing: Optional[float] = None


class BarrierCondition(BaseModel):
    direction: BarrierDirection
    level_pct: float  # e.g. 0.60 means 60% of initial
    observation: str = "european"  # "european" | "american" | "daily"


class PayoffStructure(BaseModel):
    payoff_type: PayoffType
    coupon_rate_annual: Optional[float] = None
    autocall_trigger_pct: Optional[float] = None
    barrier: Optional[BarrierCondition] = None
    capital_protection_pct: Optional[float] = None
    maturity_years: Optional[float] = None
    observation_frequency: Optional[str] = None


class StructuredNote(BaseModel):
    """Fully parsed representation of a single structured note from a PDF."""

    note_id: str
    issuer: Optional[str] = None
    issue_date: Optional[date] = None
    maturity_date: Optional[date] = None
    currency: str = "USD"
    underlying_basket: list[AssetInBasket]
    payoff: PayoffStructure
    notional: Optional[float] = None
    source_pdf: str
    raw_text_excerpt: Optional[str] = None


class ParsedNoteIndex(BaseModel):
    """Lightweight index entry for vector-store retrieval."""

    note_id: str
    tickers: list[str]
    payoff_type: PayoffType
    coupon_rate_annual: Optional[float] = None
    maturity_years: Optional[float] = None
    embedding_id: str


# ── Analyst Request ──────────────────────────────────────────────────


class AnalystRequest(BaseModel):
    """What the analyst types into the chatbot, parsed by Claude."""

    desired_payoff_type: Optional[PayoffType] = None
    desired_underlyings: Optional[list[str]] = None
    sector_preferences: Optional[list[str]] = None
    min_coupon: Optional[float] = None
    max_maturity_years: Optional[float] = None
    risk_tolerance: str = "moderate"
    target_notional: Optional[float] = None
    geopolitical_concerns: Optional[list[str]] = None
    additional_constraints: Optional[str] = None


# ── Optimization I/O ─────────────────────────────────────────────────


class HistoricalReturnsData(BaseModel):
    """Input to the historical optimizer."""

    tickers: list[str]
    mean_returns: list[float]
    cov_matrix: list[list[float]]
    lookback_days: int = 252


class PortfolioWeights(BaseModel):
    """Output from any optimizer -- the core deliverable."""

    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: Optional[float] = None
    objective_used: OptimizationObjective
    metadata: dict = Field(default_factory=dict)

    @field_validator("weights")
    @classmethod
    def weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.05:
            raise ValueError(f"Weights sum to {total}, expected ~1.0")
        return v


class RealTimeMarketSnapshot(BaseModel):
    """Current market data pulled from yfinance."""

    tickers: list[str]
    current_prices: dict[str, float]
    daily_returns_30d: list[list[float]]
    cov_matrix_30d: list[list[float]]
    mean_returns_30d: list[float]
    fetch_timestamp: datetime


class SentimentScores(BaseModel):
    """Geopolitical / news sentiment for each asset."""

    scores: dict[str, float]  # ticker -> score in [-1.0, 1.0]
    rationale: dict[str, str]
    overall_market_sentiment: float
    source_summary: str


class BlackLittermanInput(BaseModel):
    """Inputs to the BL model for Step 3 blending."""

    market_cap_weights: dict[str, float]
    historical_weights: PortfolioWeights
    realtime_weights: PortfolioWeights
    cov_matrix: list[list[float]]
    risk_aversion: float = 2.5
    tau: float = 0.05
    view_confidences: list[float]


class BlackLittermanOutput(BaseModel):
    """Final blended portfolio from Step 3."""

    posterior_weights: PortfolioWeights
    prior_returns: dict[str, float]
    posterior_returns: dict[str, float]
    blend_diagnostics: dict
