"""Tests for the deterministic PDF parser extraction logic."""

import pytest

from src.models.schemas import BarrierDirection, PayoffType
from src.pdf.parser import StructuredNotePDFParser


@pytest.fixture
def parser(tmp_path):
    return StructuredNotePDFParser(cache_dir=str(tmp_path / "cache"))


class TestNoteIdExtraction:
    def test_isin_extraction(self, parser):
        assert parser._extract_note_id("ISIN: CH0123456789 issued by UBS") == "CH0123456789"

    def test_isin_inline(self, parser):
        assert parser._extract_note_id("Note CH1234567890 with coupon") == "CH1234567890"

    def test_explicit_label(self, parser):
        assert parser._extract_note_id("Securities Code: XS1234567") == "XS1234567"

    def test_no_id(self, parser):
        assert parser._extract_note_id("No identifier here") is None


class TestIssuerExtraction:
    def test_explicit_label(self, parser):
        assert parser._extract_issuer("Issuer: Goldman Sachs International\nDate") == "Goldman Sachs International"

    def test_known_issuer_scan(self, parser):
        assert parser._extract_issuer("This product is offered by UBS AG, London Branch") == "UBS"

    def test_jp_morgan(self, parser):
        assert parser._extract_issuer("Issued by JPMorgan Chase") == "JPMorgan"

    def test_no_issuer(self, parser):
        assert parser._extract_issuer("Some random text with no bank") is None


class TestDateExtraction:
    def test_iso_date(self, parser):
        dates = parser._find_dates_in_text("Trade date: 2024-03-15")
        assert len(dates) == 1
        assert dates[0].year == 2024
        assert dates[0].month == 3
        assert dates[0].day == 15

    def test_european_date(self, parser):
        dates = parser._find_dates_in_text("15/03/2024")
        assert len(dates) == 1
        assert dates[0].day == 15
        assert dates[0].month == 3

    def test_named_month(self, parser):
        dates = parser._find_dates_in_text("12 January 2024")
        assert len(dates) == 1
        assert dates[0].month == 1
        assert dates[0].day == 12

    def test_us_named_month(self, parser):
        dates = parser._find_dates_in_text("January 12, 2024")
        assert len(dates) == 1
        assert dates[0].month == 1
        assert dates[0].day == 12

    def test_contextual_dates(self, parser):
        text = "Issue date: 15 March 2024. The maturity date is 15 March 2026."
        issue, maturity = parser._extract_dates(text, text.lower())
        assert issue is not None
        assert maturity is not None
        assert issue.year == 2024
        assert maturity.year == 2026


class TestCurrencyExtraction:
    def test_usd(self, parser):
        assert parser._extract_currency("Notional: USD 1,000,000") == "USD"

    def test_eur(self, parser):
        assert parser._extract_currency("Settlement in EUR") == "EUR"

    def test_chf(self, parser):
        assert parser._extract_currency("Denomination: CHF 5,000") == "CHF"

    def test_default(self, parser):
        assert parser._extract_currency("no currency here") == "USD"


class TestPayoffTypeDetection:
    def test_autocall(self, parser):
        assert parser._detect_payoff_type("this is an autocall note") == PayoffType.AUTOCALL

    def test_autocall_hyphen(self, parser):
        assert parser._detect_payoff_type("auto-call product") == PayoffType.AUTOCALL

    def test_phoenix(self, parser):
        assert parser._detect_payoff_type("phoenix memory coupon note") == PayoffType.PHOENIX

    def test_brc(self, parser):
        assert parser._detect_payoff_type("barrier reverse convertible") == PayoffType.BARRIER_REVERSE_CONVERTIBLE

    def test_capital_protected(self, parser):
        assert parser._detect_payoff_type("capital protected note") == PayoffType.CAPITAL_PROTECTED

    def test_range_accrual(self, parser):
        assert parser._detect_payoff_type("range accrual bond") == PayoffType.RANGE_ACCRUAL

    def test_worst_of(self, parser):
        assert parser._detect_payoff_type("worst-of basket certificate") == PayoffType.VANILLA_BASKET

    def test_default_vanilla(self, parser):
        assert parser._detect_payoff_type("some structured product") == PayoffType.VANILLA_BASKET


class TestCouponExtraction:
    def test_coupon_from_text(self, parser):
        text = "The note pays a coupon of 8.5% per annum"
        result = parser._extract_coupon(text, text.lower(), {})
        assert result == 0.085

    def test_coupon_from_kv(self, parser):
        kv = {"coupon rate": "7.25% p.a."}
        result = parser._extract_coupon("", "", kv)
        assert result == 0.0725

    def test_coupon_rejects_outliers(self, parser):
        text = "The note has 100% participation and a coupon of 6%"
        result = parser._extract_coupon(text, text.lower(), {})
        assert result == 0.06  # Should pick 6%, not 100%

    def test_no_coupon(self, parser):
        assert parser._extract_coupon("no rate here", "no rate here", {}) is None


class TestBarrierExtraction:
    def test_knock_in(self, parser):
        barrier = parser._extract_barrier("knock-in barrier at 60% of initial", {})
        assert barrier is not None
        assert barrier.direction == BarrierDirection.DOWN_IN
        assert barrier.level_pct == 0.60

    def test_down_and_in(self, parser):
        barrier = parser._extract_barrier("down-and-in barrier level 65%", {})
        assert barrier is not None
        assert barrier.direction == BarrierDirection.DOWN_IN
        assert barrier.level_pct == 0.65

    def test_barrier_from_kv(self, parser):
        kv = {"barrier level": "70%"}
        barrier = parser._extract_barrier("barrier type: down and in observation", kv)
        assert barrier is not None
        assert barrier.level_pct == 0.70

    def test_no_barrier(self, parser):
        assert parser._extract_barrier("no barrier mentioned", {}) is None

    def test_observation_type_continuous(self, parser):
        barrier = parser._extract_barrier("knock-in barrier at 60% with continuous observation", {})
        assert barrier is not None
        assert barrier.observation == "american"


class TestMaturityExtraction:
    def test_years(self, parser):
        assert parser._extract_maturity_years("2 year maturity", {}) == 2.0

    def test_months(self, parser):
        result = parser._extract_maturity_years("18 months maturity", {})
        assert result == 1.5

    def test_from_kv(self, parser):
        kv = {"tenor": "3 years"}
        assert parser._extract_maturity_years("", kv) == 3.0

    def test_months_from_kv(self, parser):
        kv = {"maturity": "24 months"}
        assert parser._extract_maturity_years("", kv) == 2.0


class TestObservationFrequency:
    def test_quarterly(self, parser):
        assert parser._extract_observation_frequency("quarterly observation dates", {}) == "quarterly"

    def test_monthly_coupon(self, parser):
        assert parser._extract_observation_frequency("monthly coupon payment", {}) == "monthly"

    def test_from_kv(self, parser):
        kv = {"observation frequency": "Semi-Annual"}
        assert parser._extract_observation_frequency("", kv) == "semi_annual"


class TestTableColumnMapping:
    def test_standard_headers(self, parser):
        header = ["Ticker", "Company Name", "Weight", "Initial Fixing"]
        col_map = parser._map_table_columns(header)
        assert col_map["ticker"] == 0
        assert col_map["name"] == 1
        assert col_map["weight"] == 2
        assert col_map["fixing"] == 3

    def test_bloomberg_header(self, parser):
        header = ["Bloomberg Code", "Underlying", "Allocation %"]
        col_map = parser._map_table_columns(header)
        assert col_map["ticker"] == 0
        assert col_map["name"] == 1
        assert col_map["weight"] == 2

    def test_no_useful_columns(self, parser):
        header = ["Date", "Amount", "Status"]
        col_map = parser._map_table_columns(header)
        assert col_map == {}


class TestTickerResolution:
    def test_known_ticker(self, parser):
        assert parser._resolve_ticker("AAPL", None) == "AAPL"

    def test_ticker_from_name(self, parser):
        assert parser._resolve_ticker(None, "Apple Inc.") == "AAPL"

    def test_unknown_ticker(self, parser):
        assert parser._resolve_ticker("ZZZZ", None) is None

    def test_ticker_with_noise(self, parser):
        assert parser._resolve_ticker(" NVDA ", None) == "NVDA"


class TestBasketFromText:
    def test_finds_known_tickers(self, parser):
        text = "The basket includes AAPL, MSFT, and NVDA as underlyings."
        basket = parser._extract_basket_from_text(text)
        tickers = {a.ticker for a in basket}
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "NVDA" in tickers

    def test_filters_false_positives(self, parser):
        text = "THE CEO said USD returns FOR ALL ETF products including AAPL"
        basket = parser._extract_basket_from_text(text)
        tickers = {a.ticker for a in basket}
        assert "AAPL" in tickers
        assert "THE" not in tickers
        assert "CEO" not in tickers
        assert "USD" not in tickers
        assert "FOR" not in tickers
        assert "ALL" not in tickers
        assert "ETF" not in tickers


class TestCapitalProtection:
    def test_from_text(self, parser):
        result = parser._extract_capital_protection("capital protection of 90%", {})
        assert result == 0.90

    def test_from_kv(self, parser):
        kv = {"capital protection": "95%"}
        result = parser._extract_capital_protection("", kv)
        assert result == 0.95

    def test_no_protection(self, parser):
        assert parser._extract_capital_protection("no protection", {}) is None
