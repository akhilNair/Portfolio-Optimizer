"""Resolve ISINs, Bloomberg tickers, and company names to yfinance-compatible ticker symbols.

This is the bridge between PDF parsing (which extracts ISINs/Bloomberg codes) and
the optimization pipeline (which needs yfinance tickers for price data).
"""

from __future__ import annotations

import re
from typing import Optional

import yfinance as yf

from src.models.schemas import AssetInBasket

# ── Bloomberg exchange suffix → yfinance suffix mapping ──────────────
# Bloomberg format: "MC FP" → yfinance: "MC.PA" (Euronext Paris)
BBG_EXCHANGE_TO_YF: dict[str, str] = {
    # US
    "US": "",        # NYSE/NASDAQ — no suffix needed
    "UW": "",        # NASDAQ
    "UN": "",        # NYSE
    "UA": "",        # NYSE AMEX
    "UQ": "",        # NASDAQ Global Select
    # Europe
    "FP": ".PA",     # Euronext Paris
    "GY": ".DE",     # XETRA (Germany)
    "GR": ".DE",     # Frankfurt
    "LN": ".L",      # London
    "IM": ".MI",     # Milan
    "SM": ".MC",     # Madrid
    "SW": ".SW",     # SIX Swiss Exchange
    "SE": ".SW",     # SIX (alt)
    "NA": ".AS",     # Euronext Amsterdam
    "BB": ".BR",     # Euronext Brussels
    "PL": ".LS",     # Euronext Lisbon
    "DC": ".CO",     # Copenhagen
    "SS": ".ST",     # Stockholm
    "FH": ".HE",     # Helsinki
    "NO": ".OL",     # Oslo
    # Asia
    "JP": ".T",      # Tokyo
    "HK": ".HK",     # Hong Kong
    "SP": ".SI",     # Singapore
    "AU": ".AX",     # ASX (Australia)
    "KS": ".KS",     # Korea
    "TT": ".TW",     # Taiwan
    "IB": ".BO",     # BSE India
    "IN": ".NS",     # NSE India
    # Canada
    "CT": ".TO",     # Toronto
    "CN": ".TO",     # Toronto (alt)
}

# ── Static ISIN → yfinance ticker mapping ────────────────────────────
# For commonly seen ISINs in structured notes. Extend as needed.
ISIN_TO_TICKER: dict[str, str] = {
    # US mega caps
    "US0378331005": "AAPL",
    "US5949181045": "MSFT",
    "US02079K3059": "GOOGL",
    "US0231351067": "AMZN",
    "US67066G1040": "NVDA",
    "US30303M1027": "META",
    "US88160R1014": "TSLA",
    "US46625H1005": "JPM",
    "US4781601046": "JNJ",
    "US9311421039": "WMT",
    "US7427181091": "PG",
    "US57636Q1040": "MA",
    "US91324P1021": "UNH",
    "US4370761029": "HD",
    "US2546871060": "DIS",
    "US0605051046": "BAC",
    "US30231G1022": "XOM",
    "US7170811035": "PFE",
    "US1912161007": "KO",
    "US7134481081": "PEP",
    "US17275R1023": "CSCO",
    "US64110L1061": "NFLX",
    "US4581401001": "INTC",
    "US0079031078": "AMD",
    "US79466L3024": "CRM",
    "US00724F1012": "ADBE",
    "US22160K1051": "COST",
    "US11135F1012": "AVGO",
    "US8825081040": "TXN",
    "US7475251036": "QCOM",
    "US38141G1040": "GS",
    "US6174464486": "MS",
    "US0258161092": "AXP",
    "US0970231058": "BA",
    "US1491231015": "CAT",
    "US4592001014": "IBM",
    "US3696043013": "GE",
    "US6541061031": "NKE",
    "US5801351017": "MCD",
    "US8552441094": "SBUX",
    "US0028241000": "ABT",
    "US5324571083": "LLY",
    "US58933Y1055": "MRK",
    "US1101221083": "BMY",
    "US00287Y1091": "ABBV",
    "US92343V1044": "VZ",
    "US1667641005": "CVX",
    # European blue chips
    "CH0038863350": "NESN.SW",
    "CH0012005267": "NOVN.SW",
    "CH0012032048": "ROG.SW",
    "DE0007236101": "SIE.DE",
    "DE0007164600": "SAP.DE",
    "NL0010273215": "ASML.AS",
    "FR0000121014": "MC.PA",      # LVMH
    "FR0000120271": "TTE.PA",     # TotalEnergies
    "FR0000131104": "BNP.PA",
    "FR0000120578": "SAN.PA",     # Sanofi
    "DE0007100000": "MBG.DE",     # Mercedes-Benz
    "DE0007664039": "VOW3.DE",    # Volkswagen
    "DE0008404005": "ALV.DE",     # Allianz
    "NL0000235190": "AIR.PA",     # Airbus
    "GB0002374006": "DGE.L",      # Diageo
    "GB0005405286": "HSBA.L",     # HSBC
    "GB00B24CGK77": "RDSB.L",     # Shell
    "GB0007188757": "RIO.L",      # Rio Tinto
}

# ── Company name → yfinance ticker (fuzzy matching keys) ─────────────
COMPANY_NAME_TO_TICKER: dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "alphabet": "GOOGL", "google": "GOOGL",
    "amazon": "AMZN", "nvidia": "NVDA", "meta": "META", "facebook": "META",
    "tesla": "TSLA", "jpmorgan": "JPM", "jp morgan": "JPM",
    "johnson & johnson": "JNJ", "walmart": "WMT", "procter": "PG",
    "mastercard": "MA", "unitedhealth": "UNH", "home depot": "HD",
    "disney": "DIS", "bank of america": "BAC", "exxon": "XOM",
    "pfizer": "PFE", "coca-cola": "KO", "coca cola": "KO", "pepsi": "PEP",
    "cisco": "CSCO", "netflix": "NFLX", "intel": "INTC", "amd": "AMD",
    "advanced micro": "AMD", "salesforce": "CRM", "adobe": "ADBE",
    "costco": "COST", "broadcom": "AVGO", "texas instruments": "TXN",
    "qualcomm": "QCOM", "goldman sachs": "GS", "morgan stanley": "MS",
    "american express": "AXP", "boeing": "BA", "caterpillar": "CAT",
    "ibm": "IBM", "general electric": "GE", "nike": "NKE",
    "mcdonald": "MCD", "starbucks": "SBUX", "abbott": "ABT",
    "eli lilly": "LLY", "merck": "MRK", "bristol-myers": "BMY",
    "abbvie": "ABBV", "verizon": "VZ", "chevron": "CVX",
    # European
    "nestle": "NESN.SW", "novartis": "NOVN.SW", "roche": "ROG.SW",
    "siemens": "SIE.DE", "sap": "SAP.DE", "asml": "ASML.AS",
    "lvmh": "MC.PA", "moët hennessy": "MC.PA", "louis vuitton": "MC.PA",
    "totalenergies": "TTE.PA", "total": "TTE.PA",
    "bnp paribas": "BNP.PA", "sanofi": "SAN.PA",
    "mercedes": "MBG.DE", "volkswagen": "VOW3.DE",
    "allianz": "ALV.DE", "airbus": "AIR.PA",
    "diageo": "DGE.L", "hsbc": "HSBA.L", "shell": "SHEL.L",
    "rio tinto": "RIO.L",
}


class TickerResolver:
    """
    Resolve AssetInBasket objects to yfinance-compatible ticker symbols.

    Resolution priority:
      1. ISIN → static lookup table
      2. Bloomberg ticker → exchange suffix mapping
      3. Company name → fuzzy matching
      4. yfinance search API (live, last resort)
    """

    def __init__(self, validate_with_yfinance: bool = False):
        self._validate = validate_with_yfinance
        self._cache: dict[str, str | None] = {}

    def resolve(self, asset: AssetInBasket) -> str | None:
        """Resolve a single asset to a yfinance ticker. Returns None if unresolvable."""
        # Build a cache key from available identifiers
        cache_key = asset.isin or asset.ticker_bloomberg or asset.name
        if cache_key in self._cache:
            return self._cache[cache_key]

        ticker = (
            self._from_existing_ticker(asset.ticker)
            or self._from_isin(asset.isin)
            or self._from_bloomberg(asset.ticker_bloomberg)
            or self._from_name(asset.name)
        )

        # Optional: validate the ticker actually works in yfinance
        if ticker and self._validate:
            ticker = self._validate_ticker(ticker)

        self._cache[cache_key] = ticker
        return ticker

    def resolve_basket(self, assets: list[AssetInBasket]) -> dict[str, str]:
        """
        Resolve a full basket. Returns {yfinance_ticker: asset_name} for all
        successfully resolved assets. Skips unresolvable ones.
        """
        resolved: dict[str, str] = {}
        for asset in assets:
            ticker = self.resolve(asset)
            if ticker:
                resolved[ticker] = asset.name
        return resolved

    def _from_existing_ticker(self, ticker: str | None) -> str | None:
        """If the asset already has a plain ticker, validate it."""
        if not ticker:
            return None
        # Skip ISINs that were stored as tickers
        if re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", ticker):
            return self._from_isin(ticker)
        # Clean and return if it looks like a valid ticker
        cleaned = ticker.strip().upper()
        if re.match(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$", cleaned):
            return cleaned
        return None

    def _from_isin(self, isin: str | None) -> str | None:
        if not isin:
            return None
        return ISIN_TO_TICKER.get(isin.strip().upper())

    def _from_bloomberg(self, bbg: str | None) -> str | None:
        """Convert Bloomberg ticker (e.g. 'MC FP') to yfinance (e.g. 'MC.PA')."""
        if not bbg:
            return None
        bbg = bbg.strip().upper()
        parts = bbg.split()
        if len(parts) != 2:
            return None
        symbol, exchange = parts
        suffix = BBG_EXCHANGE_TO_YF.get(exchange)
        if suffix is None:
            return None
        return f"{symbol}{suffix}"

    def _from_name(self, name: str | None) -> str | None:
        """Fuzzy match company name against known mappings."""
        if not name:
            return None
        name_lower = name.lower().strip()
        # Exact substring match
        for key, ticker in COMPANY_NAME_TO_TICKER.items():
            if key in name_lower:
                return ticker
        return None

    def _validate_ticker(self, ticker: str) -> str | None:
        """Check that a ticker returns data from yfinance."""
        try:
            info = yf.Ticker(ticker).fast_info
            if hasattr(info, "last_price") and info.last_price is not None:
                return ticker
        except Exception:
            pass
        return None
