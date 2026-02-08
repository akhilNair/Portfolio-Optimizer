"""PDF extraction pipeline for structured notes using deterministic data engineering.

No LLM calls — uses regex, table heuristics, and keyword mapping to extract
structured data from PDFs. Zero API cost per parse.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Optional

import  pymupdf
import pdfplumber

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from src.models.schemas import (
    AssetInBasket,
    BarrierCondition,
    BarrierDirection,
    PayoffStructure,
    PayoffType,
    StructuredNote,
)

# ── Ticker recognition ───────────────────────────────────────────────
# Common tickers found in structured notes. Extend as needed.
KNOWN_TICKERS: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "GOOG": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corp.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart Inc.",
    "PG": "Procter & Gamble Co.",
    "MA": "Mastercard Inc.",
    "UNH": "UnitedHealth Group Inc.",
    "HD": "Home Depot Inc.",
    "DIS": "Walt Disney Co.",
    "BAC": "Bank of America Corp.",
    "XOM": "Exxon Mobil Corp.",
    "PFE": "Pfizer Inc.",
    "KO": "Coca-Cola Co.",
    "PEP": "PepsiCo Inc.",
    "CSCO": "Cisco Systems Inc.",
    "NFLX": "Netflix Inc.",
    "INTC": "Intel Corp.",
    "AMD": "Advanced Micro Devices Inc.",
    "CRM": "Salesforce Inc.",
    "ADBE": "Adobe Inc.",
    "COST": "Costco Wholesale Corp.",
    "AVGO": "Broadcom Inc.",
    "TXN": "Texas Instruments Inc.",
    "QCOM": "Qualcomm Inc.",
    "GS": "Goldman Sachs Group Inc.",
    "MS": "Morgan Stanley",
    "BLK": "BlackRock Inc.",
    "SCHW": "Charles Schwab Corp.",
    "AXP": "American Express Co.",
    "BA": "Boeing Co.",
    "CAT": "Caterpillar Inc.",
    "IBM": "IBM Corp.",
    "GE": "General Electric Co.",
    "NKE": "Nike Inc.",
    "MCD": "McDonald's Corp.",
    "SBUX": "Starbucks Corp.",
    "ABT": "Abbott Laboratories",
    "LLY": "Eli Lilly & Co.",
    "MRK": "Merck & Co.",
    "BMY": "Bristol-Myers Squibb Co.",
    "ABBV": "AbbVie Inc.",
    "VZ": "Verizon Communications Inc.",
    "CVX": "Chevron Corp.",
    "COP": "ConocoPhillips",
    "SLB": "Schlumberger Ltd.",
    "NESN": "Nestle SA",
    "NOVN": "Novartis AG",
    "ROG": "Roche Holding AG",
    "SIE": "Siemens AG",
    "SAP": "SAP SE",
    "ASML": "ASML Holding NV",
    "BNP": "BNP Paribas SA",
    "SAN": "Banco Santander SA",
    "VOW3": "Volkswagen AG",
    "SPX": "S&P 500 Index",
    "NDX": "Nasdaq-100 Index",
    "SX5E": "Euro Stoxx 50",
    "UKX": "FTSE 100",
    "NKY": "Nikkei 225",
}

# Words that look like tickers but aren't
FALSE_POSITIVE_TICKERS = frozenset({
    "THE", "FOR", "AND", "NOT", "ARE", "WAS", "HAS", "HAD", "HIS", "HER",
    "ITS", "OUR", "BUT", "ALL", "CAN", "MAY", "LTD", "INC", "LLC", "PLC",
    "USD", "EUR", "GBP", "CHF", "JPY", "PDF", "CEO", "CFO", "CIO", "COO",
    "ETF", "NAV", "IPO", "GDP", "CPI", "YTD", "MTD", "QTD", "BPS", "DAY",
    "FEB", "MAR", "APR", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    "JAN", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN", "EST", "PST",
})

# ── Regex patterns ───────────────────────────────────────────────────

ISIN_RE = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}\d)\b")
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
CURRENCY_RE = re.compile(r"\b(USD|EUR|GBP|CHF|JPY|AUD|CAD|HKD|SGD)\b")
TICKER_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,5})(?![a-zA-Z])")

#ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")

BBG_TICKER_RE = re.compile(
    r"\b([A-Z]{1,5})\s+(FP|UW|US|LN|GY|PA|SW|IM)\b"
)

COMPANY_NAME_RE = re.compile(
    r"Underlying\s*[:\-]?\s*(.+)", re.IGNORECASE
)
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Payoff type keywords (most specific first)
PAYOFF_KEYWORDS: list[tuple[list[str], PayoffType]] = [
    (["autocall", "auto-call", "auto call"], PayoffType.AUTOCALL),
    (["phoenix"], PayoffType.PHOENIX),
    (["barrier reverse convertible", "brc"], PayoffType.BARRIER_REVERSE_CONVERTIBLE),
    (["reverse convertible"], PayoffType.BARRIER_REVERSE_CONVERTIBLE),
    (["capital protect", "capital guaranteed", "principal protect"], PayoffType.CAPITAL_PROTECTED),
    (["range accrual"], PayoffType.RANGE_ACCRUAL),
    (["basket", "worst-of", "worst of", "best-of", "best of"], PayoffType.VANILLA_BASKET),
]

BARRIER_KEYWORDS: list[tuple[list[str], BarrierDirection]] = [
    (["down-and-in", "down and in", "knock-in", "knock in"], BarrierDirection.DOWN_IN),
    (["down-and-out", "down and out"], BarrierDirection.DOWN_OUT),
    (["up-and-in", "up and in"], BarrierDirection.UP_IN),
    (["up-and-out", "up and out"], BarrierDirection.UP_OUT),
]

OBSERVATION_TYPE_KEYWORDS: dict[str, str] = {
    "european": "european",
    "at maturity": "european",
    "final observation": "european",
    "american": "american",
    "continuous": "american",
    "daily observation": "daily",
    "daily monitoring": "daily",
}

FREQ_KEYWORDS: dict[str, str] = {
    "monthly": "monthly",
    "quarterly": "quarterly",
    "semi-annual": "semi_annual",
    "semi annual": "semi_annual",
    "semiannual": "semi_annual",
    "annual": "annual",
    "daily": "daily",
    "weekly": "weekly",
}

# Issuer names to scan for (case-insensitive)
KNOWN_ISSUERS = [
    "Goldman Sachs", "Morgan Stanley", "J.P. Morgan", "JP Morgan", "JPMorgan",
    "Barclays", "Credit Suisse", "UBS", "Citigroup", "Citi",
    "Deutsche Bank", "BNP Paribas", "Societe Generale", "HSBC", "Nomura",
    "Bank of America", "Wells Fargo", "RBC", "Scotiabank", "TD Securities",
    "BMO", "CIBC", "Natixis", "Commerzbank", "Leonteq", "Vontobel",
    "EFG", "Julius Baer", "Zurcher Kantonalbank", "ZKB", "Raiffeisen",
]


class StructuredNotePDFParser:
    """
    Deterministic PDF parser for structured notes.

    Pipeline:
      1. pdfplumber → tables (asset baskets, term sheet key-value rows)
      2. PyMuPDF → full text
      3. Regex + keyword matching → field extraction
      4. Table column-header heuristics → basket and term sheet values

    No LLM calls. Zero API cost per document.
    """

    def __init__(self, cache_dir: str = "data/parsed_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, pdf_path: str | Path) -> StructuredNote:
        pdf_path = Path(pdf_path)
        cache_key = self._cache_key(pdf_path)
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        tables = self._extract_tables(pdf_path)
        full_text = self._extract_text(pdf_path)
        text_lower = full_text.lower()

        # Extract all fields deterministically
        note_id = self._extract_note_id(full_text) or pdf_path.stem.replace(" ", "_")
        issuer = self._extract_issuer(full_text)
        issue_date, maturity_date = self._extract_dates(full_text, text_lower)
        currency = self._extract_currency(full_text)
        basket = self._extract_basket_from_tables(tables, full_text)
        print('Basket extracted 1:', basket)
        if not basket:
            basket = self._extract_basket_from_text(full_text)
        print('Basket extracted 2:', basket)
        # Also try extracting term-sheet values from key-value tables
        kv_data = self._extract_key_value_from_tables(tables)
        payoff = self._extract_payoff(text_lower, full_text, kv_data)

        note = StructuredNote(
            note_id=note_id,
            issuer=issuer,
            issue_date=issue_date,
            maturity_date=maturity_date,
            currency=currency,
            underlying_basket=basket,
            payoff=payoff,
            source_pdf=pdf_path.name,
            raw_text_excerpt=full_text[:2000],
        )

        self._save_cache(cache_key, note)
        return note

    # ── Raw extraction ───────────────────────────────────────────────

    def _extract_tables(self, pdf_path: Path) -> list[list[list[str]]]:
        all_tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    all_tables.extend(page_tables)
        return all_tables

    def _extract_text(self, pdf_path: Path) -> str:
        doc = pymupdf.open(str(pdf_path))
        parts = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(parts)

    # ── Key-value table extraction ───────────────────────────────────

    def _extract_key_value_from_tables(
        self, tables: list[list[list[str]]]
    ) -> dict[str, str]:
        """
        Structured note term sheets are often 2-column tables:
        | Field Label | Value |
        Scan all tables for this pattern and merge into a flat dict.
        """
        kv: dict[str, str] = {}
        for table in tables:
            if not table:
                continue
            for row in table:
                if not row or len(row) < 2:
                    continue
                key = (row[0] or "").strip().lower()
                val = (row[1] or "").strip()
                if key and val and len(key) > 2:
                    kv[key] = val
        return kv

    # ── Field extractors ─────────────────────────────────────────────

    def _extract_note_id(self, text: str) -> Optional[str]:
        # ISIN (most common)
        m = ISIN_RE.search(text)
        if m:
            return m.group(1)

        # Explicit labels
        for pattern in [
            r"(?:ISIN|Securities?\s*Code|Valor)\s*[:\s]+([A-Z0-9]{6,12})",
            r"(?:Series|Tranche|Note)\s*(?:No\.?|Number)\s*[:\s]+([A-Z0-9\-]{4,20})",
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    def _extract_issuer(self, text: str) -> Optional[str]:
        # Try explicit labels first
        m = re.search(
            r"(?:Issuer|Issued\s+by|Issuing\s+(?:Bank|Entity))\s*[:\s]+(.+?)(?:\n|\.|\s{2,})",
            text,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().rstrip(".,")

        # Scan for known issuer names
        text_upper = text[:3000]  # Issuers typically appear early
        for issuer in KNOWN_ISSUERS:
            if issuer.lower() in text_upper.lower():
                return issuer

        return None

    def _extract_dates(
        self, text: str, text_lower: str
    ) -> tuple[Optional[date], Optional[date]]:
        issue_date = None
        maturity_date = None

        issue_keywords = [
            "issue date", "trade date", "initial fixing", "strike date", "pricing date",
        ]
        maturity_keywords = [
            "maturity date", "final fixing", "expiry date", "redemption date",
            "final valuation",
        ]

        for label, keywords in [("issue", issue_keywords), ("maturity", maturity_keywords)]:
            for kw in keywords:
                idx = text_lower.find(kw)
                if idx == -1:
                    continue
                region = text[idx : idx + 200]
                found = self._find_dates_in_text(region)
                if found:
                    if label == "issue":
                        issue_date = found[0]
                    else:
                        maturity_date = found[0]
                    break

        # Fallback: use earliest and latest dates in the document
        if not issue_date and not maturity_date:
            all_dates = self._find_dates_in_text(text)
            if len(all_dates) >= 2:
                sorted_dates = sorted(all_dates)
                issue_date = sorted_dates[0]
                maturity_date = sorted_dates[-1]

        return issue_date, maturity_date

    def _find_dates_in_text(self, text: str) -> list[date]:
        dates = []

        # YYYY-MM-DD
        for m in re.finditer(r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b", text):
            try:
                dates.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                pass

        # DD/MM/YYYY (European) or MM/DD/YYYY
        for m in re.finditer(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b", text):
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if mo > 12:
                d, mo = mo, d
            try:
                dates.append(date(y, mo, d))
            except ValueError:
                pass

        # "12 January 2024"
        for m in re.finditer(
            r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+(\d{4})\b",
            text, re.IGNORECASE,
        ):
            try:
                dates.append(date(int(m.group(3)), MONTH_MAP[m.group(2).lower()], int(m.group(1))))
            except (ValueError, KeyError):
                pass

        # "January 12, 2024"
        for m in re.finditer(
            r"\b(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
            text, re.IGNORECASE,
        ):
            try:
                dates.append(date(int(m.group(3)), MONTH_MAP[m.group(1).lower()], int(m.group(2))))
            except (ValueError, KeyError):
                pass

        return dates

    def _extract_currency(self, text: str) -> str:
        m = CURRENCY_RE.search(text)
        return m.group(1) if m else "USD"

    # ── Basket extraction ────────────────────────────────────────────

    def _extract_basket_from_tables(
        self, tables: list[list[list[str]]], full_text: str
    ) -> list[AssetInBasket]:
        """Extract underlying basket from tables using column-header heuristics."""
        for table in tables:
            if not table or len(table) < 2:
                continue
            print('Table : ', table)
            col_map = self._map_table_columns(table[0])
            if not col_map:
                # Try second row as header (some PDFs have a title row first)
                if len(table) >= 3:
                    col_map = self._map_table_columns(table[1])
                    if col_map:
                        table = table[1:]  # shift header
                if not col_map:
                    continue

            assets = self._parse_basket_rows(table[1:], col_map)
            if assets:
                return assets

        return []

    def _map_table_columns(self, header_row: list[str | None]) -> dict[str, int]:
        """
        Map header cells to semantic roles.
        Explicitly separates identifiers, absolute prices, and relative levels.
        """
        if not header_row:
            return {}

        col_map: dict[str, int] = {}

        for i, cell in enumerate(header_row):
            if not cell:
                continue

            cl = cell.strip().lower()

            # --- Identity / identifiers ---
            if any(kw in cl for kw in ["isin", "valor"]):
                col_map.setdefault("isin", i)

            elif any(kw in cl for kw in ["bloomberg", "ticker", "symbol"]):
                col_map.setdefault("ticker_bloomberg", i)

            elif "ric" in cl:
                col_map.setdefault("ric", i)

            elif any(kw in cl for kw in ["name", "underlying", "company", "stock", "share"]):
                col_map.setdefault("name", i)

            # --- Absolute price levels ---
            elif any(kw in cl for kw in ["reference level", "reference price", "spot", "initial price"]):
                col_map.setdefault("absolute_fixing", i)

            # --- Relative levels (percent-based) ---
            elif any(kw in cl for kw in ["strike"]):
                col_map.setdefault("strike_pct", i)

            elif any(kw in cl for kw in ["kick in", "knock in", "barrier"]):
                col_map.setdefault("barrier_pct", i)

            # --- Basket metadata ---
            elif any(kw in cl for kw in ["weight", "allocation", "proportion"]):
                col_map.setdefault("weight", i)

        # Must have at least identity info to be useful
        if not any(k in col_map for k in ["name", "isin", "ticker_bloomberg"]):
            return {}

        return col_map

    @staticmethod
    def _parse_underlying_cell(cell_text: str) -> dict:
        """Parse a rich underlying cell like:
        LVMH Moët Hennessy Louis Vuitton SE
        Bloomberg: MC FP / ISIN: FR0000121014 / Valor: 507170 / RIC: LVMH.PA
        """
        result: dict = {}
        lines = [l.strip() for l in cell_text.replace("/", "\n").split("\n") if l.strip()]

        # First line (before any key-value pairs) is the company name
        for line in lines:
            if not re.match(r"^(Bloomberg|ISIN|Valor|RIC)\s*:", line, re.IGNORECASE):
                if "name" not in result:
                    result["name"] = line
                continue

            kv_match = re.match(r"^(Bloomberg|ISIN|Valor|RIC)\s*:\s*(.+)", line, re.IGNORECASE)
            if kv_match:
                key = kv_match.group(1).lower()
                val = kv_match.group(2).strip()
                if key == "bloomberg":
                    result["ticker_bloomberg"] = val
                elif key == "isin":
                    result["isin"] = val

        return result

    @staticmethod
    def _parse_fixing_cell(cell_text: str) -> Optional[float]:
        """Extract numeric value from a reference level cell like 'EUR 631.7 (indicative)'."""
        m = re.search(r"(\d[\d,]*\.?\d*)", cell_text.replace(",", ""))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    def _parse_basket_rows(
    self, rows: list[list[str | None]], col_map: dict[str, int]
) -> list[AssetInBasket]:
        assets: list[AssetInBasket] = []

        for row in rows:
            if not row or all(not cell for cell in row):
                continue

            # --- Extract raw cells ---
            raw_name_cell = self._get_cell(row, col_map.get("name")) or ""
            raw_isin_cell = self._get_cell(row, col_map.get("isin"))
            raw_bbg_cell = self._get_cell(row, col_map.get("ticker_bloomberg"))
            raw_weight_cell = self._get_cell(row, col_map.get("weight"))
            raw_fixing_cell = self._get_cell(row, col_map.get("absolute_fixing"))

            name: str | None = None
            isin: str | None = None
            ticker_bloomberg: str | None = None

            # --- Case 1: rich underlying cell (UBS-style) ---
            is_rich = bool(
                re.search(r"(ISIN|Bloomberg|RIC)\s*:", raw_name_cell, re.IGNORECASE)
            )

            if is_rich:
                parsed = self._parse_underlying_cell(raw_name_cell)
                name = parsed.get("name")
                isin = parsed.get("isin")
                ticker_bloomberg = parsed.get("ticker_bloomberg")

            else:
                # --- Case 2: structured columns ---
                name = raw_name_cell.strip() or None
                isin = raw_isin_cell.strip() if raw_isin_cell else None
                ticker_bloomberg = raw_bbg_cell.strip() if raw_bbg_cell else None

            # --- Identity sanity check ---
            if not any([name, isin, ticker_bloomberg]):
                continue  # cannot identify asset reliably

            # --- Weight ---
            weight = self._parse_number(raw_weight_cell)
            if weight is not None and weight > 1:
                weight = weight / 100.0

            # --- Absolute initial fixing (price, not %) ---
            fixing = self._parse_fixing_cell(raw_fixing_cell) if raw_fixing_cell else None

            # --- Normalize name ---
            if not name:
                if isin:
                    name = isin
                elif ticker_bloomberg:
                    name = ticker_bloomberg

            assets.append(
                AssetInBasket(
                    name=name,
                    isin=isin,
                    ticker=None,  # never fabricate ticker
                    ticker_bloomberg=ticker_bloomberg,
                    weight_in_note=weight,
                    initial_fixing=fixing,
                )
            )

        return assets

    
    def _extract_underlying_section(self, text: str) -> str:
        markers = ["Information on Underlying", "Underlying", "Reference Level"]
        for m in markers:
            if m in text:
                start = text.index(m)
                return text[start:start + 1500]  # small window
        return ""

    def _extract_basket_from_text(self, text: str) -> list[AssetInBasket]:
        section = self._extract_underlying_section(text)

        assets = {}

        # 1. ISIN-based extraction (strongest)
        for isin in ISIN_RE.findall(section):
            assets[isin] = AssetInBasket(
                ticker=isin,
                name=isin,
                isin=isin,
            )

        # 2. Bloomberg ticker extraction
        for m in BBG_TICKER_RE.finditer(section):
            bbg = f"{m.group(1)} {m.group(2)}"
            assets[bbg] = AssetInBasket(
                ticker=m.group(1),
                name=m.group(1),
                ticker_bloomberg=bbg,
            )

        # 3. Known tickers fallback (last resort)
        for m in TICKER_RE.finditer(section):
            t = m.group(1)
            if (
                t in KNOWN_TICKERS
                and t not in FALSE_POSITIVE_TICKERS
                and t not in {"EUR", "AG", "SE", "SA", "CET"}
            ):
                assets[t] = AssetInBasket(
                    ticker=t,
                    name=KNOWN_TICKERS[t],
                )

        return list(assets.values())

    def _resolve_ticker(self, ticker: str | None, name: str | None) -> Optional[str]:
        """Validate a ticker string, or look it up from the company name."""
        if ticker:
            cleaned = re.sub(r"[^A-Z0-9]", "", ticker.upper())
            if cleaned in KNOWN_TICKERS:
                return cleaned
            # Might be a raw uppercase cell — try extracting ticker from it
            m = TICKER_RE.search(ticker.upper())
            if m and m.group(1) in KNOWN_TICKERS:
                return m.group(1)

        if name:
            name_lower = name.lower()
            for t, n in KNOWN_TICKERS.items():
                if n.lower() in name_lower or name_lower in n.lower():
                    return t

        return None

    # ── Payoff extraction ────────────────────────────────────────────

    def _extract_payoff(
        self, text_lower: str, text: str, kv_data: dict[str, str]
    ) -> PayoffStructure:
        payoff_type = self._detect_payoff_type(text_lower)
        coupon = self._extract_coupon(text, text_lower, kv_data)
        autocall_trigger = self._extract_autocall_trigger(text_lower, kv_data)
        barrier = self._extract_barrier(text_lower, kv_data)
        capital_protection = self._extract_capital_protection(text_lower, kv_data)
        maturity_years = self._extract_maturity_years(text_lower, kv_data)
        obs_frequency = self._extract_observation_frequency(text_lower, kv_data)

        return PayoffStructure(
            payoff_type=payoff_type,
            coupon_rate_annual=coupon,
            autocall_trigger_pct=autocall_trigger,
            barrier=barrier,
            capital_protection_pct=capital_protection,
            maturity_years=maturity_years,
            observation_frequency=obs_frequency,
        )

    def _detect_payoff_type(self, text_lower: str) -> PayoffType:
        for keywords, ptype in PAYOFF_KEYWORDS:
            for kw in keywords:
                if kw in text_lower:
                    return ptype
        return PayoffType.VANILLA_BASKET

    def _extract_coupon(
        self, text: str, text_lower: str, kv_data: dict[str, str]
    ) -> Optional[float]:
        # Try key-value table data first
        for key, val in kv_data.items():
            if any(kw in key for kw in ["coupon", "interest rate", "yield", "premium"]):
                pcts = PCT_RE.findall(val)
                for p in pcts:
                    v = float(p)
                    if 0.5 <= v <= 30:
                        return v / 100.0

        # Contextual text search
        for keyword in ["coupon", "interest rate", "yield", "premium", "p.a."]:
            idx = text_lower.find(keyword)
            if idx == -1:
                continue
            region = text[max(0, idx - 50) : idx + 100]
            for p in PCT_RE.findall(region):
                v = float(p)
                if 0.5 <= v <= 30:
                    return v / 100.0
        return None

    def _extract_autocall_trigger(
        self, text_lower: str, kv_data: dict[str, str]
    ) -> Optional[float]:
        # Key-value table
        for key, val in kv_data.items():
            if any(kw in key for kw in ["autocall", "auto-call", "call trigger", "early redemption"]):
                for p in PCT_RE.findall(val):
                    v = float(p)
                    if 50 <= v <= 150:
                        return v / 100.0

        # Text search
        for keyword in ["autocall trigger", "auto-call trigger", "call trigger",
                        "early redemption trigger", "autocall level"]:
            idx = text_lower.find(keyword)
            if idx == -1:
                continue
            region = text_lower[idx : idx + 100]
            for p in PCT_RE.findall(region):
                v = float(p)
                if 50 <= v <= 150:
                    return v / 100.0
        return None

    def _extract_barrier(
        self, text_lower: str, kv_data: dict[str, str]
    ) -> Optional[BarrierCondition]:
        # Detect direction
        direction = None
        for keywords, bdir in BARRIER_KEYWORDS:
            for kw in keywords:
                if kw in text_lower:
                    direction = bdir
                    break
            if direction:
                break

        if not direction and "barrier" in text_lower:
            direction = BarrierDirection.DOWN_IN

        if not direction:
            return None

        # Extract level from KV table
        level = None
        for key, val in kv_data.items():
            if "barrier" in key:
                for p in PCT_RE.findall(val):
                    v = float(p)
                    if 20 <= v <= 95:
                        level = v / 100.0
                        break
                if level:
                    break

        # Extract level from text
        if not level:
            for keyword in ["barrier level", "barrier at", "knock-in level",
                            "barrier", "protection barrier"]:
                idx = text_lower.find(keyword)
                if idx == -1:
                    continue
                region = text_lower[idx : idx + 100]
                for p in PCT_RE.findall(region):
                    v = float(p)
                    if 20 <= v <= 95:
                        level = v / 100.0
                        break
                if level:
                    break

        if not level:
            level = 0.60

        # Observation type
        observation = "european"
        for kw, obs in OBSERVATION_TYPE_KEYWORDS.items():
            if kw in text_lower:
                observation = obs
                break

        return BarrierCondition(direction=direction, level_pct=level, observation=observation)

    def _extract_capital_protection(
        self, text_lower: str, kv_data: dict[str, str]
    ) -> Optional[float]:
        for key, val in kv_data.items():
            if any(kw in key for kw in ["capital protection", "principal protection"]):
                for p in PCT_RE.findall(val):
                    v = float(p)
                    if 50 <= v <= 100:
                        return v / 100.0

        for keyword in ["capital protection", "principal protection", "capital guarantee"]:
            idx = text_lower.find(keyword)
            if idx == -1:
                continue
            region = text_lower[idx : idx + 100]
            for p in PCT_RE.findall(region):
                v = float(p)
                if 50 <= v <= 100:
                    return v / 100.0
        return None

    def _extract_maturity_years(
        self, text_lower: str, kv_data: dict[str, str]
    ) -> Optional[float]:
        # Key-value table
        for key, val in kv_data.items():
            if any(kw in key for kw in ["maturity", "tenor", "term"]):
                # Try years
                m = re.search(r"(\d+(?:\.\d+)?)\s*(?:year|yr)", val.lower())
                if m:
                    return float(m.group(1))
                # Try months
                m = re.search(r"(\d+)\s*month", val.lower())
                if m:
                    return round(float(m.group(1)) / 12.0, 2)

        # Text search
        for pattern in [
            r"(\d+(?:\.\d+)?)\s*(?:year|yr)s?\s*(?:maturity|tenor|term)",
            r"(?:maturity|tenor|term)\s*[:\s]*(\d+(?:\.\d+)?)\s*(?:year|yr)",
            r"(\d+)\s*-?\s*(?:year|yr)\s*(?:note|bond|product)",
        ]:
            m = re.search(pattern, text_lower)
            if m:
                return float(m.group(1))

        for pattern in [
            r"(\d+)\s*months?\s*(?:maturity|tenor|term)",
            r"(?:maturity|tenor|term)\s*[:\s]*(\d+)\s*months?",
        ]:
            m = re.search(pattern, text_lower)
            if m:
                return round(float(m.group(1)) / 12.0, 2)

        return None

    def _extract_observation_frequency(
        self, text_lower: str, kv_data: dict[str, str]
    ) -> Optional[str]:
        # Key-value table
        for key, val in kv_data.items():
            if any(kw in key for kw in ["observation", "coupon frequency", "frequency"]):
                val_lower = val.lower()
                for keyword, freq in FREQ_KEYWORDS.items():
                    if keyword in val_lower:
                        return freq

        # Text contextual
        for keyword, freq in FREQ_KEYWORDS.items():
            if re.search(rf"{keyword}\s*(?:observation|coupon|review|period)", text_lower):
                return freq

        # Simple presence
        for keyword, freq in FREQ_KEYWORDS.items():
            if keyword in text_lower:
                return freq

        return None

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def _get_cell(row: list[str | None], col_idx: int | None) -> str | None:
        if col_idx is None or col_idx >= len(row):
            return None
        cell = row[col_idx]
        return cell.strip().replace("\n", " ") if cell else None

    @staticmethod
    def _parse_number(text: str | None) -> Optional[float]:
        if not text:
            return None
        cleaned = re.sub(r"[^\d.\-]", "", text.strip())
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _cache_key(self, pdf_path: Path) -> str:
        return hashlib.md5(pdf_path.read_bytes()).hexdigest()

    def _load_cache(self, key: str) -> Optional[StructuredNote]:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return StructuredNote(**json.loads(cache_file.read_text()))
        return None

    def _save_cache(self, key: str, note: StructuredNote) -> None:
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(note.model_dump_json(indent=2))
