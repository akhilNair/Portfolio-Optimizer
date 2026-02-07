"""PDF table + text extraction pipeline for structured notes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from src.models.schemas import StructuredNote


class StructuredNotePDFParser:
    """
    Two-phase extraction:
      Phase A: pdfplumber extracts tables (asset baskets, terms)
      Phase B: PyMuPDF extracts full text, Claude interprets payoff structure
    Results are cached by content hash to avoid re-parsing.
    """

    def __init__(self, cache_dir: str = "data/parsed_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.0)

    def parse(self, pdf_path: str | Path) -> StructuredNote:
        pdf_path = Path(pdf_path)
        cache_key = self._cache_key(pdf_path)
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        tables = self._extract_tables(pdf_path)
        full_text = self._extract_text(pdf_path)
        note = self._llm_interpret(pdf_path.name, tables, full_text)

        self._save_cache(cache_key, note)
        return note

    def _extract_tables(self, pdf_path: Path) -> list[list[list[str]]]:
        """Extract all tables from all pages using pdfplumber."""
        all_tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
        return all_tables

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract full text using PyMuPDF."""
        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)

    def _llm_interpret(
        self,
        filename: str,
        tables: list[list[list[str]]],
        full_text: str,
    ) -> StructuredNote:
        """Use Claude to interpret extracted data into a StructuredNote."""
        tables_str = json.dumps(tables, indent=2)[:4000]
        text_excerpt = full_text[:6000]

        prompt = f"""Analyze this structured note PDF data and extract the following fields.
Return ONLY valid JSON matching the schema below, with no other text.

Fields to extract:
1. ISIN or unique identifier (note_id)
2. Issuer name
3. Issue date and maturity date (YYYY-MM-DD format)
4. Currency
5. Underlying asset basket: ticker symbols, names, weights, initial fixings
6. Payoff type: autocall, barrier_reverse_convertible, phoenix,
   capital_protected, vanilla_basket, or range_accrual
7. Coupon rate (annual, as decimal e.g. 0.08 for 8%)
8. Autocall trigger level (as fraction of initial e.g. 1.0 for 100%)
9. Barrier: direction (down_in/down_out/up_in/up_out), level (fraction), observation type
10. Capital protection percentage (as fraction)
11. Maturity in years
12. Observation frequency (monthly/quarterly/semi_annual)

Tables extracted from PDF:
{tables_str}

Text extracted from PDF:
{text_excerpt}

JSON schema:
{{
  "note_id": "string",
  "issuer": "string or null",
  "issue_date": "YYYY-MM-DD or null",
  "maturity_date": "YYYY-MM-DD or null",
  "currency": "USD",
  "underlying_basket": [
    {{"ticker": "AAPL", "name": "Apple Inc.", "weight_in_note": 0.25, "initial_fixing": 150.0}}
  ],
  "payoff": {{
    "payoff_type": "autocall",
    "coupon_rate_annual": 0.08,
    "autocall_trigger_pct": 1.0,
    "barrier": {{"direction": "down_in", "level_pct": 0.60, "observation": "european"}},
    "capital_protection_pct": null,
    "maturity_years": 2.0,
    "observation_frequency": "quarterly"
  }},
  "source_pdf": "{filename}"
}}"""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        # Extract JSON from response (handle potential markdown wrapping)
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        data = json.loads(content)
        data["source_pdf"] = filename
        data["raw_text_excerpt"] = text_excerpt[:2000]
        return StructuredNote(**data)

    def _cache_key(self, pdf_path: Path) -> str:
        content_hash = hashlib.md5(pdf_path.read_bytes()).hexdigest()
        return content_hash

    def _load_cache(self, key: str) -> Optional[StructuredNote]:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return StructuredNote(**json.loads(cache_file.read_text()))
        return None

    def _save_cache(self, key: str, note: StructuredNote) -> None:
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(note.model_dump_json(indent=2))
