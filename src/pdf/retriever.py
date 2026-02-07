"""Search indexed structured notes by analyst criteria."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.models.schemas import PayoffType, StructuredNote


class NoteRetriever:
    """Semantic + metadata search over the parsed structured notes index."""

    def __init__(
        self,
        index_dir: str = "data/embeddings",
        cache_dir: str = "data/parsed_cache",
    ):
        self.index_dir = Path(index_dir)
        self.cache_dir = Path(cache_dir)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self._vectorstore: Optional[FAISS] = None

    @property
    def vectorstore(self) -> FAISS:
        if self._vectorstore is None:
            if not self.index_dir.exists():
                raise FileNotFoundError(
                    f"No index found at {self.index_dir}. "
                    "Run scripts/ingest_pdfs.py first."
                )
            self._vectorstore = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return self._vectorstore

    def search(
        self,
        payoff_type: Optional[PayoffType] = None,
        tickers: Optional[list[str]] = None,
        sectors: Optional[list[str]] = None,
        min_coupon: Optional[float] = None,
        max_maturity: Optional[float] = None,
        top_k: int = 10,
    ) -> list[StructuredNote]:
        """
        Two-stage retrieval:
         1. Semantic search with a synthesized query
         2. Post-filter on metadata constraints
        """
        query_parts = []
        if payoff_type:
            query_parts.append(f"{payoff_type.value} structured note")
        if tickers:
            query_parts.append(f"underlying assets {', '.join(tickers)}")
        if sectors:
            query_parts.append(f"sectors {', '.join(sectors)}")
        if min_coupon:
            query_parts.append(f"coupon at least {min_coupon:.1%}")

        query = " ".join(query_parts) or "structured note"

        # Fetch 3x top_k for post-filtering headroom
        results = self.vectorstore.similarity_search_with_score(query, k=top_k * 3)

        filtered_notes = []
        for doc, _score in results:
            meta = doc.metadata

            if payoff_type and meta.get("payoff_type") != payoff_type.value:
                continue
            if min_coupon and meta.get("coupon_rate", 0.0) < min_coupon:
                continue
            if max_maturity and meta.get("maturity_years", 0.0) > max_maturity:
                continue

            note = self._load_note(meta["note_id"])
            if note:
                filtered_notes.append(note)

            if len(filtered_notes) >= top_k:
                break

        return filtered_notes

    def _load_note(self, note_id: str) -> Optional[StructuredNote]:
        """Load StructuredNote from the JSON cache by scanning cache files."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text())
                if data.get("note_id") == note_id:
                    return StructuredNote(**data)
            except Exception:
                continue
        return None
