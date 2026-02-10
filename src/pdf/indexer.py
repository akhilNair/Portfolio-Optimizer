"""Build FAISS vector index from parsed structured notes."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.market.ticker_resolver import TickerResolver
from src.models.schemas import StructuredNote
from src.pdf.parser import StructuredNotePDFParser


class NoteIndexer:
    """Parse all PDFs in a directory and build a FAISS vector index."""

    def __init__(
        self,
        pdf_dir: str = "data/training_notes",
        index_dir: str = "data/embeddings",
        cache_dir: str = "data/parsed_cache",
    ):
        self.pdf_dir = Path(pdf_dir)
        self.index_dir = Path(index_dir)
        self.cache_dir = Path(cache_dir)
        self.parser = StructuredNotePDFParser()
        self.resolver = TickerResolver()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def build_index(self) -> FAISS:
        """Parse all PDFs and create a FAISS index."""
        documents = []
        metadatas = []

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {self.pdf_dir}. "
                "Place structured note PDFs in data/training_notes/"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        for pdf_file in pdf_files:
            try:
                note = self.parser.parse(pdf_file)

                # Save parsed note as JSON for retriever lookups
                cache_path = self.cache_dir / f"{note.note_id}.json"
                cache_path.write_text(note.model_dump_json(indent=2))

                doc_text = self._note_to_document(note)
                documents.append(doc_text)

                # Resolve each asset to best available identifier
                identifiers = []
                for a in note.underlying_basket:
                    resolved = self.resolver.resolve(a)
                    if resolved:
                        identifiers.append(resolved)
                    elif a.isin:
                        identifiers.append(a.isin)
                    elif a.ticker_bloomberg:
                        identifiers.append(a.ticker_bloomberg)
                    else:
                        identifiers.append(a.name)

                metadatas.append(
                    {
                        "note_id": note.note_id,
                        "source_pdf": note.source_pdf,
                        "tickers": ",".join(identifiers),
                        "payoff_type": note.payoff.payoff_type.value,
                        "coupon_rate": note.payoff.coupon_rate_annual or 0.0,
                        "maturity_years": note.payoff.maturity_years or 0.0,
                    }
                )
            except Exception as e:
                print(f"Failed to parse {pdf_file.name}: {e}")
                continue

        if not documents:
            raise ValueError("No PDFs were successfully parsed.")

        vectorstore = FAISS.from_texts(
            texts=documents,
            embedding=self.embeddings,
            metadatas=metadatas,
        )
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.index_dir))
        print(f"Index built with {len(documents)} documents, saved to {self.index_dir}")
        return vectorstore

    def _note_to_document(self, note: StructuredNote) -> str:
        """Convert a StructuredNote into a flat text document for embedding."""
        asset_parts = []
        for a in note.underlying_basket:
            desc = a.name
            if a.ticker:
                desc += f" ({a.ticker})"
            elif a.ticker_bloomberg:
                desc += f" ({a.ticker_bloomberg})"
            elif a.isin:
                desc += f" ({a.isin})"
            asset_parts.append(desc)

        basket_str = ", ".join(asset_parts)
        barrier_info = "none"
        if note.payoff.barrier:
            barrier_info = (
                f"{note.payoff.barrier.direction.value} at {note.payoff.barrier.level_pct}"
            )

        return (
            f"Structured note {note.note_id} issued by {note.issuer}. "
            f"Payoff type: {note.payoff.payoff_type.value}. "
            f"Underlying basket: {basket_str}. "
            f"Coupon: {note.payoff.coupon_rate_annual or 'N/A'}. "
            f"Maturity: {note.payoff.maturity_years or 'N/A'} years. "
            f"Barrier: {barrier_info}. "
            f"Currency: {note.currency}."
        )
