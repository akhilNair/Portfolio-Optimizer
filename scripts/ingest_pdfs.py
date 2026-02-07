#!/usr/bin/env python3
"""CLI script to batch-parse structured note PDFs and build the FAISS index."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.pdf.indexer import NoteIndexer


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest structured note PDFs into FAISS index")
    parser.add_argument(
        "--pdf-dir",
        default="data/training_notes",
        help="Directory containing PDF files (default: data/training_notes)",
    )
    parser.add_argument(
        "--index-dir",
        default="data/embeddings",
        help="Output directory for FAISS index (default: data/embeddings)",
    )
    args = parser.parse_args()

    indexer = NoteIndexer(pdf_dir=args.pdf_dir, index_dir=args.index_dir)

    print(f"Scanning {args.pdf_dir} for PDF files...")
    indexer.build_index()
    print("Done.")


if __name__ == "__main__":
    main()
