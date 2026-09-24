"""
Production CLI for batch ingestion and indexing of multimodal documents.
Usage:
    python -m scripts.ingest_cli --pdf docs/NovaCore_Multimodal_Company_Report_2026.pdf
"""
import argparse
import sys
import time
from pathlib import Path

from app.chunking.hierarchical import HierarchicalChunker
from app.config.settings import get_settings
from app.parsing.pdf_parser import PDFParser
from app.retrieval.vector_store import PineconeHybridVectorStore
from app.storage.asset_store import get_asset_store


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a multimodal document (PDF) into the Pinecone Hybrid Vector Store."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF document to ingest.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Pinecone namespace to index into (overrides settings).",
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing vectors with this document's hash before upserting.",
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        print(f"Error: File not found at '{pdf_path}'")
        sys.exit(1)

    settings = get_settings()
    asset_store = get_asset_store()
    doc_hash = asset_store.compute_file_hash(pdf_path)

    print("=" * 70)
    print("NOVACORE MULTIMODAL INGESTION PIPELINE")
    print("=" * 70)
    print(f"Target Document:  {pdf_path.name}")
    print(f"Document SHA-256: {doc_hash}")
    print(f"Pinecone Index:   {settings.pinecone_index_name}")
    print(f"Target Namespace: {args.namespace or settings.pinecone_namespace}")
    print("-" * 70)

    start_time = time.perf_counter()

    # 1. Parsing
    print("[1/3] Parsing PDF layout, extracting tables, and rendering visuals...")
    pdf_parser = PDFParser(asset_store=asset_store)
    bundle = pdf_parser.parse(pdf_path)
    print(f"      Parsed {bundle.total_pages} pages successfully.")

    # 2. Chunking & Visual Summarization
    print("[2/3] Executing hierarchical parent-child chunking & VLM summarization...")
    chunker = HierarchicalChunker(settings=settings)
    child_chunks, parent_docs = chunker.chunk_bundle(bundle)
    print(f"      Generated {len(child_chunks)} child chunks and {len(parent_docs)} parent documents.")

    # 3. Vector Storage Upsert
    print("[3/3] Generating dense embeddings, BM25 sparse vectors, and indexing to Pinecone...")
    vector_store = PineconeHybridVectorStore(settings=settings)

    if args.delete_existing:
        print("      Deleting existing vectors matching this document hash...")
        vector_store.delete_document(doc_hash, namespace=args.namespace)

    upserted = vector_store.upsert_chunks(child_chunks, namespace=args.namespace)
    elapsed = round(time.perf_counter() - start_time, 2)

    print("=" * 70)
    print("INGESTION COMPLETE!")
    print(f"Total Vectors Indexed: {upserted}")
    print(f"Total Elapsed Time:    {elapsed}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
