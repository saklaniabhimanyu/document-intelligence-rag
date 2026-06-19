"""
CLI: ingest every PDF/DOCX/TXT in a directory and build a persisted FAISS index.

Usage:
    python -m scripts.build_index --data_dir data/corpus
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion import ingest_directory
from src.embeddings import build_index, save_index
from src.config import ingestion_config, embedding_config


def main():
    parser = argparse.ArgumentParser(description="Build the FAISS index from a document corpus.")
    parser.add_argument("--data_dir", default="data/corpus", help="Folder of PDF/DOCX/TXT files")
    parser.add_argument("--chunk_size", type=int, default=ingestion_config.chunk_size)
    parser.add_argument("--chunk_overlap", type=int, default=ingestion_config.chunk_overlap)
    args = parser.parse_args()

    ingestion_config.chunk_size = args.chunk_size
    ingestion_config.chunk_overlap = args.chunk_overlap

    print(f"Ingesting documents from {args.data_dir} ...")
    t0 = time.time()
    chunks = ingest_directory(args.data_dir, config=ingestion_config)
    print(f"  -> {len(chunks)} chunks from corpus (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})")
    print(f"  -> ingestion took {time.time() - t0:.1f}s")

    if not chunks:
        print("No chunks produced — check that --data_dir contains .pdf/.docx/.txt files.")
        return

    print(f"Embedding with {embedding_config.model_name} ...")
    t0 = time.time()
    index = build_index(chunks)
    print(f"  -> embedding + indexing took {time.time() - t0:.1f}s")

    path = save_index(index)
    print(f"Index persisted to: {path}")
    print("Page count summary:")
    pages_per_source = {}
    for c in chunks:
        pages_per_source.setdefault(c.source, set()).add(c.page)
    for source, pages in pages_per_source.items():
        print(f"  {source}: {len(pages)} pages, {sum(1 for c in chunks if c.source == source)} chunks")


if __name__ == "__main__":
    main()
