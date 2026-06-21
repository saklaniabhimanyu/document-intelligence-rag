"""
Component B — Embedding Generation + Vector Store.

Embeds chunks with sentence-transformers/all-MiniLM-L6-v2 (384-dim, free,
runs on CPU) and stores them in a FAISS index alongside metadata
(source, page, chunk_id). The index is persisted to disk so re-running the
app doesn't re-embed everything from scratch.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document

from src.config import EmbeddingConfig, embedding_config
from src.ingestion import Chunk


def get_embedder(config: EmbeddingConfig = embedding_config):
    """Lazily import + construct the HF embedding model. Imported inside the
    function so modules that don't need embeddings (e.g. pure ingestion
    tests) don't pay the import cost / need the model downloaded."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.model_name,
        model_kwargs={"device": config.device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
    return [
        Document(
            page_content=c.text,
            metadata={"source": c.source, "page": c.page, "chunk_id": c.chunk_id},
        )
        for c in chunks
    ]


def build_index(
    chunks: List[Chunk],
    config: EmbeddingConfig = embedding_config,
    embedder=None,
) -> FAISS:
    """Embed all chunks and build a fresh FAISS index in memory."""
    if not chunks:
        raise ValueError("No chunks to index — did ingestion produce any output?")
    docs = chunks_to_documents(chunks)
    embedder = embedder or get_embedder(config)
    return FAISS.from_documents(docs, embedder)


def save_index(index: FAISS, config: EmbeddingConfig = embedding_config) -> str:
    Path(config.index_dir).mkdir(parents=True, exist_ok=True)
    index.save_local(config.index_dir)
    return config.index_dir


def load_index(config: EmbeddingConfig = embedding_config, embedder=None) -> Optional[FAISS]:
    index_path = Path(config.index_dir)
    if not (index_path / "index.faiss").exists():
        return None
    embedder = embedder or get_embedder(config)
    return FAISS.load_local(config.index_dir, embedder, allow_dangerous_deserialization=True)


def add_chunks_to_index(
    index: FAISS,
    chunks: List[Chunk],
    embedder=None,
) -> FAISS:
    """Incrementally add new chunks (e.g. a newly uploaded file) to an
    existing index without re-embedding everything."""
    docs = chunks_to_documents(chunks)
    index.add_documents(docs)
    return index


if __name__ == "__main__":
    from src.ingestion import ingest_file

    sample_chunks = ingest_file("data/sample.txt")
    print(f"Embedding {len(sample_chunks)} chunks with {embedding_config.model_name} ...")
    idx = build_index(sample_chunks)
    path = save_index(idx)
    print(f"Index saved to {path}")
