"""
Component C — Retriever.

Three retrieval modes, selectable via RetrievalConfig:
  1. Plain similarity search (top-k)              -- the "baseline" in eval
  2. MMR (Maximal Marginal Relevance)              -- reduces redundant chunks
  3. Retrieve-then-rerank with a cross-encoder     -- the "optimised" path
"""
from __future__ import annotations

from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document

from src.config import RetrievalConfig, retrieval_config


_reranker_cache = {}


def get_reranker(model_name: str):
    """Lazily load + cache the cross-encoder reranker model."""
    if model_name not in _reranker_cache:
        from sentence_transformers import CrossEncoder

        _reranker_cache[model_name] = CrossEncoder(model_name)
    return _reranker_cache[model_name]


def similarity_retrieve(index: FAISS, query: str, k: int) -> List[Document]:
    return index.similarity_search(query, k=k)


def mmr_retrieve(index: FAISS, query: str, k: int, fetch_k: int, lambda_mult: float) -> List[Document]:
    return index.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
    )


def rerank(query: str, candidates: List[Document], top_k: int, model_name: str) -> List[Document]:
    """Cross-encoder reranking: score every (query, candidate) pair directly
    (more accurate than embedding cosine similarity, but slower — hence we
    only run it over a small candidate pool, not the whole corpus)."""
    if not candidates:
        return candidates
    model = get_reranker(model_name)
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def retrieve(
    index: FAISS,
    query: str,
    config: RetrievalConfig = retrieval_config,
) -> List[Document]:
    """Single entry point used by the QA chain and the eval harness — swaps
    behaviour purely based on the config object passed in, so baseline vs.
    optimised runs are just two different RetrievalConfig instances."""
    if config.use_reranker:
        pool = similarity_retrieve(index, query, k=config.rerank_candidate_pool)
        return rerank(query, pool, top_k=config.top_k, model_name=config.reranker_model)
    if config.use_mmr:
        return mmr_retrieve(
            index, query, k=config.top_k, fetch_k=config.mmr_fetch_k, lambda_mult=config.mmr_lambda
        )
    return similarity_retrieve(index, query, k=config.top_k)


class ConfigurableRetriever:
    """A thin LangChain-compatible retriever wrapper so it can be dropped
    straight into RetrievalQA.from_chain_type(retriever=...)."""

    def __init__(self, index: FAISS, config: RetrievalConfig = retrieval_config):
        self.index = index
        self.config = config

    def get_relevant_documents(self, query: str) -> List[Document]:
        return retrieve(self.index, query, self.config)

    # LangChain's newer Runnable-style retrievers call .invoke()
    def invoke(self, query: str, *args, **kwargs) -> List[Document]:
        return self.get_relevant_documents(query)
