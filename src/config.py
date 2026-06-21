"""
Central configuration for the Document Intelligence RAG system.
All tunables live here so experiments (chunk size, k, reranking on/off)
are one-line changes — this is what you sweep across in scripts/run_eval.py
to produce the baseline vs. optimised comparison in the README.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class IngestionConfig:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 50))
    separators: list = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])


@dataclass
class EmbeddingConfig:
    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    device: str = "cpu"
    index_dir: str = os.getenv("INDEX_DIR", "faiss_index")


@dataclass
class RetrievalConfig:
    top_k: int = int(os.getenv("TOP_K", 5))
    use_mmr: bool = os.getenv("USE_MMR", "true").lower() == "true"
    mmr_fetch_k: int = 20          # candidate pool MMR diversifies over
    mmr_lambda: float = 0.5        # 0 = max diversity, 1 = max relevance
    use_reranker: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_pool: int = 20  # retrieve this many, rerank down to top_k


@dataclass
class LLMConfig:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    model_name: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class PromptConfig:
    template: str = (
        "You must answer ONLY using the context provided below. "
        "Do not use any outside knowledge or invent information. "
        "If the context does not contain the answer, respond exactly with: "
        "\"I don't know based on the provided documents.\"\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (using ONLY the context above):"
    )


# Baseline config used in scripts/run_eval.py for the "before optimisation" RAGAS run
BASELINE_INGESTION = IngestionConfig(chunk_size=1000, chunk_overlap=0)
BASELINE_RETRIEVAL = RetrievalConfig(top_k=5, use_mmr=False, use_reranker=False)

ingestion_config = IngestionConfig()
embedding_config = EmbeddingConfig()
retrieval_config = RetrievalConfig()
llm_config = LLMConfig()
prompt_config = PromptConfig()
