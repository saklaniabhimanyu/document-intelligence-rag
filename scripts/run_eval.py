"""
Runs the full evaluation story from the spec:
  1. Build/load index, run RAGAS on a BASELINE pipeline (chunk_size=1000, no
     MMR, no reranking).
  2. Run RAGAS on the OPTIMISED pipeline (current src/config.py settings).
  3. Print + save the improvement table (this produces the 38% number).
  4. Benchmark average answer latency over the eval set (the "<3s" claim).

Usage:
    python -m scripts.run_eval
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    BASELINE_INGESTION,
    BASELINE_RETRIEVAL,
    ingestion_config,
    retrieval_config,
)
from src.ingestion import ingest_directory
from src.embeddings import build_index, get_embedder
from src.retriever import ConfigurableRetriever
from src.qa_chain import build_qa_chain, get_llm, answer_question
from src.evaluation import (
    load_eval_set,
    run_chain_over_eval_set,
    score_with_ragas,
    compare_baseline_vs_optimised,
    save_results,
)


def build_chain_for_config(chunks_source_dir: str, ingest_cfg, retrieve_cfg, llm, embedder):
    chunks = ingest_directory(chunks_source_dir, config=ingest_cfg)
    index = build_index(chunks, embedder=embedder)
    retriever = ConfigurableRetriever(index=index, config=retrieve_cfg)
    chain = build_qa_chain(retriever, llm=llm)
    return chain


def main():
    data_dir = "data/corpus"
    eval_set = load_eval_set()
    llm = get_llm()
    embedder = get_embedder()

    print("=== Building BASELINE pipeline (chunk_size=1000, no MMR, no reranker) ===")
    baseline_chain = build_chain_for_config(data_dir, BASELINE_INGESTION, BASELINE_RETRIEVAL, llm, embedder)
    baseline_dataset = run_chain_over_eval_set(baseline_chain, eval_set)
    print("Scoring baseline with RAGAS ...")
    baseline_scores = score_with_ragas(baseline_dataset)
    print(baseline_scores)

    print("\n=== Building OPTIMISED pipeline (current src/config.py settings) ===")
    optimised_chain = build_chain_for_config(data_dir, ingestion_config, retrieval_config, llm, embedder)
    optimised_dataset = run_chain_over_eval_set(optimised_chain, eval_set)
    print("Scoring optimised with RAGAS ...")
    optimised_scores = score_with_ragas(optimised_dataset)
    print(optimised_scores)

    print("\n=== Improvement ===")
    comparison = compare_baseline_vs_optimised(baseline_scores, optimised_scores)
    print(comparison.to_string(index=False))
    out_path = save_results(comparison)
    print(f"Saved comparison table to {out_path}")

    print("\n=== Latency benchmark (optimised pipeline, avg over eval set) ===")
    latencies = []
    for row in eval_set:
        result = answer_question(optimised_chain, row["question"])
        latencies.append(result.latency_seconds)
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average latency: {avg_latency:.2f}s over {len(latencies)} queries "
          f"(target: < 3.0s per the spec)")


if __name__ == "__main__":
    main()
