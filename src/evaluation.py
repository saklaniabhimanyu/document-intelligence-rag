"""
Component E — RAGAS Evaluation.

Builds/loads a (question, ground_truth, source) eval set, runs it through a
given QA chain, scores it with RAGAS (faithfulness, answer_relevancy,
context_precision, context_recall), and computes the % improvement of an
"optimised" pipeline over a "baseline" one — this is where the 38% headline
number in the README comes from.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


EVAL_SET_PATH = "data/eval_set.json"


def load_eval_set(path: str = EVAL_SET_PATH) -> List[Dict[str, str]]:
    """Each row: {"question": ..., "ground_truth": ..., "source": ...}
    Aim for ~50 rows spanning the breadth of your corpus per the spec."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/generate_eval_dataset.py first to "
            f"create a template, then fill in ground_truth answers by hand."
        )
    return json.loads(p.read_text())


def run_chain_over_eval_set(chain, eval_set: List[Dict[str, str]]) -> Dict[str, list]:
    """Executes every question through the chain and assembles the
    RAGAS-required columns: question, answer, contexts, ground_truth."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for row in eval_set:
        result = chain.invoke({"query": row["question"]})
        questions.append(row["question"])
        answers.append(result["result"])
        contexts.append([d.page_content for d in result.get("source_documents", [])])
        ground_truths.append(row["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def score_with_ragas(dataset_dict: Dict[str, list]) -> Dict[str, float]:
    """Computes the 4 core RAGAS metrics required by the spec."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    ds = Dataset.from_dict(dataset_dict)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {k: float(v) for k, v in result.items()}


def compare_baseline_vs_optimised(
    baseline_scores: Dict[str, float],
    optimised_scores: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for metric in optimised_scores:
        base = baseline_scores.get(metric, 0.0)
        opt = optimised_scores.get(metric, 0.0)
        pct_improvement = ((opt - base) / base * 100) if base else float("nan")
        rows.append(
            {
                "metric": metric,
                "baseline": round(base, 3),
                "optimised": round(opt, 3),
                "improvement_pct": round(pct_improvement, 1),
            }
        )
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, path: str = "eval_results/comparison.csv") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
