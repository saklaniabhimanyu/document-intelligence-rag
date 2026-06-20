"""
Generates a template eval_set.json from the indexed corpus by sampling
random chunks and turning each into a draft question. You should review
and rewrite the ground_truth answers by hand (auto-generated ground truth
defeats the purpose of evaluation) before running scripts/run_eval.py.

Usage:
    python -m scripts.generate_eval_dataset --n 50
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.embeddings import load_index
from src.qa_chain import get_llm


DRAFT_QUESTION_PROMPT = (
    "Read the passage below and write ONE clear, specific question that the "
    "passage answers directly. Return only the question, no preamble.\n\n"
    "Passage:\n{passage}"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of Q&A pairs to draft")
    parser.add_argument("--out", default="data/eval_set.json")
    args = parser.parse_args()

    index = load_index()
    if index is None:
        raise RuntimeError("No FAISS index found. Run scripts/build_index.py first.")

    all_docs = list(index.docstore._dict.values())
    if len(all_docs) < args.n:
        print(f"Warning: corpus only has {len(all_docs)} chunks, drafting {len(all_docs)} instead of {args.n}.")
    sample = random.sample(all_docs, min(args.n, len(all_docs)))

    llm = get_llm()
    rows = []
    for doc in sample:
        question = llm.invoke(DRAFT_QUESTION_PROMPT.format(passage=doc.page_content)).content.strip()
        rows.append(
            {
                "question": question,
                "ground_truth": "<<< FILL IN THE CORRECT ANSWER BY HAND >>>",
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
            }
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"Drafted {len(rows)} questions -> {args.out}")
    print("IMPORTANT: open this file and fill in real ground_truth answers before evaluating.")


if __name__ == "__main__":
    main()
