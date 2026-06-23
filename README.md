# 📄 Document Intelligence System — RAG over your own documents

A Retrieval-Augmented Generation (RAG) pipeline that answers natural-language
questions over PDF / DOCX / TXT collections with grounded, cited answers —
instead of letting an LLM hallucinate from memory.

```
PDF / DOCX / TXT ──▶ clean + chunk ──▶ embed ──▶ FAISS index
                                                      │
your question ──▶ embed query ──▶ retrieve top-k ◀───┘
                                       │
                          (optional: MMR / cross-encoder rerank)
                                       │
                                       ▼
             Groq openai/gpt-oss-120b  +  "answer only from context"
                                       │
                                       ▼
                         answer + source citations (file, page)
```

## Why RAG instead of plain LLM chat

A general-purpose LLM can only answer from what it memorized during
training — ask it about an internal report or a document published after
its cutoff and it will confidently make something up. RAG fixes this by
retrieving the actual relevant passages from your documents at query time
and forcing the model to answer only from that retrieved text.

## Architecture (4 layers, matching `src/`)

| Layer | File | What it does |
|---|---|---|
| A. Ingestion | `src/ingestion.py` | Loads PDF (PyMuPDF) / DOCX (python-docx) / TXT, strips page numbers & repeated headers/footers, splits into 400-char/50-overlap chunks via `RecursiveCharacterTextSplitter` |
| B. Embedding + Vector Store | `src/embeddings.py` | Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, free, CPU), stores in a persisted FAISS index |
| C. Retrieval + LLM | `src/retriever.py`, `src/qa_chain.py` | Similarity / MMR / cross-encoder-reranked retrieval (`cross-encoder/ms-marco-MiniLM-L-6-v2`) → LangChain `RetrievalQA` → Groq `llama3-70b-8192` |
| D. Evaluation | `src/evaluation.py` | RAGAS: faithfulness, answer_relevancy, context_precision, context_recall — baseline vs. optimised comparison |

Plus `app.py` — a Streamlit chat UI with file upload, an ingestion progress
bar, and per-answer source citations.


## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd document_intelligence_rag
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

Alternatively, set them directly in the terminal.

#### PowerShell

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
$env:HF_TOKEN="your_huggingface_token"
```

#### Bash

```bash
export GROQ_API_KEY="your_groq_api_key"
export HF_TOKEN="your_huggingface_token"
```

### Why is HF_TOKEN needed?

The project uses Hugging Face models for embeddings and reranking. Some model downloads may require authentication through Hugging Face Hub.

---

## Building the Index

Place your documents inside:

```text
data/corpus/
```

Then build the vector index:

```bash
python -m scripts.build_index --data_dir data/corpus
```

---

## Running the Chat Application

```bash
streamlit run app.py
```

The application allows:

- Uploading documents
- Building an index
- Asking questions about uploaded content
- Viewing source citations

---

## Evaluation

Generate an evaluation dataset:

```bash
python -m scripts.generate_eval_dataset --n 50
```

Fill in the `ground_truth` answers manually inside:

```text
data/eval_set.json
```
Models Used
- Answer Generation (RAG Pipeline): openai/gpt-oss-120b via Groq
- RAGAS Evaluation: llama-3.3-70b-versatile via Groq

A separate evaluation model is used for RAGAS scoring to assess faithfulness, answer relevancy, context precision, and context recall independently of the answer-generation model.

Run evaluation:

```bash
python -m scripts.run_eval
```

Metrics reported:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Results are saved to:

```text
eval_results/comparison.csv
```

---

## Testing

```bash
pytest tests -v
```

---

## Project Structure

```text
document_intelligence_rag/
├── app.py
├── requirements.txt
├── data/
│   ├── corpus/
│   └── eval_set.json
├── scripts/
│   ├── build_index.py
│   ├── generate_eval_dataset.py
│   └── run_eval.py
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── qa_chain.py
│   └── evaluation.py
└── tests/
```

---
