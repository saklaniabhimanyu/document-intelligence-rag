"""
Component D — LangChain QA Chain.

Wires the ConfigurableRetriever (Component C) to Groq's hosted Llama 3 70B
via a custom prompt that forces grounded, citation-bearing answers and an
explicit "I don't know" fallback when the context doesn't contain the answer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.docstore.document import Document

from src.config import LLMConfig, PromptConfig, llm_config, prompt_config
from src.retriever import ConfigurableRetriever


@dataclass
class QAResult:
    answer: str
    sources: List[dict]
    latency_seconds: float


def get_llm(config: LLMConfig = llm_config):
    from langchain_groq import ChatGroq

    if not config.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free at https://console.groq.com/keys)."
        )
    return ChatGroq(
        api_key=config.groq_api_key,
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def build_qa_chain(
    retriever: ConfigurableRetriever,
    llm=None,
    prompt_cfg: PromptConfig = prompt_config,
) -> RetrievalQA:
    llm = llm or get_llm()
    prompt = PromptTemplate(
        template=prompt_cfg.template, input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


def format_sources(docs: List[Document]) -> List[dict]:
    seen = set()
    sources = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", "?"),
                "snippet": d.page_content[:200],
            }
        )
    return sources


def answer_question(chain: RetrievalQA, question: str) -> QAResult:
    """Run the chain and time it — this is what scripts/run_eval.py averages
    over 50 queries for the 'under 3 seconds' README claim."""
    start = time.time()
    result = chain.invoke({"query": question})
    elapsed = time.time() - start

    return QAResult(
        answer=result["result"],
        sources=format_sources(result.get("source_documents", [])),
        latency_seconds=round(elapsed, 2),
    )


def answer_without_rag(question: str, llm=None) -> str:
    """Bare-LLM answer with no retrieved context — used purely for the
    RAG-vs-no-RAG comparison screenshot called out in the build sequence
    (step 5: 'RAG should win')."""
    llm = llm or get_llm()
    response = llm.invoke(question)
    return response.content
