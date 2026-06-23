"""
Component F — Streamlit UI.

File upload (PDF/DOCX/TXT) -> ingestion progress bar -> chat interface.
Every answer shows source citations (filename + page) and, optionally,
the raw retrieved chunks for transparency.

Run with:
    streamlit run app.py
"""
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from src.config import ingestion_config, retrieval_config, embedding_config
from src.ingestion import ingest_file
from src.embeddings import build_index, add_chunks_to_index, get_embedder
from src.retriever import ConfigurableRetriever
from src.qa_chain import build_qa_chain, answer_question, get_llm


st.set_page_config(page_title="Document Intelligence (RAG)", layout="wide")

if "index" not in st.session_state:
    st.session_state.index = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

st.title(" Document Intelligence System")
st.caption("Ask questions over your own PDFs / DOCX / TXT files — answers are grounded in retrieved passages, with citations.")

with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )
    show_chunks = st.toggle("Show retrieved chunks for transparency", value=False)

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.ingested_files]
        if new_files:
            progress = st.progress(0.0, text="Starting ingestion...")
            all_new_chunks = []
            for i, uf in enumerate(new_files):
                progress.progress(
                    i / len(new_files), text=f"Ingesting {uf.name} ({i + 1}/{len(new_files)})"
                )
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uf.name).suffix) as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name
                chunks = ingest_file(tmp_path, config=ingestion_config)
                # restore the real filename in metadata instead of the temp path
                for c in chunks:
                    c.source = uf.name
                all_new_chunks.extend(chunks)
                st.session_state.ingested_files.append(uf.name)

            progress.progress(0.9, text="Embedding chunks...")
            if st.session_state.embedder is None:
                st.session_state.embedder = get_embedder(embedding_config)

            if st.session_state.index is None:
                st.session_state.index = build_index(all_new_chunks, embedder=st.session_state.embedder)
            else:
                add_chunks_to_index(st.session_state.index, all_new_chunks)

            retriever = ConfigurableRetriever(index=st.session_state.index, config=retrieval_config)
            try:
                st.session_state.chain = build_qa_chain(retriever)
            except RuntimeError as e:
                st.error(str(e))

            progress.progress(1.0, text="Done!")
            time.sleep(0.3)
            progress.empty()
            st.success(f"Ingested {len(new_files)} file(s), {len(all_new_chunks)} chunks total.")

    if st.session_state.ingested_files:
        st.markdown("**Indexed files:**")
        for f in st.session_state.ingested_files:
            st.markdown(f"- {f}")

    st.divider()
    st.header("Retrieval settings")
    retrieval_config.top_k = st.slider("Top-k chunks", 1, 10, retrieval_config.top_k)
    retrieval_config.use_mmr = st.toggle("Use MMR (reduce redundancy)", value=retrieval_config.use_mmr)
    retrieval_config.use_reranker = st.toggle(
        "Use cross-encoder reranker", value=retrieval_config.use_reranker
    )

st.header("2. Ask questions")

if st.session_state.chain is None:
    st.info("Upload at least one document in the sidebar to start chatting.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 Sources ({msg['latency']}s)"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['source']}**, page {s['page']}")
                        if show_chunks:
                            st.caption(s["snippet"])

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving + generating..."):
                try:
                    result = answer_question(st.session_state.chain, question)
                    st.markdown(result.answer)
                    with st.expander(f"📎 Sources ({result.latency_seconds}s)"):
                        for s in result.sources:
                            st.markdown(f"**{s['source']}**, page {s['page']}")
                            if show_chunks:
                                st.caption(s["snippet"])
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.sources,
                            "latency": result.latency_seconds,
                        }
                    )
                except RuntimeError as e:
                    st.error(str(e))
