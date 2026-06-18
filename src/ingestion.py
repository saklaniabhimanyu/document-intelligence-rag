"""
Component A — Document Ingestion Pipeline.

Loads PDF / DOCX / TXT, cleans the raw text (strips page numbers, repeated
headers/footers, excess whitespace), and splits it into overlapping chunks
ready for embedding. Every chunk carries metadata: {source, page, chunk_id}.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import IngestionConfig, ingestion_config


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loaders — each returns List[(page_number, raw_text)]
# --------------------------------------------------------------------------- #
def load_pdf(path: str) -> List[tuple[int, str]]:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append((i, page.get_text("text")))
    return pages


def load_docx(path: str) -> List[tuple[int, str]]:
    import docx

    document = docx.Document(path)
    full_text = "\n".join(p.text for p in document.paragraphs)
    # DOCX has no native page concept; treat the whole doc as "page 1".
    return [(1, full_text)]


def load_txt(path: str) -> List[tuple[int, str]]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [(1, text)]


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
}


def load_document(path: str) -> List[tuple[int, str]]:
    ext = Path(path).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADERS)}")
    return LOADERS[ext](path)


# --------------------------------------------------------------------------- #
# Cleaner
# --------------------------------------------------------------------------- #
_PAGE_NUM_RE = re.compile(r"^\s*-?\s*\d+\s*-?\s*$", re.MULTILINE)
_MULTI_WS_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str) -> str:
    """Strip standalone page numbers, control characters, and collapse whitespace."""
    text = _NON_PRINTABLE_RE.sub("", text)
    text = _PAGE_NUM_RE.sub("", text)
    text = _MULTI_WS_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def remove_repeated_lines(pages: List[tuple[int, str]], min_repeats: int = 3) -> List[tuple[int, str]]:
    """Detect lines (e.g. running headers/footers) that repeat on >= min_repeats
    pages verbatim and strip them — a cheap, dependency-free header/footer filter."""
    if len(pages) < min_repeats:
        return pages

    line_counts: Dict[str, int] = {}
    for _, text in pages:
        for line in {l.strip() for l in text.splitlines() if l.strip()}:
            line_counts[line] = line_counts.get(line, 0) + 1

    boilerplate = {line for line, count in line_counts.items() if count >= min_repeats}

    cleaned_pages = []
    for page_num, text in pages:
        kept_lines = [l for l in text.splitlines() if l.strip() not in boilerplate]
        cleaned_pages.append((page_num, "\n".join(kept_lines)))
    return cleaned_pages


# --------------------------------------------------------------------------- #
# Chunker
# --------------------------------------------------------------------------- #
def chunk_pages(
    pages: List[tuple[int, str]],
    source: str,
    config: IngestionConfig = ingestion_config,
) -> List[Chunk]:
    """Split cleaned page text into overlapping chunks, preserving page metadata.
    Uses LangChain's RecursiveCharacterTextSplitter so splits prefer paragraph/
    sentence boundaries over mid-sentence cuts."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
    )

    chunks: List[Chunk] = []
    for page_num, raw_text in pages:
        text = clean_text(raw_text)
        if not text:
            continue
        for piece in splitter.split_text(text):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    text=piece.strip(),
                    source=source,
                    page=page_num,
                    chunk_id=str(uuid.uuid4())[:8],
                )
            )
    return chunks


def ingest_file(path: str, config: IngestionConfig = ingestion_config) -> List[Chunk]:
    """Full pipeline for a single file: load -> dehead/defoot -> clean -> chunk."""
    pages = load_document(path)
    pages = remove_repeated_lines(pages)
    source = Path(path).name
    return chunk_pages(pages, source=source, config=config)


def ingest_directory(directory: str, config: IngestionConfig = ingestion_config) -> List[Chunk]:
    """Ingest every supported file in a directory."""
    all_chunks: List[Chunk] = []
    for path in sorted(Path(directory).glob("*")):
        if path.suffix.lower() in LOADERS:
            all_chunks.extend(ingest_file(str(path), config=config))
    return all_chunks


if __name__ == "__main__":
    import sys
    import json

    target = sys.argv[1] if len(sys.argv) > 1 else "data/sample.txt"
    result_chunks = ingest_file(target)
    print(f"Ingested {target} -> {len(result_chunks)} chunks")
    print(json.dumps(result_chunks[0].to_dict(), indent=2) if result_chunks else "No chunks produced.")
