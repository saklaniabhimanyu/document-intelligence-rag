"""
Tests for Component A (ingestion). These run with no network access and no
model downloads — only langchain's text splitter, which is pure Python —
so they're safe to run in CI without a Groq key or HF model cache.

Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion import clean_text, remove_repeated_lines, chunk_pages, ingest_file
from src.config import IngestionConfig


def test_clean_text_strips_page_numbers():
    raw = "Some content here.\n42\nMore content after the page number."
    cleaned = clean_text(raw)
    assert "42" not in cleaned.split("\n")


def test_clean_text_collapses_whitespace():
    raw = "Too    many     spaces"
    assert clean_text(raw) == "Too many spaces"


def test_clean_text_collapses_blank_lines():
    raw = "Line one\n\n\n\n\nLine two"
    cleaned = clean_text(raw)
    assert "\n\n\n" not in cleaned


def test_remove_repeated_lines_strips_running_header():
    pages = [
        (1, "CONFIDENTIAL\nPage one content."),
        (2, "CONFIDENTIAL\nPage two content."),
        (3, "CONFIDENTIAL\nPage three content."),
    ]
    cleaned = remove_repeated_lines(pages, min_repeats=3)
    for _, text in cleaned:
        assert "CONFIDENTIAL" not in text


def test_remove_repeated_lines_keeps_unique_content():
    pages = [
        (1, "HEADER\nUnique content A."),
        (2, "HEADER\nUnique content B."),
        (3, "HEADER\nUnique content C."),
    ]
    cleaned = remove_repeated_lines(pages, min_repeats=3)
    assert "Unique content A." in cleaned[0][1]
    assert "Unique content B." in cleaned[1][1]


def test_chunk_pages_respects_chunk_size_roughly():
    long_text = "Sentence number {}. " * 1  # build below
    body = " ".join(f"Sentence number {i}." for i in range(200))
    pages = [(1, body)]
    config = IngestionConfig(chunk_size=200, chunk_overlap=20)
    chunks = chunk_pages(pages, source="test.txt", config=config)
    assert len(chunks) > 1
    # allow some slack since the splitter prefers sentence boundaries
    assert all(len(c.text) <= 250 for c in chunks)


def test_chunk_pages_preserves_metadata():
    pages = [(3, "Some page three text that is long enough to form a chunk on its own.")]
    chunks = chunk_pages(pages, source="report.pdf", config=IngestionConfig(chunk_size=500, chunk_overlap=50))
    assert len(chunks) >= 1
    assert chunks[0].source == "report.pdf"
    assert chunks[0].page == 3
    assert len(chunks[0].chunk_id) == 8


def test_chunk_overlap_preserves_boundary_sentence():
    # A sentence that would fall right on a naive hard-cut boundary should
    # still appear whole in at least one chunk thanks to overlap.
    body = (
        "Alpha section discusses background information at some length to pad it out. " * 3
        + "This exact sentence must not be cut in half by chunking."
        + " Beta section continues with more padding text to push past the boundary. " * 3
    )
    config = IngestionConfig(chunk_size=150, chunk_overlap=40)
    chunks = chunk_pages([(1, body)], source="x.txt", config=config)
    # the splitter prefers ". " as a split point, so the trailing period may
    # be stripped as the separator itself — check the sentence body instead
    assert any("This exact sentence must not be cut in half by chunking" in c.text for c in chunks)


def test_ingest_file_end_to_end_on_sample_txt():
    chunks = ingest_file("data/sample.txt")
    assert len(chunks) > 0
    assert all(c.source == "sample.txt" for c in chunks)
    assert all(c.page == 1 for c in chunks)  # txt loader treats whole file as page 1
    # the confidential footer line shouldn't survive cleaning as a lone artifact
    assert all(c.text.strip() != "" for c in chunks)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
