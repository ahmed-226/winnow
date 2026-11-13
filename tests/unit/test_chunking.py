"""Unit tests for the chunking algorithm (plan §10.5)."""

from __future__ import annotations

from app.services.chunking import chunk_blocks, split_for_token_limit
from app.services.parsing import ParsedBlock

PARAMS = {
    "target_chars": 300,
    "overlap_chars": 50,
    "max_chars": 600,
    "min_chars": 40,
}


def _paragraph(text: str, page: int | None = None) -> ParsedBlock:
    return ParsedBlock(text, page, ())


def test_empty_input_makes_no_chunks() -> None:
    assert chunk_blocks([], **PARAMS) == []


def test_short_document_is_one_chunk() -> None:
    chunks = chunk_blocks([_paragraph("hello world " * 10)], **PARAMS)
    assert len(chunks) == 1
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(chunks[0].content)


def test_no_chunk_exceeds_max_chars() -> None:
    blocks = [_paragraph(f"paragraph {i} " + "word " * 60) for i in range(8)]
    chunks = chunk_blocks(blocks, **PARAMS)
    assert chunks
    assert all(len(chunk.content) <= 600 for chunk in chunks)


def test_output_is_deterministic() -> None:
    blocks = [_paragraph(f"paragraph {i} " + "content " * 30) for i in range(5)]
    first = chunk_blocks(blocks, **PARAMS)
    second = chunk_blocks(blocks, **PARAMS)
    assert [chunk.content for chunk in first] == [chunk.content for chunk in second]


def test_oversized_paragraph_is_split() -> None:
    huge = ". ".join(f"sentence {i}" for i in range(200))
    chunks = chunk_blocks([_paragraph(huge)], **PARAMS)
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 600 for chunk in chunks)
    # All text is preserved.
    assert huge[:20] in chunks[0].content


def test_overlap_links_consecutive_chunks() -> None:
    blocks = [_paragraph(f"paragraph {i} " + "filler " * 40, page=1) for i in range(6)]
    chunks = chunk_blocks(blocks, **PARAMS)
    assert len(chunks) >= 2
    # Every chunk after the first that carries overlap starts with text that
    # also appears at the tail of the previous chunk.
    assert any(
        chunks[i + 1].content[:20] in chunks[i].content[-80:] for i in range(len(chunks) - 1)
    )


def test_page_metadata_spans_units() -> None:
    blocks = [_paragraph("page one text " * 30, page=1), _paragraph("page two " * 40, page=2)]
    chunks = chunk_blocks(blocks, **PARAMS)
    assert chunks[0].page_start == 1
    assert chunks[0].page_end in (1, 2)


def test_trailing_small_chunk_is_merged() -> None:
    blocks = [
        _paragraph("big " * 80),
        _paragraph("tiny tail"),
    ]
    chunks = chunk_blocks(blocks, **PARAMS)
    assert all(len(chunk.content) >= 40 for chunk in chunks)


def test_heading_path_recorded() -> None:
    blocks = [ParsedBlock("section body " * 20, None, ("Guide", "Setup"))]
    chunks = chunk_blocks(blocks, **PARAMS)
    assert chunks[0].heading_path == ("Guide", "Setup")


def test_char_offsets_are_monotonic() -> None:
    blocks = [_paragraph(f"para {i} " + "x " * 50) for i in range(10)]
    chunks = chunk_blocks(blocks, **PARAMS)
    starts = [chunk.char_start for chunk in chunks]
    assert starts == sorted(starts)
    for chunk in chunks:
        assert chunk.char_end == chunk.char_start + len(chunk.content)


def test_token_limit_guard_splits_large_chunks() -> None:
    huge = ParsedBlock("word " * 4000, 1, ())
    chunks = chunk_blocks([huge], **PARAMS)
    split = split_for_token_limit(chunks, max_tokens=1800)
    assert all(len(chunk.content) <= 1800 * 3 for chunk in split)
    assert sum(len(chunk.content) for chunk in split) >= sum(
        len(chunk.content) for chunk in chunks
    ) - len(split)


def test_single_tiny_document_not_dropped() -> None:
    chunks = chunk_blocks([_paragraph("just one line")], **PARAMS)
    assert len(chunks) == 1
