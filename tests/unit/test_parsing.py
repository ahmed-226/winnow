"""Unit tests for parsing and normalization (plan §10)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.services.parsing import (
    NoExtractableTextError,
    ParsedBlock,
    ParseFailure,
    normalize_text,
    parse,
    total_text_length,
)


def test_normalize_removes_control_characters_and_collapses_whitespace() -> None:
    text = "hello\x00\x07world\n\n\n\n\nsecond\x1fline  with\t\tspaces "
    normalized = normalize_text(text)
    assert "\x00" not in normalized
    assert "\x07" not in normalized
    assert "\x1f" not in normalized
    assert "helloworld" in normalized
    assert "secondline with spaces" in normalized


def test_markdown_headings_and_front_matter() -> None:
    data = (
        b"---\ntitle: secret\ntags: [a]\n---\n"
        b"# Guide\n\nIntro paragraph.\n\n## Setup\n\nBody text here.\n"
        b"```python\nprint('hi')\n```\n"
    )
    blocks = parse(data, source_type="markdown", pdf_max_pages=10)
    texts = [block.text for block in blocks]
    assert not any("secret" in text for text in texts)  # front matter removed
    assert any("Intro paragraph." in text for text in texts)
    code_blocks = [block for block in blocks if "print('hi')" in block.text]
    assert len(code_blocks) == 1
    assert code_blocks[0].text.count("```") == 2  # fence kept intact
    heading_paths = [block.heading_path for block in blocks if "Body text" in block.text]
    assert heading_paths == [("Guide", "Setup")]


def test_markdown_heading_is_leading_line_of_its_block() -> None:
    blocks = parse(b"# Title\nBody text", source_type="markdown", pdf_max_pages=10)
    assert len(blocks) == 1
    assert blocks[0].text.startswith("# Title")
    assert "Body text" in blocks[0].text


def test_pdf_extraction_with_pages() -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "First page paragraph.")
    page2 = document.new_page()
    page2.insert_text((72, 100), "Second page paragraph.")
    data = document.tobytes()
    document.close()

    blocks = parse(data, source_type="pdf", pdf_max_pages=10)
    assert [block.page for block in blocks if "First page" in block.text] == [1]
    assert [block.page for block in blocks if "Second page" in block.text] == [2]


def test_encrypted_pdf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "enc.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(
        str(path),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    data = path.read_bytes()
    document.close()
    with pytest.raises(ParseFailure):
        parse(data, source_type="pdf", pdf_max_pages=10)


def test_pdf_page_limit() -> None:
    document = pymupdf.open()
    for _ in range(3):
        document.new_page()
    data = document.tobytes()
    document.close()
    with pytest.raises(ParseFailure):
        parse(data, source_type="pdf", pdf_max_pages=2)


def test_garbage_pdf_raises_clean_failure() -> None:
    with pytest.raises(ParseFailure):
        parse(b"%PDF-1.4 not really a pdf at all", source_type="pdf", pdf_max_pages=10)


def test_total_text_length_counts_separators() -> None:
    blocks = [ParsedBlock("abc", None, ()), ParsedBlock("def", None, ())]
    assert total_text_length(blocks) == 3 + 2 + 3
    assert total_text_length([]) == 0


def test_no_text_blocks_are_dropped() -> None:
    blocks = parse(b"\n\n\n\n   \n", source_type="markdown", pdf_max_pages=10)
    assert blocks == []
    assert NoExtractableTextError.code == "no_extractable_text"
