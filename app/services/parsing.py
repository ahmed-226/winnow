"""Parsers for PDF, Markdown, and HTML into normalized text blocks.

Plan §10.1-§10.4. All parsers return `ParsedBlock` items whose text is
already normalized (NFC, control characters removed, whitespace collapsed)
and non-empty; the caller rejects documents with too little total text.
PDF metadata (author, producer, dates) is never extracted.
"""

from __future__ import annotations

import re
import unicodedata
from typing import NamedTuple

import pymupdf

_MIN_TOTAL_TEXT_CHARS = 50


class ParsedBlock(NamedTuple):
    text: str
    page: int | None  # PDF only, 1-based
    heading_path: tuple[str, ...]  # Markdown/HTML headings, empty for PDF


class ParseFailure(Exception):
    """A parse-stage failure with a worker error-code classification."""

    code = "parse_failed"
    retryable = True


class PasswordProtectedPdfError(ParseFailure):
    code = "password_protected_pdf"
    retryable = False


class PdfTooLargeError(ParseFailure):
    code = "pdf_too_large"
    retryable = False


class NoExtractableTextError(ParseFailure):
    code = "no_extractable_text"
    retryable = False


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MANY_BLANK_LINES = re.compile(r"\n{4,}")
_SPACES_IN_LINE = re.compile(r"[ \t]{2,}")
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def normalize_text(text: str) -> str:
    """NFC-normalize and strip noise so identical inputs chunk identically."""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = _MANY_BLANK_LINES.sub("\n\n\n", text)
    text = "\n".join(_SPACES_IN_LINE.sub(" ", line) for line in text.split("\n"))
    return text.strip()


def parse(data: bytes, *, source_type: str, pdf_max_pages: int) -> list[ParsedBlock]:
    """Parse raw bytes into normalized blocks; raises ParseFailure subclasses."""
    if source_type == "pdf":
        blocks = _parse_pdf(data, pdf_max_pages=pdf_max_pages)
    elif source_type == "markdown":
        blocks = _parse_markdown(data.decode("utf-8", errors="replace"))
    elif source_type == "html":
        blocks = _parse_html(data)
    else:  # pragma: no cover - upload validation rejects other values
        raise ParseFailure(f"Unsupported source type {source_type!r}.")

    normalized: list[ParsedBlock] = []
    for block in blocks:
        text = normalize_text(block.text)
        if text:
            normalized.append(ParsedBlock(text, block.page, block.heading_path))
    return normalized


def total_text_length(blocks: list[ParsedBlock]) -> int:
    """Length of the normalized document text (blocks joined by blank lines)."""
    if not blocks:
        return 0
    return sum(len(block.text) for block in blocks) + 2 * (len(blocks) - 1)


def _parse_pdf(data: bytes, *, pdf_max_pages: int) -> list[ParsedBlock]:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise ParseFailure("Could not open the PDF file.") from exc
    try:
        if document.needs_pass:
            raise PasswordProtectedPdfError("The PDF is password protected.")
        if document.page_count > pdf_max_pages:
            raise PdfTooLargeError(
                f"The PDF has {document.page_count} pages; the limit is {pdf_max_pages}."
            )
        blocks: list[ParsedBlock] = []
        for page in document:  # type: ignore[attr-defined]
            for item in page.get_text("blocks"):
                # Block type 0 is a text block; 1 is an image block.
                if item[6] == 0 and item[4]:
                    blocks.append(ParsedBlock(item[4], page.number + 1, ()))
        return blocks
    finally:
        document.close()  # type: ignore[no-untyped-call]


def _parse_markdown(text: str) -> list[ParsedBlock]:
    lines = text.splitlines()
    blocks: list[ParsedBlock] = []

    # Remove YAML front matter: metadata noise, never a searchable chunk.
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                lines = lines[index + 1 :]
                break

    heading_path: list[str] = []
    paragraph: list[str] = []
    code_lines: list[str] | None = None
    fence_marker = ""

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(ParsedBlock("\n".join(paragraph), None, tuple(heading_path)))
            paragraph.clear()

    def flush_code() -> None:
        if code_lines is not None:
            blocks.append(ParsedBlock("\n".join(code_lines), None, tuple(heading_path)))

    for line in lines:
        stripped = line.strip()
        fence_match = re.fullmatch(r"(`{3,}|~{3,})(.*)", stripped)
        if code_lines is not None:
            if fence_match and fence_match.group(1)[0] == fence_marker[0]:
                code_lines.append(stripped)  # keep the closing fence with the block
                flush_code()
                code_lines = None
            else:
                code_lines.append(line)
            continue
        if fence_match:
            flush_paragraph()
            fence_marker = fence_match.group(1)
            code_lines = [stripped]
            continue
        heading = _ATX_HEADING.match(stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            del heading_path[level - 1 :]
            heading_path.append(heading.group(2))
            paragraph.append(stripped)  # the heading leads its section text
            continue
        if not stripped:
            flush_paragraph()
            continue
        paragraph.append(line)
    flush_code()
    flush_paragraph()
    return blocks


def _parse_html(data: bytes) -> list[ParsedBlock]:
    from trafilatura import extract

    text = extract(
        data.decode("utf-8", errors="replace"),
        output_format="markdown",
        favor_recall=True,
    )
    if text and text.strip():
        blocks = _parse_markdown(text)
        if blocks:
            return blocks

    # Fallback: BeautifulSoup walk over headings, paragraphs, lists, code, tables.
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    heading_path: list[str] = []
    fallback_blocks: list[ParsedBlock] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "td"]):
        name = element.name
        text = element.get_text(" ")
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"} and text.strip():
            level = int(name[1])
            del heading_path[level - 1 :]
            heading_path.append(text.strip())
        if text.strip():
            fallback_blocks.append(ParsedBlock(text, None, tuple(heading_path)))
    return fallback_blocks
