"""Paragraph-aware chunking (plan §10.5).

The output is a deterministic function of the input blocks, so the same bytes
always produce the same chunk boundaries and reprocessing is stable. Chunks
carry citation metadata (page range, heading path, char offsets relative to
the normalized document text, which is the blocks joined by blank lines).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.parsing import ParsedBlock

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

_Unit = tuple[str, int | None, tuple[str, ...], int]  # text, page, heading_path, char_start


@dataclass(frozen=True, slots=True)
class Chunk:
    content: str
    page_start: int | None
    page_end: int | None
    heading_path: tuple[str, ...]
    char_start: int
    char_end: int

    @property
    def token_estimate(self) -> int:
        return -(-len(self.content) // 3)  # ceil(len / 3)


@dataclass(frozen=True, slots=True)
class _Paragraph:
    text: str
    page: int | None
    heading_path: tuple[str, ...]
    char_start: int


def build_paragraphs(blocks: list[ParsedBlock]) -> list[_Paragraph]:
    """Join blocks with blank lines; track each block's char offset."""
    paragraphs: list[_Paragraph] = []
    offset = 0
    for block in blocks:
        paragraphs.append(_Paragraph(block.text, block.page, block.heading_path, offset))
        offset += len(block.text) + 2  # "\n\n" separator
    return paragraphs


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split text over the hard limit at sentence boundaries, then words."""
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = -1
        for match in _SENTENCE_BOUNDARY.finditer(window):
            cut = match.start()
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining.strip():
        pieces.append(remaining.strip())
    return [piece for piece in pieces if piece]


def _overlap_tail(content: str, overlap_chars: int, max_chars: int, next_len: int) -> str:
    """Tail of the emitted chunk that opens the next buffer (may be empty)."""
    if overlap_chars <= 0 or len(content) <= overlap_chars:
        return ""  # overlap would duplicate the entire chunk
    if next_len + overlap_chars > max_chars:
        return ""  # rule 5: the hard limit wins over the overlap rule
    tail = content[-overlap_chars:].lstrip()
    boundary = tail.find(" ")
    if 0 <= boundary < len(tail) - 1:
        tail = tail[boundary + 1 :]
    return tail


def chunk_blocks(
    blocks: list[ParsedBlock],
    *,
    target_chars: int,
    overlap_chars: int,
    max_chars: int,
    min_chars: int,
) -> list[Chunk]:
    """Greedily pack paragraphs into chunks with sentence-preserving overlap."""
    units: list[_Unit] = []
    for paragraph in build_paragraphs(blocks):
        for piece in _split_oversized(paragraph.text, max_chars):
            # Pieces are prefix-anchored slices of the paragraph; they keep the
            # paragraph's char_start so offsets stay monotonic.
            units.append((piece, paragraph.page, paragraph.heading_path, paragraph.char_start))

    chunks: list[Chunk] = []
    buffer: list[_Unit] = []
    buffer_prefix = ""
    prefix_end: int | None = None  # char_end of the previous chunk behind the prefix

    def buffer_content() -> str:
        parts = ([buffer_prefix] if buffer_prefix else []) + [unit[0] for unit in buffer]
        return "\n\n".join(parts)

    def emit() -> None:
        nonlocal buffer, buffer_prefix, prefix_end
        content = buffer_content()
        first, last = buffer[0], buffer[-1]
        if buffer_prefix and prefix_end is not None:
            char_start = prefix_end - len(buffer_prefix)
        else:
            char_start = first[3]
        chunks.append(
            Chunk(
                content=content,
                page_start=first[1],
                page_end=last[1],
                heading_path=first[2],
                char_start=char_start,
                char_end=last[3] + len(last[0]),
            )
        )
        buffer = []
        buffer_prefix = ""
        prefix_end = None

    for unit in units:
        if buffer and len(buffer_content()) + 2 + len(unit[0]) > target_chars:
            emit()
            tail = _overlap_tail(chunks[-1].content, overlap_chars, max_chars, len(unit[0]) + 2)
            if tail:
                buffer_prefix = tail
                prefix_end = chunks[-1].char_end
        buffer.append(unit)
        if len(buffer_content()) > max_chars:
            # The hard limit wins: drop the overlap prefix for this chunk.
            buffer_prefix = ""
            prefix_end = None
    if buffer:
        emit()

    if len(chunks) > 1 and len(chunks[-1].content) < min_chars:
        last = chunks[-1]
        previous = chunks[-2]
        merged = f"{previous.content}\n\n{last.content}"
        if len(merged) <= max_chars:
            chunks[-2] = Chunk(
                content=merged,
                page_start=previous.page_start,
                page_end=last.page_end or previous.page_end,
                heading_path=previous.heading_path,
                char_start=previous.char_start,
                char_end=last.char_end,
            )
            chunks.pop()
    return chunks


def split_for_token_limit(chunks: list[Chunk], *, max_tokens: int = 1800) -> list[Chunk]:
    """Second guard before embedding: split chunks over the token estimate."""
    max_chars = max_tokens * 3
    result: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.content) <= max_chars:
            result.append(chunk)
            continue
        offset = chunk.char_start
        for piece in _split_oversized(chunk.content, max_chars):
            result.append(
                Chunk(
                    content=piece,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    heading_path=chunk.heading_path,
                    char_start=offset,
                    char_end=offset + len(piece),
                )
            )
            offset += len(piece) + 1
    return result
