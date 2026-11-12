"""Deterministic CPU-only text chunking implementation.

Splits input text into bounded chunks with configurable overlap, exact offsets,
Unicode normalization, and boundary-aware slicing (paragraphs, sentences, spaces).
Generates stable, deterministic chunk IDs based on input content, config, and offsets.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from flowforge.ai.retrieval.exceptions import ChunkingError
from flowforge.ai.retrieval.validation import validate_chunk_bounds


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration parameters for deterministic text chunking.

    Attributes:
        max_chars: Maximum number of characters allowed in each chunk (must be >= 1).
        overlap: Number of characters to overlap between successive chunks (>= 0 and < max_chars).
        max_chunks: Safety bound on total number of chunks produced (prevents runaways).
        normalize_unicode: Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD', or None).
    """

    max_chars: int = 1000
    overlap: int = 200
    max_chunks: int = 100_000
    normalize_unicode: str | None = "NFC"

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ChunkingError(
                f"max_chars must be greater than or equal to 1, got {self.max_chars}"
            )
        if self.overlap < 0:
            raise ChunkingError(f"overlap cannot be negative, got {self.overlap}")
        if self.overlap >= self.max_chars:
            raise ChunkingError(
                f"overlap must be strictly less than max_chars ({self.overlap} >= {self.max_chars})"
            )
        if self.max_chunks < 1:
            raise ChunkingError(
                f"max_chunks must be greater than or equal to 1, got {self.max_chunks}"
            )
        if self.normalize_unicode is not None and self.normalize_unicode not in (
            "NFC",
            "NFD",
            "NFKC",
            "NFKD",
        ):
            raise ChunkingError(
                f"Invalid normalize_unicode form: '{self.normalize_unicode}'. Expected NFC, NFD, NFKC, or NFKD"
            )


@dataclass(frozen=True, slots=True)
class TextChunk:
    """Represents a bounded text chunk produced by the chunker.

    Attributes:
        chunk_id: Deterministic SHA-256 derived hex string identifier.
        text: Slice of text contained in this chunk.
        start_offset: 0-indexed start character offset in original normalized text.
        end_offset: 0-indexed end character offset in original normalized text (exclusive).
        index: Sequential chunk index within the document (0-based).
        metadata: Optional metadata for tracking origin and provenance.
    """

    chunk_id: str
    text: str
    start_offset: int
    end_offset: int
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_chunk_id(
    text: str,
    start_offset: int,
    end_offset: int,
    index: int,
    config: ChunkingConfig,
    salt: str = "",
) -> str:
    """Compute a stable, deterministic chunk ID from content and configuration.

    Args:
        text: Chunk text content.
        start_offset: Start offset.
        end_offset: End offset.
        index: Chunk sequence index.
        config: ChunkingConfig used.
        salt: Optional salt/prefix for document context.

    Returns:
        Hex-encoded SHA-256 hash string (32 characters / 128-bit truncated).
    """
    hasher = hashlib.sha256()
    hasher.update(salt.encode())
    hasher.update(b":")
    hasher.update(f"{config.max_chars}:{config.overlap}:{config.normalize_unicode}:".encode())
    hasher.update(f"{index}:{start_offset}:{end_offset}:".encode())
    hasher.update(text.encode())
    return hasher.hexdigest()[:32]


def find_best_boundary(
    text: str,
    window_start: int,
    window_end: int,
    min_cut_ratio: float = 0.5,
) -> int:
    """Find the best natural split boundary (paragraph, newline, sentence, whitespace) within a window.

    Evaluates boundaries in order of semantic importance:
    1. Paragraph break: '\n\n'
    2. Line break: '\n'
    3. Sentence boundary: '. ' or '? ' or '! '
    4. Word boundary: ' '

    To avoid tiny fragments, only considers boundaries positioned past
    window_start + min_cut_ratio * (window_end - window_start).

    Args:
        text: The full text.
        window_start: Current chunk start offset.
        window_end: Maximum chunk end offset (window_start + max_chars).
        min_cut_ratio: Minimum fraction of window to accept for a boundary.

    Returns:
        The character index in text where the split should occur.
    """
    window = text[window_start:window_end]
    min_progress = max(1, int(len(window) * min_cut_ratio))

    # 1. Paragraph boundary: '\n\n'
    last_para = window.rfind("\n\n")
    if last_para != -1 and last_para >= min_progress:
        return window_start + last_para

    # 2. Line break boundary: '\n'
    last_newline = window.rfind("\n")
    if last_newline != -1 and last_newline >= min_progress:
        return window_start + last_newline

    # 3. Sentence boundary: '. ' or '? ' or '! '
    best_sentence = -1
    for punct in (". ", "? ", "! "):
        pos = window.rfind(punct)
        if pos != -1 and (pos + 1) > best_sentence:
            best_sentence = pos + 1  # include punctuation in current chunk

    if best_sentence != -1 and best_sentence >= min_progress:
        return window_start + best_sentence

    # 4. Word boundary: whitespace ' '
    last_space = window.rfind(" ")
    if last_space != -1 and last_space >= min_progress:
        return window_start + last_space

    # 5. Fallback: hard cut at window_end
    return window_end


def chunk_text(
    text: str | None,
    config: ChunkingConfig | None = None,
    metadata: dict[str, Any] | None = None,
    document_id: str = "",
) -> list[TextChunk]:
    """Chunk input text into bounded, overlapping chunks with deterministic IDs and exact offsets.

    Args:
        text: Input string to chunk. If None, empty, or whitespace-only, returns empty list.
        config: ChunkingConfig parameters (defaults to ChunkingConfig()).
        metadata: Optional dictionary attached to each chunk's metadata.
        document_id: Optional document identifier included in deterministic chunk ID generation.

    Returns:
        List of TextChunk instances.

    Raises:
        ChunkingError: If text exceeds safety bounds or chunk count exceeds max_chunks.
    """
    if text is None:
        return []

    if config is None:
        config = ChunkingConfig()

    validate_chunk_bounds(text)

    if config.normalize_unicode:
        text = unicodedata.normalize(config.normalize_unicode, text)

    # If completely empty or whitespace only
    if not text.strip():
        return []

    total_len = len(text)
    max_chars = config.max_chars
    overlap = config.overlap
    chunks: list[TextChunk] = []

    start = 0
    chunk_index = 0

    while start < total_len:
        # Determine maximum potential end for current window
        target_end = min(start + max_chars, total_len)
        end = target_end if target_end == total_len else find_best_boundary(text, start, target_end)

        chunk_slice = text[start:end]

        chunk_id = compute_chunk_id(
            text=chunk_slice,
            start_offset=start,
            end_offset=end,
            index=chunk_index,
            config=config,
            salt=document_id,
        )

        chunk_meta = dict(metadata or {})
        chunk = TextChunk(
            chunk_id=chunk_id,
            text=chunk_slice,
            start_offset=start,
            end_offset=end,
            index=chunk_index,
            metadata=chunk_meta,
        )
        chunks.append(chunk)
        chunk_index += 1

        if len(chunks) > config.max_chunks:
            raise ChunkingError(f"Generated chunk count exceeds limit of {config.max_chunks}")

        if end >= total_len:
            break

        # Advance start offset: advance by (end - start - overlap)
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1

        # If overlap is 0 or next_start is at/past previous end, trim boundary whitespace
        if next_start >= end:
            while next_start < total_len and text[next_start] in ("\n", "\r", " "):
                next_start += 1

        start = next_start

    return chunks
