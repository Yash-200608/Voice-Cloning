"""Sentence/clause chunking for incremental real-time synthesis.

This is application-layer chunking — not native model streaming.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")


def split_speech_units(text: str, *, max_chars: int = 180) -> list[str]:
    """Split text into speech units preserving punctuation and boundaries.

    Strategy:
      1. Prefer sentence boundaries (.?!)
      2. If a sentence exceeds max_chars, split on clause boundaries (,;:)
      3. As a last resort, hard-wrap on whitespace near max_chars

    Empty / whitespace-only text yields an empty list.
    """
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return []

    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]
    if not sentences:
        sentences = [cleaned]

    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue
        clauses = [c.strip() for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]
        if len(clauses) == 1:
            units.extend(_hard_wrap(sentence, max_chars))
            continue
        buf = ""
        for clause in clauses:
            candidate = f"{buf} {clause}".strip() if buf else clause
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                if buf:
                    units.append(buf)
                if len(clause) <= max_chars:
                    buf = clause
                else:
                    units.extend(_hard_wrap(clause, max_chars))
                    buf = ""
        if buf:
            units.append(buf)
    return units


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for word in words:
        add = len(word) + (1 if buf else 0)
        if buf and size + add > max_chars:
            parts.append(" ".join(buf))
            buf = [word]
            size = len(word)
        else:
            buf.append(word)
            size += add
    if buf:
        parts.append(" ".join(buf))
    return parts
