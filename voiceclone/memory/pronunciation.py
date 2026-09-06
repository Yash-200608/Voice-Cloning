"""Safe pronunciation matching for voice memory."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import VoiceMemoryItem

# Whole-word/phrase boundaries: do not match inside larger alphanumeric tokens.
# Example: key "AI" must not match inside "said".
WORD_BOUNDARY = r"(?<![\w'])(?:{term})(?![\w'])"


@dataclass(frozen=True)
class PronunciationMatch:
    memory_id: str
    key: str
    start: int
    end: int
    replacement: str


def _spoken_form(item: VoiceMemoryItem) -> str:
    return str(item.value["pronunciation_value"]).strip()


def find_pronunciation_matches(
    text: str,
    memories: list[VoiceMemoryItem],
) -> list[PronunciationMatch]:
    """Find non-overlapping whole-word/phrase pronunciation matches.

    Matching rules:
    - Case-insensitive
    - Longer keys preferred when spans overlap
    - Word/phrase boundaries required (no substring false positives)
    - Disabled memories are ignored by the caller
    """
    candidates: list[PronunciationMatch] = []
    pronunciation_memories = [
        item
        for item in memories
        if item.enabled and item.category in {"pronunciation", "name", "vocabulary"}
    ]
    pronunciation_memories.sort(key=lambda item: len(item.key), reverse=True)

    for item in pronunciation_memories:
        key = item.key.strip()
        if not key:
            continue
        pattern = re.compile(WORD_BOUNDARY.format(term=re.escape(key)), re.IGNORECASE)
        for match in pattern.finditer(text):
            candidates.append(
                PronunciationMatch(
                    memory_id=item.id,
                    key=item.key,
                    start=match.start(),
                    end=match.end(),
                    replacement=_spoken_form(item),
                )
            )

    candidates.sort(key=lambda match: (match.start, -(match.end - match.start)))
    selected: list[PronunciationMatch] = []
    occupied: list[tuple[int, int]] = []
    for candidate in candidates:
        if any(not (candidate.end <= start or candidate.start >= end) for start, end in occupied):
            continue
        selected.append(candidate)
        occupied.append((candidate.start, candidate.end))
    selected.sort(key=lambda match: match.start)
    return selected


def apply_pronunciation_memories(
    text: str,
    memories: list[VoiceMemoryItem],
) -> tuple[str, list[str]]:
    """Apply pronunciation memories without mutating the caller-owned original text."""
    matches = find_pronunciation_matches(text, memories)
    if not matches:
        return text, []
    parts: list[str] = []
    cursor = 0
    used_ids: list[str] = []
    for match in matches:
        parts.append(text[cursor : match.start])
        parts.append(match.replacement)
        cursor = match.end
        used_ids.append(match.memory_id)
    parts.append(text[cursor:])
    # Preserve first-applied order uniqueness while keeping encounter order.
    unique_ids = list(dict.fromkeys(used_ids))
    return "".join(parts), unique_ids
