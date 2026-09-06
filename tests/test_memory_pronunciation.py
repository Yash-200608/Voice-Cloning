"""Tests for safe pronunciation matching."""

import uuid

from voiceclone.memory.models import VoiceMemoryItem, generate_memory_id
from voiceclone.memory.pronunciation import apply_pronunciation_memories, find_pronunciation_matches


def _mem(key: str, spoken: str, category: str = "pronunciation") -> VoiceMemoryItem:
    return VoiceMemoryItem(
        id=generate_memory_id(),
        identity_id=f"voice_{uuid.uuid4()}",
        category=category,
        key=key,
        value={"pronunciation_type": "custom_text", "pronunciation_value": spoken},
    )


def test_exact_word_match():
    mem = _mem("Minitorch", "mini-torch")
    text, ids = apply_pronunciation_memories("Minitorch is ready.", [mem])
    assert "mini-torch" in text
    assert ids == [mem.id]


def test_exact_phrase_match():
    mem = _mem("neural net", "neural-net")
    text, ids = apply_pronunciation_memories("A neural net helps.", [mem])
    assert "neural-net" in text
    assert ids


def test_case_variation():
    mem = _mem("Minitorch", "mini-torch")
    text, _ = apply_pronunciation_memories("minitorch rocks.", [mem])
    assert text.lower().startswith("mini-torch")


def test_punctuation():
    mem = _mem("Minitorch", "mini-torch")
    text, _ = apply_pronunciation_memories("Hello, Minitorch!", [mem])
    assert "mini-torch" in text


def test_repeated_occurrence():
    mem = _mem("Minitorch", "mini-torch")
    text, ids = apply_pronunciation_memories("Minitorch and Minitorch again.", [mem])
    assert text.count("mini-torch") == 2
    assert ids == [mem.id]


def test_similar_substring_should_not_match():
    mem = _mem("AI", "A I")
    text, ids = apply_pronunciation_memories("He said nothing.", [mem])
    assert text == "He said nothing."
    assert ids == []
    assert find_pronunciation_matches("He said nothing.", [mem]) == []


def test_disabled_memory_ignored_by_matcher_input():
    mem = _mem("Minitorch", "mini-torch")
    mem = mem.with_updates(enabled=False)
    text, ids = apply_pronunciation_memories("Minitorch is ready.", [mem])
    assert text == "Minitorch is ready."
    assert ids == []
