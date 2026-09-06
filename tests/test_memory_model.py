"""Tests for VoiceMemoryItem validation."""

import math
import uuid

import pytest

from voiceclone.core.exceptions import InvalidVoiceMemory
from voiceclone.memory.models import VoiceMemoryItem, generate_memory_id


def _voice_id() -> str:
    return f"voice_{uuid.uuid4()}"


def _item(**overrides):
    data = {
        "id": generate_memory_id(),
        "identity_id": _voice_id(),
        "category": "pronunciation",
        "key": "Minitorch",
        "value": {"pronunciation_type": "custom_text", "pronunciation_value": "mini-torch"},
        "confidence": 1.0,
        "source": "explicit_user",
    }
    data.update(overrides)
    return VoiceMemoryItem(**data)


def test_valid_pronunciation_memory():
    item = _item()
    assert item.category == "pronunciation"
    assert item.version == 1
    assert item.enabled is True


def test_invalid_category():
    with pytest.raises(InvalidVoiceMemory):
        _item(category="chat_history")


def test_invalid_confidence_nan():
    with pytest.raises(InvalidVoiceMemory):
        _item(confidence=float("nan"))


def test_invalid_confidence_inf():
    with pytest.raises(InvalidVoiceMemory):
        _item(confidence=float("inf"))


def test_invalid_confidence_range():
    with pytest.raises(InvalidVoiceMemory):
        _item(confidence=1.5)


def test_malformed_memory_id():
    with pytest.raises(InvalidVoiceMemory):
        _item(id="not-a-memory-id")


def test_malformed_identity_id():
    with pytest.raises(InvalidVoiceMemory):
        _item(identity_id="bad")


def test_empty_key():
    with pytest.raises(InvalidVoiceMemory):
        _item(key="  ")


def test_invalid_version():
    with pytest.raises(InvalidVoiceMemory):
        _item(version=0)


def test_style_preference_requires_dimension():
    with pytest.raises(InvalidVoiceMemory):
        _item(
            category="style_preference",
            key="speaking_rate",
            value={"preference": 0.7},
        )


def test_with_updates_increments_version():
    item = _item()
    updated = item.with_updates(confidence=0.8)
    assert updated.version == item.version + 1
    assert updated.confidence == 0.8
    assert updated.id == item.id
