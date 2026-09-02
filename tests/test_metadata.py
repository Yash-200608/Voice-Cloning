"""Tests for metadata validation."""

import pytest

from voiceclone.core.exceptions import InvalidVoiceIdentity
from voiceclone.core.models import VoiceIdentity, utc_now
from voiceclone.identity.metadata import save_metadata, validate_metadata_dict


def test_metadata_requires_relative_paths(tmp_path):
    identity = VoiceIdentity(
        id="voice_00000000-0000-4000-8000-000000000001",
        name="Test",
        created_at=utc_now(),
        updated_at=utc_now(),
        reference_audio="/absolute/raw.wav",
        processed_audio="processed/reference.wav",
        embedding_path="embeddings/speaker.npy",
        embedding_model="resemblyzer",
        embedding_version="0.1.4",
        sample_rate=24000,
        duration_seconds=1.0,
        renderer_model="chatterbox",
        renderer_version="0.1.7",
    )
    with pytest.raises(InvalidVoiceIdentity):
        save_metadata(tmp_path / "metadata.json", identity)


def test_metadata_schema_validation():
    with pytest.raises(InvalidVoiceIdentity):
        validate_metadata_dict({"id": "bad", "name": "x"})
