"""Tests for error handling."""

import json
from pathlib import Path

import pytest

from voiceclone.core.exceptions import (
    InvalidVoiceIdentity,
    UnsupportedIdentityVersion,
    VoiceIdentityNotFound,
)
from voiceclone.core.service import VoiceIdentityService
from voiceclone.identity.metadata import load_metadata


def test_nonexistent_identity(service):
    with pytest.raises(VoiceIdentityNotFound):
        service.get_identity("voice_00000000-0000-0000-0000-000000000000")


def test_corrupt_metadata(service, synthetic_wav, voiceclone_home):
    identity = service.create_from_file("Corrupt", synthetic_wav)
    meta_path = service.repository.identity_dir(identity.id) / "metadata.json"
    meta_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(InvalidVoiceIdentity):
        service.get_identity(identity.id)


def test_missing_files(service, synthetic_wav):
    identity = service.create_from_file("Missing", synthetic_wav)
    processed = service.repository.resolve_path(identity, identity.processed_audio)
    processed.unlink()

    with pytest.raises(InvalidVoiceIdentity):
        service.repository.ensure_files_exist(service.get_identity(identity.id))


def test_unsupported_schema_version(service, synthetic_wav):
    identity = service.create_from_file("OldSchema", synthetic_wav)
    meta_path = service.repository.identity_dir(identity.id) / "metadata.json"
    data = json.loads(meta_path.read_text())
    data["schema_version"] = 999
    meta_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(UnsupportedIdentityVersion):
        load_metadata(meta_path)
