"""Tests for voice identity creation and directory structure."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest


def test_create_identity_directory_structure(service, synthetic_wav):
    identity = service.create_from_file("TestVoice", synthetic_wav)
    root = service.repository.identity_dir(identity.id)

    assert root.exists()
    assert (root / "metadata.json").exists()
    assert (root / "raw/reference.wav").exists()
    assert (root / "processed/reference.wav").exists()
    assert (root / "embeddings/speaker.npy").exists()


def test_raw_and_processed_are_separate(service, synthetic_wav):
    identity = service.create_from_file("TestVoice", synthetic_wav)
    root = service.repository.identity_dir(identity.id)
    raw = (root / "raw/reference.wav").read_bytes()
    processed = (root / "processed/reference.wav").read_bytes()
    assert raw != processed


def test_stable_voice_id_in_metadata(service, synthetic_wav):
    identity = service.create_from_file("TestVoice", synthetic_wav)
    assert re.match(r"^voice_[0-9a-f-]{36}$", identity.id)

    with open(service.repository.identity_dir(identity.id) / "metadata.json") as f:
        meta = json.load(f)
    assert meta["id"] == identity.id
    assert meta["id"] == identity.id == service.repository.identity_dir(identity.id).name


def test_external_source_not_modified(service, synthetic_wav, tmp_path):
    source = tmp_path / "external.wav"
    source.write_bytes(synthetic_wav.read_bytes())
    original_hash = hashlib.sha256(source.read_bytes()).digest()
    service.create_from_file("External", source)
    assert hashlib.sha256(source.read_bytes()).digest() == original_hash


def test_embedding_created_and_valid(service, synthetic_wav):
    identity = service.create_from_file("TestVoice", synthetic_wav)
    emb_path = service.repository.resolve_path(identity, identity.embedding_path)
    vector = service.embedding_store.validate(
        emb_path,
        embedding_model=identity.embedding_model,
        embedding_version=identity.embedding_version,
    )
    assert vector.shape == (256,)
    assert np.all(np.isfinite(vector))
