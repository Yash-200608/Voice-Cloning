"""Tests for embedding cache behavior."""

from pathlib import Path

import numpy as np
import pytest

from voiceclone.core.exceptions import InvalidVoiceIdentity
from voiceclone.identity.embeddings import EmbeddingStore


def test_cached_embedding_reused_in_compare(service, synthetic_wav, mock_embedding, monkeypatch):
    identity = service.create_from_file("CacheTest", synthetic_wav)
    output = service.synthesize(identity.id, "Hello world.")

    validate_calls = {"n": 0}
    original_validate = EmbeddingStore.validate

    def counting_validate(self, path, **kwargs):
        validate_calls["n"] += 1
        vector = np.load(path, allow_pickle=False)
        self._validate_vector(vector)
        return vector

    monkeypatch.setattr(EmbeddingStore, "validate", counting_validate)

    score1 = service.compare(identity.id, output)
    score2 = service.compare(identity.id, output)
    assert 0.0 <= score1 <= 1.0
    assert validate_calls["n"] >= 2


def test_missing_embedding_triggers_regeneration(service, synthetic_wav, mock_embedding, monkeypatch):
    identity = service.create_from_file("Rebuild", synthetic_wav)
    emb_path = service.repository.resolve_path(identity, identity.embedding_path)
    emb_path.unlink()

    regen_called = {"n": 0}

    def tracking_generate(self, processed_audio, output_path):
        regen_called["n"] += 1
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mock_embedding, allow_pickle=False)
        return mock_embedding

    monkeypatch.setattr(EmbeddingStore, "generate", tracking_generate)

    output = service.synthesize(identity.id, "Test.")
    service.compare(identity.id, output)
    assert regen_called["n"] >= 1
    assert emb_path.exists()


def test_invalid_embedding_detected(service, synthetic_wav):
    identity = service.create_from_file("Invalid", synthetic_wav)
    emb_path = service.repository.resolve_path(identity, identity.embedding_path)
    np.save(emb_path, np.zeros(256), allow_pickle=False)

    with pytest.raises(InvalidVoiceIdentity):
        service.embedding_store.validate(emb_path)


def test_version_mismatch_triggers_rebuild(service, synthetic_wav, monkeypatch):
    identity = service.create_from_file("Version", synthetic_wav)
    identity.embedding_version = "0.0.0-old"
    from voiceclone.identity.metadata import save_metadata
    save_metadata(service.repository.identity_dir(identity.id) / "metadata.json", identity)

    rebuilt = service.rebuild_embedding(identity.id)
    assert rebuilt.embedding_version != "0.0.0-old"
