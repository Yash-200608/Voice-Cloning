"""Tests for legacy voice migration."""

import hashlib
import shutil

import pytest

from voiceclone.compatibility.legacy import migrate_legacy_voices
from voiceclone.core.service import VoiceIdentityService
from voiceclone.identity.repository import VoiceRepository


def test_legacy_wav_migrated(service, synthetic_wav, voiceclone_home, mock_renderer, mock_embedding, monkeypatch):
    voices_dir = voiceclone_home / "voices"
    legacy_path = voices_dir / "LegacyVoice.wav"
    shutil.copy2(synthetic_wav, legacy_path)
    original_hash = hashlib.sha256(legacy_path.read_bytes()).digest()

    from voiceclone.identity.embeddings import EmbeddingStore

    store = EmbeddingStore()

    def fake_generate(self, processed_audio, output_path):
        import numpy as np
        from pathlib import Path
        vec = mock_embedding
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, vec, allow_pickle=False)
        return vec

    def fake_validate(self, path, **kwargs):
        return mock_embedding

    monkeypatch.setattr(EmbeddingStore, "generate", fake_generate)
    monkeypatch.setattr(EmbeddingStore, "validate", fake_validate)

    repo = VoiceRepository(voices_dir=voices_dir, outputs_dir=voiceclone_home / "outputs", embedding_store=store)
    migrated = migrate_legacy_voices(repo)
    assert len(migrated) == 1

    svc = VoiceIdentityService(repository=repo, renderer=mock_renderer, embedding_store=store, auto_migrate=False)
    identities = svc.list_identities()
    assert any(i.name == "LegacyVoice" for i in identities)
    assert hashlib.sha256(legacy_path.read_bytes()).digest() == original_hash


def test_failed_migration_preserves_legacy(service, synthetic_wav, voiceclone_home, monkeypatch):
    voices_dir = voiceclone_home / "voices"
    legacy_path = voices_dir / "FailVoice.wav"
    shutil.copy2(synthetic_wav, legacy_path)
    original_hash = hashlib.sha256(legacy_path.read_bytes()).digest()

    from voiceclone.identity.embeddings import EmbeddingStore

    def fail_generate(self, processed_audio, output_path):
        raise RuntimeError("Simulated embedding failure")

    monkeypatch.setattr(EmbeddingStore, "generate", fail_generate)

    repo = VoiceRepository(voices_dir=voices_dir, outputs_dir=voiceclone_home / "outputs")
    migrated = migrate_legacy_voices(repo)
    assert migrated == []
    assert legacy_path.exists()
    assert hashlib.sha256(legacy_path.read_bytes()).digest() == original_hash


def test_migration_idempotent(service, synthetic_wav, voiceclone_home, mock_embedding, monkeypatch):
    voices_dir = voiceclone_home / "voices"
    legacy_path = voices_dir / "Once.wav"
    shutil.copy2(synthetic_wav, legacy_path)

    from voiceclone.identity.embeddings import EmbeddingStore

    def fake_generate(self, processed_audio, output_path):
        import numpy as np
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mock_embedding, allow_pickle=False)
        return mock_embedding

    monkeypatch.setattr(EmbeddingStore, "generate", fake_generate)

    repo = VoiceRepository(voices_dir=voices_dir, outputs_dir=voiceclone_home / "outputs")
    first = migrate_legacy_voices(repo)
    second = migrate_legacy_voices(repo)
    assert len(first) == 1
    assert len(second) == 0
