"""Tests for identity persistence and reload."""

import pytest

from voiceclone.core.service import VoiceIdentityService
from voiceclone.identity.repository import VoiceRepository


def test_identity_survives_restart(service, synthetic_wav, voiceclone_home, mock_renderer, mock_embedding, monkeypatch):
    identity = service.create_from_file("Persist", synthetic_wav)
    identity_id = identity.id
    original_name = identity.name

    from voiceclone.identity.embeddings import EmbeddingStore

    store = EmbeddingStore()

    def fake_validate(self, path, **kwargs):
        return mock_embedding

    monkeypatch.setattr(EmbeddingStore, "validate", fake_validate)

    repo2 = VoiceRepository(
        voices_dir=voiceclone_home / "voices",
        outputs_dir=voiceclone_home / "outputs",
        embedding_store=store,
    )
    service2 = VoiceIdentityService(
        repository=repo2,
        renderer=mock_renderer,
        embedding_store=store,
        auto_migrate=False,
    )

    reloaded = service2.get_identity(identity_id)
    assert reloaded.id == identity_id
    assert reloaded.name == original_name
    assert reloaded.duration_seconds > 0


def test_list_identities_after_reload(service, synthetic_wav, voiceclone_home, mock_renderer, mock_embedding, monkeypatch):
    service.create_from_file("VoiceA", synthetic_wav)
    service.create_from_file("VoiceB", synthetic_wav)

    from voiceclone.identity.embeddings import EmbeddingStore

    store = EmbeddingStore()
    repo2 = VoiceRepository(
        voices_dir=voiceclone_home / "voices",
        outputs_dir=voiceclone_home / "outputs",
        embedding_store=store,
    )
    service2 = VoiceIdentityService(repository=repo2, renderer=mock_renderer, embedding_store=store, auto_migrate=False)
    names = {i.name for i in service2.list_identities()}
    assert names == {"VoiceA", "VoiceB"}
