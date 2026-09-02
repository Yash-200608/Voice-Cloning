"""Shared pytest fixtures."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Ensure VOICECLONE_HOME is set before voiceclone imports
TESTS_ROOT = Path(__file__).parent


@pytest.fixture
def voiceclone_home(tmp_path, monkeypatch):
    home = tmp_path / ".voiceclone"
    monkeypatch.setenv("VOICECLONE_HOME", str(home))
    # Reload config paths
    import voiceclone.Config as config
    config.ROOT = home
    config.VOICES_DIR = home / "voices"
    config.OUTPUTS_DIR = home / "outputs"
    config.CACHE_DIR = home / "cache"
    return home


@pytest.fixture
def synthetic_wav(tmp_path):
    path = tmp_path / "reference.wav"
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def mock_embedding():
    vec = np.random.randn(256).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


@pytest.fixture
def mock_renderer(tmp_path):
    class FakeRenderer:
        def synthesize(self, text, reference_audio, output_path=None, **kwargs):
            import uuid
            out = Path(output_path) if output_path else tmp_path / f"out_{uuid.uuid4().hex[:8]}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sr = 24000
            t = np.linspace(0, 0.5, sr // 2, endpoint=False)
            audio = 0.2 * np.sin(2 * np.pi * 330 * t).astype(np.float32)
            sf.write(str(out), audio, sr)
            return str(out)

    return FakeRenderer()


@pytest.fixture
def service(voiceclone_home, mock_renderer, mock_embedding, monkeypatch):
    from voiceclone.core.service import VoiceIdentityService
    from voiceclone.identity.embeddings import EmbeddingStore
    from voiceclone.identity.repository import VoiceRepository

    store = EmbeddingStore()

    def fake_generate(self, processed_audio, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, mock_embedding, allow_pickle=False)
        return mock_embedding

    def fake_validate(self, path, **kwargs):
        from voiceclone.core.exceptions import InvalidVoiceIdentity, MissingEmbedding

        path = Path(path)
        if not path.exists():
            raise MissingEmbedding(f"Embedding not found: {path}")
        vector = np.load(path, allow_pickle=False)
        self._validate_vector(vector)
        if kwargs.get("embedding_model") and kwargs["embedding_model"] != self.embedding_model:
            raise InvalidVoiceIdentity(f"Embedding model mismatch: {kwargs['embedding_model']}")
        return vector

    monkeypatch.setattr(EmbeddingStore, "generate", fake_generate)
    monkeypatch.setattr(EmbeddingStore, "validate", fake_validate)

    def fake_compare_with_embedding(ref_embedding, generated_path):
        return float(np.dot(ref_embedding, mock_embedding))

    monkeypatch.setattr(
        "voiceclone.similarity.compare_with_embedding",
        fake_compare_with_embedding,
    )

    repo = VoiceRepository(
        voices_dir=voiceclone_home / "voices",
        outputs_dir=voiceclone_home / "outputs",
        embedding_store=store,
    )
    return VoiceIdentityService(
        repository=repo,
        renderer=mock_renderer,
        embedding_store=store,
        auto_migrate=False,
    )
