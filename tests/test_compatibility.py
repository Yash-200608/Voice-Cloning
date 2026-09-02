"""Tests for path-based compatibility APIs."""

import pytest

from voiceclone.core.service import VoiceIdentityService


def test_compatibility_list_and_path(service, synthetic_wav, monkeypatch):
    service.create_from_file("CompatVoice", synthetic_wav)

    monkeypatch.setattr(
        "voiceclone.voices._service",
        lambda: service,
    )

    from voiceclone import list_voices, voice_path

    names = list_voices()
    assert "CompatVoice" in names
    path = voice_path("CompatVoice")
    assert path.endswith("processed/reference.wav")


def test_compatibility_delete(service, synthetic_wav, monkeypatch):
    identity = service.create_from_file("ToDelete", synthetic_wav)
    monkeypatch.setattr("voiceclone.voices._service", lambda: service)

    from voiceclone import delete_voice

    assert delete_voice("ToDelete") is True
    assert not service.repository.identity_dir(identity.id).exists()


def test_compatibility_clone_compare(monkeypatch, service, synthetic_wav, mock_renderer):
    identity = service.create_from_file("CloneCompat", synthetic_wav)
    ref = str(service.repository.resolve_path(identity, identity.processed_audio))

    monkeypatch.setattr(
        "voiceclone.cloner.clone",
        lambda text, voice_file, **kw: mock_renderer.synthesize(text, voice_file, **kw),
    )
    monkeypatch.setattr("voiceclone.similarity.compare", lambda reference, generated: 0.85)

    from voiceclone import clone, clone_best_of, compare

    output = clone("Hello.", ref)
    score = compare(ref, output)
    assert 0.0 <= score <= 1.0

    best, best_score = clone_best_of("Hello again.", ref, n=3)
    assert best_score >= 0.0


def test_compatibility_benchmark(monkeypatch, service, synthetic_wav, mock_renderer):
    identity = service.create_from_file("BenchCompat", synthetic_wav)
    ref = str(service.repository.resolve_path(identity, identity.processed_audio))

    monkeypatch.setattr(
        "voiceclone.benchmarking.clone",
        lambda text, voice_file, **kw: mock_renderer.synthesize(text, voice_file, **kw),
    )
    monkeypatch.setattr("voiceclone.benchmarking._transcribe", lambda path: None)
    monkeypatch.setattr("voiceclone.benchmarking.compare", lambda reference, generated: 0.9)

    from voiceclone.benchmarking import benchmark

    summary = benchmark(ref, sentences=["Test sentence."])
    assert summary["n"] == 1
    assert "similarity_mean" in summary
