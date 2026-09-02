"""Tests for synthesis and Best-of-3."""

from pathlib import Path


def test_synthesize_returns_wav(service, synthetic_wav):
    identity = service.create_from_file("Synth", synthetic_wav)
    output = service.synthesize(identity.id, "Hello there.")
    assert Path(output).exists()
    assert Path(output).suffix == ".wav"


def test_best_of_three(service, synthetic_wav):
    identity = service.create_from_file("BestOf", synthetic_wav)
    best_path, score = service.synthesize_best_of(identity.id, "Testing best of three.", n=3)
    assert Path(best_path).exists()
    assert 0.0 <= score <= 1.0

    out_dir = service.repository.output_dir(identity.id)
    wav_files = list(out_dir.glob("*.wav"))
    assert len(wav_files) == 1
