import numpy as np
from pathlib import Path

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        print("Loading Resemblyzer voice encoder (first run only)...")
        _encoder = VoiceEncoder(verbose=False)
    return _encoder


def embed(audio_path):
    from resemblyzer import preprocess_wav
    encoder = _get_encoder()
    wav = preprocess_wav(Path(audio_path))
    return encoder.embed_utterance(wav)


def compare_with_embedding(reference_embedding: np.ndarray, generated_path: str | Path) -> float:
    """Compare a precomputed reference embedding against generated audio."""
    e2 = embed(str(generated_path))
    similarity = float(
        np.dot(reference_embedding, e2)
        / (np.linalg.norm(reference_embedding) * np.linalg.norm(e2))
    )
    return float(np.clip(similarity, -1.0, 1.0))


def compare(reference, generated):
    e1 = embed(reference)
    return compare_with_embedding(e1, generated)
