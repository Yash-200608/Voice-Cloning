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


def compare(reference, generated):
    e1 = embed(reference)
    e2 = embed(generated)
    similarity = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    return similarity