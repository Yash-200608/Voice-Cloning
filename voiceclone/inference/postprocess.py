"""Post-processing for expressive delivery dimensions."""

from __future__ import annotations

import numpy as np

from ..inference.chatterbox_mapping import ChatterboxRenderSettings


def apply_expression_postprocess(
    audio: np.ndarray,
    sr: int,
    settings: ChatterboxRenderSettings,
) -> np.ndarray:
    """Apply rate/pitch adjustments derived from expression mapping."""
    result = audio.astype(np.float32)

    if abs(settings.speaking_rate_factor - 1.0) > 0.01:
        try:
            import librosa
            result = librosa.effects.time_stretch(result, rate=settings.speaking_rate_factor)
        except Exception:
            pass

    if abs(settings.pitch_semitones) > 0.05:
        try:
            import librosa
            result = librosa.effects.pitch_shift(result, sr=sr, n_steps=settings.pitch_semitones)
        except Exception:
            pass

    return result
