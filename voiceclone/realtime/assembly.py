"""Assemble ordered AudioChunks into a contiguous waveform / WAV file."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..Config import SAMPLE_RATE
from ..audio_utils import save_audio
from ..core.exceptions import PlaybackError
from .models import AudioChunk


def assemble_chunks(
    chunks: list[AudioChunk],
    *,
    expected_sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Concatenate chunks in sequence order with integrity checks."""
    if not chunks:
        return np.zeros(0, dtype=np.float32), expected_sample_rate

    ordered = sorted(chunks, key=lambda c: c.sequence)
    seen: set[int] = set()
    parts: list[np.ndarray] = []
    sample_rate = ordered[0].sample_rate

    for chunk in ordered:
        if chunk.sequence in seen:
            raise PlaybackError(
                f"Duplicate chunk sequence {chunk.sequence} in session {chunk.session_id}",
                user_message="Generated audio could not be assembled (duplicate chunk).",
            )
        seen.add(chunk.sequence)
        if chunk.sample_rate != sample_rate:
            raise PlaybackError(
                f"Sample rate mismatch: chunk {chunk.sequence} has {chunk.sample_rate}, "
                f"expected {sample_rate}",
                user_message="Generated audio could not be assembled (format mismatch).",
            )
        if chunk.sample_rate != expected_sample_rate:
            raise PlaybackError(
                f"Unexpected sample rate {chunk.sample_rate}, expected {expected_sample_rate}",
                user_message="Generated audio could not be assembled (sample rate mismatch).",
            )
        samples = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            continue
        parts.append(samples)

    # Detect gaps (missing sequences) only among non-empty collected sequences.
    if seen:
        expected = set(range(min(seen), max(seen) + 1))
        missing = expected - seen
        if missing:
            raise PlaybackError(
                f"Missing chunk sequences: {sorted(missing)}",
                user_message="Generated audio could not be assembled (missing chunk).",
            )

    if not parts:
        return np.zeros(0, dtype=np.float32), sample_rate
    return np.concatenate(parts), sample_rate


def save_assembled_wav(
    chunks: list[AudioChunk],
    output_path: str | Path,
    *,
    expected_sample_rate: int = SAMPLE_RATE,
) -> tuple[str, float]:
    """Assemble chunks and write a WAV. Returns (path, duration_seconds)."""
    audio, sr = assemble_chunks(chunks, expected_sample_rate=expected_sample_rate)
    path = Path(output_path)
    save_audio(path, audio, sr)
    duration = float(audio.shape[0]) / float(sr) if audio.size else 0.0
    return str(path), duration
