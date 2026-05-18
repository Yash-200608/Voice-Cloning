import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

from .Config import SAMPLE_RATE, TARGET_LUFS


def load_audio(path, sr=SAMPLE_RATE):
    audio, file_sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    return audio.astype(np.float32), sr


def save_audio(path, audio, sr=SAMPLE_RATE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def trim_silence(audio, top_db=30):
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def denoise(audio, sr=SAMPLE_RATE):
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.75)
    except ImportError:
        return audio


def normalize_loudness(audio, sr=SAMPLE_RATE, target_lufs=TARGET_LUFS):
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if np.isfinite(loudness):
            return pyln.normalize.loudness(audio, loudness, target_lufs)
    except ImportError:
        pass
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio * (0.7 / peak)
    return audio


def preprocess_reference(in_path, out_path=None, max_seconds=None):
    audio, sr = load_audio(in_path)
    audio = denoise(audio, sr)
    audio = trim_silence(audio)
    audio = normalize_loudness(audio, sr)

    if max_seconds is not None:
        max_samples = int(max_seconds * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

    if out_path is None:
        out_path = Path(in_path).with_suffix(".clean.wav")
    save_audio(out_path, audio, sr)
    return str(out_path)