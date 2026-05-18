import sounddevice as sd
import numpy as np
from pathlib import Path
import tempfile

from .Config import VOICES_DIR, SAMPLE_RATE, REFERENCE_SECONDS
from .audio_utils import save_audio, preprocess_reference, load_audio


def record(name, duration=REFERENCE_SECONDS, sr=SAMPLE_RATE, preprocess=True):
    path = VOICES_DIR / f"{name}.wav"

    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
    except sd.PortAudioError as e:
        raise RuntimeError(
            f"Microphone unavailable: {e}. "
            "Check that a mic is connected and not in use by another app."
        )

    audio = audio.flatten().astype(np.float32)
    save_audio(path, audio, sr)

    if preprocess:
        preprocess_reference(path, out_path=path)

    return str(path)


def import_reference(name, source_path, preprocess=True):
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Reference file not found: {source}")

    target = VOICES_DIR / f"{name}.wav"
    suffix = source.suffix.lower()
    audio_exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".wma"}

    if suffix in audio_exts:
        if preprocess:
            preprocess_reference(source, out_path=target, max_seconds=REFERENCE_SECONDS)
        else:
            audio, sr = load_audio(source, sr=SAMPLE_RATE)
            max_samples = int(REFERENCE_SECONDS * sr)
            save_audio(target, audio[:max_samples], sr)
        return str(target)

    # For video/unknown containers, extract mono wav first then preprocess.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        extracted = Path(tmp.name)
    try:
        try:
            import ffmpeg
        except ImportError as e:
            raise RuntimeError(
                "ffmpeg-python is required for importing video files. "
                "Install it with: pip install ffmpeg-python"
            ) from e

        try:
            (
                ffmpeg
                .input(str(source))
                .output(str(extracted), ac=1, ar=SAMPLE_RATE, format="wav")
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            details = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
            raise RuntimeError(
                "Failed to extract audio from reference file. "
                "Make sure ffmpeg is installed and the file is valid.\n"
                f"{details}"
            ) from e

        preprocess_reference(extracted, out_path=target, max_seconds=REFERENCE_SECONDS)
        return str(target)
    finally:
        try:
            extracted.unlink(missing_ok=True)
        except OSError:
            pass