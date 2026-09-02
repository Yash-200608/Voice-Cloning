"""Audio capture and import primitives."""

from pathlib import Path
import shutil
import tempfile

from .Config import SAMPLE_RATE, REFERENCE_SECONDS, ensure_directories
from .audio_utils import save_audio, load_audio


def capture_to_path(
    output_path: Path,
    duration: float = REFERENCE_SECONDS,
    sr: int = SAMPLE_RATE,
) -> str:
    """Record audio from microphone to a destination path (raw reference)."""
    import sounddevice as sd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
    except sd.PortAudioError as e:
        raise RuntimeError(
            f"Microphone unavailable: {e}. "
            "Check that a mic is connected and not in use by another app."
        )

    audio = audio.flatten().astype(np.float32)
    save_audio(output_path, audio, sr)
    return str(output_path)


def import_to_path(source_path: Path, output_path: Path, sr: int = SAMPLE_RATE) -> str:
    """
    Import audio/video source to a raw reference WAV without preprocessing.

    The source file is never modified.
    """
    source = Path(source_path)
    output_path = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(f"Reference file not found: {source}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    audio_exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".wma"}

    if suffix in audio_exts:
        audio, file_sr = load_audio(source, sr=sr)
        save_audio(output_path, audio, file_sr)
        return str(output_path)

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
                .output(str(extracted), ac=1, ar=sr, format="wav")
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

        shutil.copy2(extracted, output_path)
        return str(output_path)
    finally:
        try:
            extracted.unlink(missing_ok=True)
        except OSError:
            pass


def record(name, duration=REFERENCE_SECONDS, sr=SAMPLE_RATE, preprocess=True):
    """Compatibility API: record and create a voice identity."""
    from .core.service import VoiceIdentityService

    ensure_directories()
    service = VoiceIdentityService()
    identity = service.create_from_recording(name, duration=duration, sr=sr)
    return str(service.repository.resolve_path(identity, identity.processed_audio))


def import_reference(name, source_path, preprocess=True):
    """Compatibility API: import and create a voice identity."""
    from .core.service import VoiceIdentityService

    ensure_directories()
    service = VoiceIdentityService()
    identity = service.create_from_file(name, source_path)
    return str(service.repository.resolve_path(identity, identity.processed_audio))
