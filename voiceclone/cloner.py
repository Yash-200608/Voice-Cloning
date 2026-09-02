import uuid
from pathlib import Path

from .Config import OUTPUTS_DIR, DEVICE, ensure_directories
from .audio_utils import save_audio, normalize_loudness, load_audio
from .text_utils import normalize_text

_model = None


def _get_model():
    global _model
    if _model is None:
        from chatterbox.tts import ChatterboxTTS
        print(f"Loading Chatterbox TTS on {DEVICE} (first run only)...")
        _model = ChatterboxTTS.from_pretrained(device=DEVICE)
    return _model


def clone(text, voice_file, output_path=None, exaggeration=0.5, cfg_weight=0.5, normalize=True):
    if not Path(voice_file).exists():
        raise FileNotFoundError(f"Voice file not found: {voice_file}")

    ensure_directories()

    text = normalize_text(text)
    model = _get_model()

    wav = model.generate(
        text,
        audio_prompt_path=str(voice_file),
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )

    audio = wav.squeeze(0).cpu().numpy()
    sr = model.sr

    if normalize:
        audio = normalize_loudness(audio, sr)

    if output_path is None:
        output_path = OUTPUTS_DIR / f"out_{uuid.uuid4().hex[:8]}.wav"

    save_audio(output_path, audio, sr)
    return str(output_path)


def synthesize_with_settings(
    text,
    voice_file,
    output_path=None,
    exaggeration=0.5,
    cfg_weight=0.5,
    normalize=True,
):
    """Generate speech with explicit renderer settings (used by ChatterboxRenderer)."""
    return clone(
        text,
        voice_file,
        output_path=output_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        normalize=normalize,
    )


def clone_best_of(text, voice_file, n=3, exaggeration=0.5, cfg_weight=0.5):
    from .similarity import compare

    candidates = []
    for i in range(n):
        path = clone(
            text,
            voice_file,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        score = compare(voice_file, path)
        candidates.append((score, path))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = candidates[0]

    for _, path in candidates[1:]:
        try:
            Path(path).unlink()
        except OSError:
            pass

    return best_path, best_score