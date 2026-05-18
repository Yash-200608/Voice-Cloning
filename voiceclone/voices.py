from pathlib import Path
from .Config import VOICES_DIR


def list_voices():
    return sorted(p.stem for p in VOICES_DIR.glob("*.wav"))


def voice_path(name):
    path = VOICES_DIR / f"{name}.wav"
    if not path.exists():
        raise FileNotFoundError(f"No voice named '{name}' in {VOICES_DIR}")
    return str(path)


def delete_voice(name):
    path = VOICES_DIR / f"{name}.wav"
    if path.exists():
        path.unlink()
        return True
    return False