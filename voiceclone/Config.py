import os
from pathlib import Path

ROOT = Path(os.environ.get("VOICECLONE_HOME", Path.home() / ".voiceclone"))
VOICES_DIR = ROOT / "voices"
OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "cache"

SAMPLE_RATE = 24000
REFERENCE_SECONDS = 12
TARGET_LUFS = -23.0


def ensure_directories() -> None:
    """Create required storage directories if they do not exist."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


DEVICE = get_device()
