from .Config import *  # noqa: F401,F403
import os
from pathlib import Path

ROOT = Path(os.environ.get("VOICECLONE_HOME", Path.home() / ".voiceclone"))
VOICES_DIR = ROOT / "voices"
OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "cache"

VOICES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 24000
REFERENCE_SECONDS = 12
TARGET_LUFS = -23.0


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