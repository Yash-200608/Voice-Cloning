from .audio_utils import preprocess_reference, normalize_loudness
from .text_utils import normalize_text
from .core.models import VoiceIdentity
from .core.service import VoiceIdentityService
from .core.exceptions import (
    VoiceCloneError,
    VoiceIdentityNotFound,
    InvalidVoiceIdentity,
    MissingReferenceAudio,
    MissingEmbedding,
    EmbeddingGenerationError,
    VoiceRepositoryError,
    UnsupportedIdentityVersion,
    VoiceRenderError,
)


def __getattr__(name: str):
    if name in ("record", "import_reference"):
        from .recorder import record, import_reference
        return {"record": record, "import_reference": import_reference}[name]
    if name in ("clone", "clone_best_of"):
        from .cloner import clone, clone_best_of
        return {"clone": clone, "clone_best_of": clone_best_of}[name]
    if name in ("compare", "embed", "compare_with_embedding"):
        from .similarity import compare, embed, compare_with_embedding
        return {"compare": compare, "embed": embed, "compare_with_embedding": compare_with_embedding}[name]
    if name in ("list_voices", "voice_path", "delete_voice"):
        from .voices import list_voices, voice_path, delete_voice
        return {"list_voices": list_voices, "voice_path": voice_path, "delete_voice": delete_voice}[name]
    if name == "benchmark":
        from .benchmarking import benchmark
        return benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "record",
    "import_reference",
    "clone",
    "clone_best_of",
    "compare",
    "embed",
    "compare_with_embedding",
    "list_voices",
    "voice_path",
    "delete_voice",
    "benchmark",
    "preprocess_reference",
    "normalize_loudness",
    "normalize_text",
    "VoiceIdentity",
    "VoiceIdentityService",
    "VoiceCloneError",
    "VoiceIdentityNotFound",
    "InvalidVoiceIdentity",
    "MissingReferenceAudio",
    "MissingEmbedding",
    "EmbeddingGenerationError",
    "VoiceRepositoryError",
    "UnsupportedIdentityVersion",
    "VoiceRenderError",
]
