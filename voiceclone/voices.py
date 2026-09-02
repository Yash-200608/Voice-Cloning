from pathlib import Path

from .Config import VOICES_DIR, ensure_directories
from .core.exceptions import VoiceIdentityNotFound
from .core.service import VoiceIdentityService


def _service() -> VoiceIdentityService:
    ensure_directories()
    return VoiceIdentityService()


def list_voices():
    """List voice names (compatibility API)."""
    return [i.name for i in _service().list_identities()]


def voice_path(name):
    """Return processed reference path for a voice name (compatibility API)."""
    service = _service()
    try:
        identity = service.get_identity_by_name(name)
    except VoiceIdentityNotFound as e:
        raise FileNotFoundError(str(e)) from e
    return str(service.repository.resolve_path(identity, identity.processed_audio))


def delete_voice(name):
    """Delete a voice by name (compatibility API)."""
    service = _service()
    try:
        identity = service.get_identity_by_name(name)
    except VoiceIdentityNotFound:
        return False
    return service.delete_identity(identity.id)
