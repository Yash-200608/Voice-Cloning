"""Domain exceptions for the Voice Identity Engine."""


class VoiceCloneError(Exception):
    """Base exception for voiceclone domain errors."""

    def __init__(self, message: str, *, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or message


class VoiceIdentityNotFound(VoiceCloneError):
    """Raised when a voice identity cannot be found."""


class InvalidVoiceIdentity(VoiceCloneError):
    """Raised when identity metadata or files are invalid."""


class MissingReferenceAudio(VoiceCloneError):
    """Raised when reference audio is missing."""


class MissingEmbedding(VoiceCloneError):
    """Raised when a required embedding is missing."""


class EmbeddingGenerationError(VoiceCloneError):
    """Raised when embedding generation fails."""


class VoiceRepositoryError(VoiceCloneError):
    """Raised when repository operations fail."""


class UnsupportedIdentityVersion(VoiceCloneError):
    """Raised when identity schema version is unsupported."""


class VoiceRenderError(VoiceCloneError):
    """Raised when speech synthesis fails."""


class InvalidExpression(VoiceCloneError):
    """Raised when expression profile data is invalid."""


class UnknownExpressionPreset(VoiceCloneError):
    """Raised when a named expression preset is not found."""


class InvalidContext(VoiceCloneError):
    """Raised when context profile data is invalid."""


class UnknownContextPreset(VoiceCloneError):
    """Raised when a named context preset is not found."""


class InvalidVoiceMemory(VoiceCloneError):
    """Raised when voice memory data is invalid."""


class VoiceMemoryNotFound(VoiceCloneError):
    """Raised when a voice memory item cannot be found."""


class VoiceMemoryStoreError(VoiceCloneError):
    """Raised when voice memory persistence fails."""


class RealtimeSessionError(VoiceCloneError):
    """Raised when a real-time voice session fails."""


class PlaybackError(VoiceCloneError):
    """Raised when audio playback or chunk assembly fails."""


class AudioDeviceError(VoiceCloneError):
    """Raised when the local audio output device is unavailable."""


class StreamingUnavailableError(VoiceCloneError):
    """Raised when a requested streaming mode is not supported by the backend."""


class SessionCancelledError(VoiceCloneError):
    """Raised when a real-time session was cancelled or interrupted."""


class BufferUnderrunError(VoiceCloneError):
    """Raised when playback underruns beyond an explicit hard threshold."""


class SessionNotFound(VoiceCloneError):
    """Raised when a real-time session ID cannot be found."""

