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
