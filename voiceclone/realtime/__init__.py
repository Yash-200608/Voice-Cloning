"""Real-time voice package (Phase 5).

Honest capability: chunked generation + buffered playback.
Chatterbox does not expose native token/waveform streaming.
"""

from .models import (
    STREAMING_MODE,
    AudioChunk,
    RealtimeSession,
    SessionMetrics,
    SessionState,
    generate_session_id,
)
from .service import RealTimeVoiceService

__all__ = [
    "STREAMING_MODE",
    "AudioChunk",
    "RealtimeSession",
    "SessionMetrics",
    "SessionState",
    "generate_session_id",
    "RealTimeVoiceService",
]
