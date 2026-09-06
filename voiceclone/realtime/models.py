"""Real-time voice session models and metrics.

Honest capability label: chunked_generation (application-layer incremental
synthesis + buffered playback). Chatterbox does not expose native streaming.
"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..Config import SAMPLE_RATE


class SessionState(str, enum.Enum):
    """Explicit real-time session lifecycle states."""

    CREATED = "created"
    PREPARING = "preparing"
    GENERATING = "generating"
    PLAYING = "playing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_STATES = frozenset(
    {SessionState.COMPLETED, SessionState.CANCELLED, SessionState.FAILED}
)

# Honest capability label for this backend generation mode.
STREAMING_MODE = "chunked_generation"


def generate_session_id() -> str:
    return f"rtsession_{uuid.uuid4()}"


@dataclass(frozen=True)
class AudioChunk:
    """Backend-independent audio fragment for a real-time session."""

    session_id: str
    sequence: int
    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE
    channels: int = 1
    dtype: str = "float32"
    is_final: bool = False
    text: str = ""
    created_at: float = field(default_factory=time.perf_counter)

    @property
    def duration_seconds(self) -> float:
        if self.samples.size == 0 or self.sample_rate <= 0:
            return 0.0
        frames = self.samples.shape[0] if self.samples.ndim > 0 else 0
        return float(frames) / float(self.sample_rate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "dtype": self.dtype,
            "is_final": self.is_final,
            "text": self.text,
            "duration_seconds": round(self.duration_seconds, 4),
            "frames": int(self.samples.shape[0]) if self.samples.size else 0,
        }


@dataclass
class SessionMetrics:
    """Latency and throughput instrumentation for one session."""

    request_time: float | None = None
    prepare_start: float | None = None
    prepare_end: float | None = None
    inference_start: float | None = None
    first_audio_ready: float | None = None
    playback_start: float | None = None
    completion_time: float | None = None
    cancel_requested_at: float | None = None
    cancel_completed_at: float | None = None
    chunk_count: int = 0
    underrun_count: int = 0
    generated_audio_seconds: float = 0.0
    streaming_mode: str = STREAMING_MODE
    error: str | None = None

    def mark_request(self) -> None:
        self.request_time = time.perf_counter()

    def mark_prepare_start(self) -> None:
        self.prepare_start = time.perf_counter()

    def mark_prepare_end(self) -> None:
        self.prepare_end = time.perf_counter()

    def mark_inference_start(self) -> None:
        if self.inference_start is None:
            self.inference_start = time.perf_counter()

    def mark_first_audio(self) -> None:
        if self.first_audio_ready is None:
            self.first_audio_ready = time.perf_counter()

    def mark_playback_start(self) -> None:
        if self.playback_start is None:
            self.playback_start = time.perf_counter()

    def mark_complete(self) -> None:
        self.completion_time = time.perf_counter()

    def mark_cancel_requested(self) -> None:
        if self.cancel_requested_at is None:
            self.cancel_requested_at = time.perf_counter()

    def mark_cancel_completed(self) -> None:
        self.cancel_completed_at = time.perf_counter()

    def _delta_ms(self, start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return round((end - start) * 1000.0, 3)

    @property
    def ttfa_ms(self) -> float | None:
        """Time to first audio ready (from request)."""
        return self._delta_ms(self.request_time, self.first_audio_ready)

    @property
    def ttfp_ms(self) -> float | None:
        """Time to first playback start (from request)."""
        return self._delta_ms(self.request_time, self.playback_start)

    @property
    def total_generation_ms(self) -> float | None:
        end = self.completion_time or self.cancel_completed_at
        return self._delta_ms(self.inference_start or self.request_time, end)

    @property
    def cancel_latency_ms(self) -> float | None:
        return self._delta_ms(self.cancel_requested_at, self.cancel_completed_at)

    @property
    def rtf(self) -> float | None:
        """Real-time factor: generation_time / audio_duration."""
        if not self.generated_audio_seconds:
            return None
        total_ms = self.total_generation_ms
        if total_ms is None:
            return None
        return round((total_ms / 1000.0) / self.generated_audio_seconds, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "streaming_mode": self.streaming_mode,
            "ttfa_ms": self.ttfa_ms,
            "ttfp_ms": self.ttfp_ms,
            "total_generation_ms": self.total_generation_ms,
            "cancel_latency_ms": self.cancel_latency_ms,
            "generated_audio_seconds": round(self.generated_audio_seconds, 4),
            "chunk_count": self.chunk_count,
            "underrun_count": self.underrun_count,
            "rtf": self.rtf,
            "error": self.error,
        }


@dataclass
class RealtimeSession:
    """One active real-time speech operation."""

    session_id: str
    identity_id: str
    text: str
    state: SessionState = SessionState.CREATED
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    expression_name: str | None = None
    context_name: str | None = None
    use_memory: bool = True
    memory_item_ids: tuple[str, ...] = ()
    chunking_strategy: str = "sentence"
    final_audio_path: str | None = None
    error_message: str | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def transition(self, new_state: SessionState) -> None:
        with self._lock:
            current = self.state
            if current in TERMINAL_STATES and new_state != current:
                raise ValueError(
                    f"Cannot transition real-time session from terminal state "
                    f"{current} to {new_state}"
                )
            allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
            if new_state not in allowed and new_state != current:
                raise ValueError(
                    f"Invalid real-time session transition: {current} → {new_state}"
                )
            self.state = new_state

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "identity_id": self.identity_id,
            "state": self.state.value,
            "expression_name": self.expression_name,
            "context_name": self.context_name,
            "use_memory": self.use_memory,
            "memory_item_ids": list(self.memory_item_ids),
            "chunking_strategy": self.chunking_strategy,
            "streaming_mode": STREAMING_MODE,
            "final_audio_path": self.final_audio_path,
            "error_message": self.error_message,
            "metrics": self.metrics.to_dict(),
        }


_ALLOWED_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.CREATED: frozenset(
        {SessionState.PREPARING, SessionState.CANCELLED, SessionState.FAILED}
    ),
    SessionState.PREPARING: frozenset(
        {SessionState.GENERATING, SessionState.CANCELLED, SessionState.FAILED}
    ),
    SessionState.GENERATING: frozenset(
        {
            SessionState.PLAYING,
            SessionState.COMPLETED,
            SessionState.CANCELLED,
            SessionState.FAILED,
        }
    ),
    SessionState.PLAYING: frozenset(
        {SessionState.COMPLETED, SessionState.CANCELLED, SessionState.FAILED}
    ),
    SessionState.COMPLETED: frozenset(),
    SessionState.CANCELLED: frozenset(),
    SessionState.FAILED: frozenset(),
}
