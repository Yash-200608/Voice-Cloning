"""Thread-safe event bus for real-time session notifications."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict[str, Any]], None]

SESSION_STARTED = "session_started"
FIRST_AUDIO_READY = "first_audio_ready"
PLAYBACK_STARTED = "playback_started"
PROGRESS = "progress"
COMPLETED = "completed"
CANCELLED = "cancelled"
FAILED = "failed"
INTERRUPTED = "interrupted"


class SessionEventBus:
    """Delivers session events to registered listeners without blocking producers."""

    def __init__(self):
        self._listeners: list[EventCallback] = []
        self._lock = threading.RLock()

    def subscribe(self, callback: EventCallback) -> None:
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        with self._lock:
            self._listeners = [cb for cb in self._listeners if cb is not callback]

    def emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        data = dict(payload or {})
        data.setdefault("event", event)
        with self._lock:
            listeners = list(self._listeners)
        for callback in listeners:
            try:
                callback(event, data)
            except Exception:
                logger.exception("Real-time event listener failed for %s", event)
