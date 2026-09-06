"""Serialized inference scheduler for the TTS model.

Chatterbox is not assumed to be thread-safe. Multiple real-time sessions may
exist concurrently for playback/state, but model execution is serialized.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


class InferenceScheduler:
    """Global (or shared) lock that serializes renderer.synthesize calls."""

    def __init__(self, name: str = "tts"):
        self.name = name
        self._lock = threading.RLock()
        self._owner: str | None = None
        self._owner_lock = threading.Lock()

    @contextmanager
    def hold(self, session_id: str | None = None) -> Iterator[None]:
        logger.debug("InferenceScheduler waiting (%s) session=%s", self.name, session_id)
        self._lock.acquire()
        with self._owner_lock:
            self._owner = session_id
        try:
            logger.debug("InferenceScheduler acquired (%s) session=%s", self.name, session_id)
            yield
        finally:
            with self._owner_lock:
                self._owner = None
            self._lock.release()
            logger.debug("InferenceScheduler released (%s) session=%s", self.name, session_id)

    @property
    def current_owner(self) -> str | None:
        with self._owner_lock:
            return self._owner


# Process-wide default scheduler shared by RealTimeVoiceService instances that
# share a renderer/model. Tests may construct private schedulers.
DEFAULT_INFERENCE_SCHEDULER = InferenceScheduler()
