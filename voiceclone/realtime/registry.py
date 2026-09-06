"""Thread-safe registry of active real-time sessions."""

from __future__ import annotations

import threading

from ..core.exceptions import SessionNotFound
from .models import RealtimeSession, TERMINAL_STATES


class RuntimeSessionRegistry:
    """Explicit, thread-safe map of session_id → RealtimeSession."""

    def __init__(self):
        self._sessions: dict[str, RealtimeSession] = {}
        self._lock = threading.RLock()

    def add(self, session: RealtimeSession) -> None:
        with self._lock:
            if session.session_id in self._sessions:
                raise ValueError(f"Session already registered: {session.session_id}")
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> RealtimeSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(
                    f"Real-time session not found: {session_id}",
                    user_message="That real-time speech session no longer exists.",
                )
            return session

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_active(self) -> list[RealtimeSession]:
        with self._lock:
            return [s for s in self._sessions.values() if s.state not in TERMINAL_STATES]

    def list_all(self) -> list[RealtimeSession]:
        with self._lock:
            return list(self._sessions.values())

    def clear_terminal(self) -> int:
        """Drop completed/cancelled/failed sessions from the registry."""
        with self._lock:
            to_drop = [
                sid for sid, session in self._sessions.items() if session.state in TERMINAL_STATES
            ]
            for sid in to_drop:
                del self._sessions[sid]
            return len(to_drop)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def __contains__(self, session_id: object) -> bool:
        with self._lock:
            return session_id in self._sessions
