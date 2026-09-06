"""Bounded audio queue with backpressure for real-time sessions."""

from __future__ import annotations

import queue
import threading
from typing import Iterator

from .models import AudioChunk


class AudioChunkQueue:
    """Thread-safe bounded queue of AudioChunk items.

    Putting blocks (with optional timeout) when full — providing backpressure
    so generation cannot grow memory without bound.
    """

    def __init__(self, maxsize: int = 8):
        if maxsize < 1:
            raise ValueError("AudioChunkQueue maxsize must be >= 1")
        self._queue: queue.Queue[AudioChunk | None] = queue.Queue(maxsize=maxsize)
        self._closed = False
        self._underruns = 0
        self._lock = threading.Lock()

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    @property
    def underruns(self) -> int:
        return self._underruns

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def qsize(self) -> int:
        return self._queue.qsize()

    def put(self, chunk: AudioChunk, timeout: float | None = 5.0) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot put onto a closed AudioChunkQueue")
        self._queue.put(chunk, timeout=timeout)

    def get(self, timeout: float | None = 0.5) -> AudioChunk | None:
        """Return next chunk, None on timeout (counts as underrun), or sentinel None after close."""
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                if not self._closed:
                    self._underruns += 1
            return None
        return item

    def close(self) -> None:
        """Signal that no more chunks will be produced."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

    def clear(self) -> None:
        """Discard queued chunks (used on cancel)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def flush_and_close(self) -> None:
        self.clear()
        self.close()

    def __iter__(self) -> Iterator[AudioChunk]:
        while True:
            item = self.get(timeout=0.5)
            if item is None:
                with self._lock:
                    closed = self._closed
                if closed and self._queue.empty():
                    break
                continue
            yield item
