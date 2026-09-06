"""Audio playback sinks for real-time voice output."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ..Config import SAMPLE_RATE
from ..core.exceptions import AudioDeviceError, PlaybackError
from .models import AudioChunk

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict], None]


class AudioSink(ABC):
    """Where real-time audio chunks are delivered for playback."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def enqueue(self, chunk: AudioChunk) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def pause(self) -> None:
        """Optional pause; default is a no-op."""

    def resume(self) -> None:
        """Optional resume; default is a no-op."""


class NullAudioSink(AudioSink):
    """Silent sink for tests and headless environments."""

    def __init__(self):
        self.chunks: list[AudioChunk] = []
        self.started = False
        self.stopped = False
        self.closed = False
        self._lock = threading.Lock()

    def start(self) -> None:
        self.started = True

    def enqueue(self, chunk: AudioChunk) -> None:
        with self._lock:
            if self.closed or self.stopped:
                return
            self.chunks.append(chunk)

    def stop(self) -> None:
        self.stopped = True

    def flush(self) -> None:
        with self._lock:
            self.chunks.clear()

    def close(self) -> None:
        self.closed = True
        self.stopped = True


class CollectingAudioSink(NullAudioSink):
    """Collects chunks and can assemble a contiguous waveform."""

    def assemble(self) -> tuple[np.ndarray, int]:
        with self._lock:
            if not self.chunks:
                return np.zeros(0, dtype=np.float32), SAMPLE_RATE
            sr = self.chunks[0].sample_rate
            parts = []
            for chunk in sorted(self.chunks, key=lambda c: c.sequence):
                if chunk.sample_rate != sr:
                    raise PlaybackError(
                        f"Sample rate mismatch in collected chunks: {chunk.sample_rate} vs {sr}",
                        user_message="Audio chunks could not be assembled due to format mismatch.",
                    )
                samples = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
                parts.append(samples)
            return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32), sr


class SoundDeviceAudioSink(AudioSink):
    """Local speaker playback via sounddevice OutputStream.

    Generation and device callbacks remain decoupled through an internal buffer.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = 1,
        on_event: EventCallback | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.on_event = on_event
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = None
        self._started = False
        self._stopped = False
        self._closed = False
        self._playback_started = False
        self._underruns = 0

    @property
    def underruns(self) -> int:
        return self._underruns

    def start(self) -> None:
        if self._started:
            return
        try:
            import sounddevice as sd
        except Exception as e:
            raise AudioDeviceError(
                f"sounddevice unavailable: {e}",
                user_message="Audio playback is unavailable in this environment.",
            ) from e
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            raise AudioDeviceError(
                f"Failed to open audio output device: {e}",
                user_message="Could not open the audio output device.",
            ) from e
        self._started = True
        self._stopped = False

    def enqueue(self, chunk: AudioChunk) -> None:
        if self._closed or self._stopped:
            return
        if chunk.sample_rate != self.sample_rate:
            raise PlaybackError(
                f"Unexpected sample rate {chunk.sample_rate}, expected {self.sample_rate}",
                user_message="Playback failed due to audio format mismatch.",
            )
        samples = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
        with self._lock:
            self._buffer = np.concatenate([self._buffer, samples])
            if not self._playback_started and self._buffer.size > 0:
                self._playback_started = True
                if self.on_event:
                    self.on_event("playback_started", {"session_id": chunk.session_id})

    def stop(self) -> None:
        self._stopped = True
        self.flush()
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                logger.exception("Error stopping audio stream")

    def flush(self) -> None:
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        self._closed = True
        self.stop()
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                logger.exception("Error closing audio stream")
            self._stream = None

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.debug("sounddevice status: %s", status)
        with self._lock:
            available = self._buffer.size
            if available >= frames:
                outdata[:, 0] = self._buffer[:frames]
                self._buffer = self._buffer[frames:]
            elif available > 0:
                outdata[:available, 0] = self._buffer
                outdata[available:, 0] = 0.0
                self._buffer = np.zeros(0, dtype=np.float32)
                self._underruns += 1
            else:
                outdata.fill(0.0)
                if self._playback_started and not self._stopped:
                    self._underruns += 1


def create_default_sink(
    *,
    play_audio: bool = True,
    on_event: EventCallback | None = None,
) -> AudioSink:
    """Create the best available sink for the current environment."""
    if not play_audio:
        return CollectingAudioSink()
    try:
        return SoundDeviceAudioSink(on_event=on_event)
    except Exception as e:
        logger.warning("Falling back to CollectingAudioSink: %s", e)
        return CollectingAudioSink()
