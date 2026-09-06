"""Real-time voice runtime: sessions, chunked generation, playback, cancellation.

Honest capability: INCREMENTAL CHUNK GENERATION + BUFFERED PLAYBACK.

Chatterbox does not expose native token/waveform streaming. Phase 5 splits
resolved render text into speech units, synthesizes each unit under a
serialized inference lock, enqueues AudioChunks, and starts playback as soon
as the first chunk is ready.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..Config import SAMPLE_RATE
from ..audio_utils import load_audio
from ..core.context import ContextProfile
from ..core.exceptions import (
    MissingReferenceAudio,
    RealtimeSessionError,
    SessionCancelledError,
    SessionNotFound,
    VoiceRenderError,
)
from ..core.expression import ExpressionProfile
from ..core.models import VoiceIdentity
from ..evaluation.render_metadata import save_render_metadata
from ..identity.repository import get_renderer_version
from ..inference.chatterbox_mapping import map_expression_to_chatterbox
from ..inference.renderer import VoiceRenderer
from ..memory.resolver import FullRenderPlan
from .assembly import assemble_chunks, save_assembled_wav
from .chunking import split_speech_units
from .events import (
    CANCELLED,
    COMPLETED,
    FAILED,
    FIRST_AUDIO_READY,
    INTERRUPTED,
    PLAYBACK_STARTED,
    PROGRESS,
    SESSION_STARTED,
    SessionEventBus,
)
from .models import (
    STREAMING_MODE,
    AudioChunk,
    RealtimeSession,
    SessionState,
    generate_session_id,
)
from .queue import AudioChunkQueue
from .registry import RuntimeSessionRegistry
from .scheduler import DEFAULT_INFERENCE_SCHEDULER, InferenceScheduler
from .sink import AudioSink, CollectingAudioSink, NullAudioSink, create_default_sink

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict[str, Any]], None]

# Soft target for instrumentation / benchmarks — not a hard SLA.
DEFAULT_TARGET_FIRST_AUDIO_MS = 1500.0


class RealTimeVoiceService:
    """Orchestrates chunked real-time synthesis without owning identity/memory logic."""

    def __init__(
        self,
        identity_service: Any,
        *,
        scheduler: InferenceScheduler | None = None,
        event_bus: SessionEventBus | None = None,
        queue_maxsize: int = 8,
        target_first_audio_ms: float = DEFAULT_TARGET_FIRST_AUDIO_MS,
    ):
        self._identity_service = identity_service
        self._scheduler = scheduler or DEFAULT_INFERENCE_SCHEDULER
        self.events = event_bus or SessionEventBus()
        self.registry = RuntimeSessionRegistry()
        self.queue_maxsize = queue_maxsize
        self.target_first_audio_ms = target_first_audio_ms
        self._cancel_flags: dict[str, threading.Event] = {}
        self._workers: dict[str, threading.Thread] = {}
        self._chunk_stores: dict[str, list[AudioChunk]] = {}
        self._sinks: dict[str, AudioSink] = {}
        self._plans: dict[str, FullRenderPlan] = {}
        self._lock = threading.RLock()
        self._model_ready = False

    # ------------------------------------------------------------------ model lifecycle

    def warm_up(self) -> dict[str, Any]:
        """Optionally load the TTS model without forcing app startup load."""
        start = time.perf_counter()
        renderer: VoiceRenderer = self._identity_service.renderer
        # Prefer cloner lazy-load path when using Chatterbox.
        try:
            from .. import cloner

            if hasattr(cloner, "_get_model"):
                cloner._get_model()
            elif hasattr(renderer, "warm_up"):
                renderer.warm_up()  # type: ignore[attr-defined]
        except Exception as e:
            raise RealtimeSessionError(
                f"Model warm-up failed: {e}",
                user_message="Could not warm up the speech model.",
            ) from e
        self._model_ready = True
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return {"ready": True, "warm_up_ms": elapsed_ms, "streaming_mode": STREAMING_MODE}

    def is_model_ready(self) -> bool:
        return self._model_ready

    # ------------------------------------------------------------------ public API

    def start_session(
        self,
        identity_id: str,
        text: str,
        *,
        expression: str | ExpressionProfile | dict[str, Any] | None = None,
        context: str | ContextProfile | dict[str, Any] | None = None,
        use_memory: bool = True,
        play_audio: bool = True,
        save_final: bool = True,
        output_path: str | Path | None = None,
        sink: AudioSink | None = None,
        on_event: EventCallback | None = None,
        max_chunk_chars: int = 180,
    ) -> RealtimeSession:
        """Start a real-time speech session (non-blocking)."""
        identity = self._identity_service.repository.get(identity_id)
        processed = self._identity_service.repository.resolve_path(
            identity, identity.processed_audio
        )
        if not processed.exists():
            raise MissingReferenceAudio(
                f"Processed reference missing for {identity_id}",
                user_message="Reference audio is missing for this voice identity.",
            )

        session = RealtimeSession(
            session_id=generate_session_id(),
            identity_id=identity_id,
            text=text,
            use_memory=use_memory,
            chunking_strategy="sentence",
        )
        session.metrics.mark_request()
        session.metrics.streaming_mode = STREAMING_MODE

        if on_event is not None:
            self.events.subscribe(on_event)

        audio_sink = sink or create_default_sink(
            play_audio=play_audio,
            on_event=lambda ev, payload: self._on_sink_event(session.session_id, ev, payload),
        )

        cancel_flag = threading.Event()
        with self._lock:
            self.registry.add(session)
            self._cancel_flags[session.session_id] = cancel_flag
            self._chunk_stores[session.session_id] = []
            self._sinks[session.session_id] = audio_sink

        worker = threading.Thread(
            target=self._run_session,
            name=f"realtime-{session.session_id[-8:]}",
            args=(
                session,
                identity,
                processed,
                expression,
                context,
                use_memory,
                save_final,
                output_path,
                audio_sink,
                cancel_flag,
                max_chunk_chars,
            ),
            daemon=True,
        )
        with self._lock:
            self._workers[session.session_id] = worker
        worker.start()
        return session

    def get_session(self, session_id: str) -> RealtimeSession:
        return self.registry.get(session_id)

    def wait(
        self,
        session_id: str,
        timeout: float | None = None,
    ) -> RealtimeSession:
        """Block until the session reaches a terminal state."""
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            session = self.registry.get(session_id)
            if session.is_terminal():
                return session
            with self._lock:
                worker = self._workers.get(session_id)
            if worker is not None and not worker.is_alive() and not session.is_terminal():
                # Worker died without transitioning — mark failed.
                session.error_message = "Real-time worker exited unexpectedly"
                session.metrics.error = session.error_message
                try:
                    session.transition(SessionState.FAILED)
                except ValueError:
                    pass
                self.events.emit(
                    FAILED,
                    {"session_id": session_id, "error": session.error_message},
                )
                return session
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError(f"Timed out waiting for session {session_id}")
            time.sleep(0.02)

    def cancel(self, session_id: str) -> RealtimeSession:
        """Cancel generation/playback at the next generation boundary."""
        session = self.registry.get(session_id)
        if session.is_terminal():
            return session
        session.metrics.mark_cancel_requested()
        with self._lock:
            flag = self._cancel_flags.get(session_id)
            sink = self._sinks.get(session_id)
        if flag is not None:
            flag.set()
        if sink is not None:
            try:
                sink.flush()
                sink.stop()
            except Exception:
                logger.exception("Error stopping sink on cancel for %s", session_id)
        self.events.emit(CANCELLED, {"session_id": session_id, "state": session.state.value})
        return session

    def interrupt(self, session_id: str) -> RealtimeSession:
        """Barge-in ready API — currently equivalent to cancel (no mic detection)."""
        session = self.cancel(session_id)
        self.events.emit(INTERRUPTED, {"session_id": session_id})
        return session

    def streaming_mode(self) -> str:
        return STREAMING_MODE

    # ------------------------------------------------------------------ worker

    def _run_session(
        self,
        session: RealtimeSession,
        identity: VoiceIdentity,
        processed: Path,
        expression: str | ExpressionProfile | dict[str, Any] | None,
        context: str | ContextProfile | dict[str, Any] | None,
        use_memory: bool,
        save_final: bool,
        output_path: str | Path | None,
        sink: AudioSink,
        cancel_flag: threading.Event,
        max_chunk_chars: int,
    ) -> None:
        chunk_queue = AudioChunkQueue(maxsize=self.queue_maxsize)
        playback_thread: threading.Thread | None = None
        collected: list[AudioChunk] = []

        try:
            session.transition(SessionState.PREPARING)
            session.metrics.mark_prepare_start()
            self.events.emit(
                SESSION_STARTED,
                {"session_id": session.session_id, "identity_id": identity.id},
            )

            if cancel_flag.is_set():
                self._finalize_cancelled(session, sink, chunk_queue)
                return

            # Resolve once for the entire session — identity/expression/context/memory stable.
            plan = self._identity_service.resolve_render_plan(
                expression,
                context,
                identity_id=identity.id,
                text=session.text,
                use_memory=use_memory,
            )
            with self._lock:
                self._plans[session.session_id] = plan

            session.expression_name = plan.resolved_expression.versioned_name
            session.context_name = plan.context.versioned_name
            session.memory_item_ids = tuple(plan.memory.memories_applied)
            session.metrics.mark_prepare_end()

            units = split_speech_units(plan.render_text, max_chars=max_chunk_chars)
            if not units:
                raise RealtimeSessionError(
                    "No speakable text after resolution",
                    user_message="There is no text to speak.",
                )

            session.transition(SessionState.GENERATING)
            try:
                sink.start()
            except Exception as e:
                raise RealtimeSessionError(
                    f"Playback sink failed to start: {e}",
                    user_message="Could not start audio playback.",
                ) from e

            playback_thread = threading.Thread(
                target=self._playback_loop,
                name=f"playback-{session.session_id[-8:]}",
                args=(session, chunk_queue, sink, cancel_flag),
                daemon=True,
            )
            playback_thread.start()

            renderer: VoiceRenderer = self._identity_service.renderer
            for index, unit in enumerate(units):
                if cancel_flag.is_set():
                    break

                session.metrics.mark_inference_start()
                with self._scheduler.hold(session.session_id):
                    if cancel_flag.is_set():
                        break
                    chunk = self._synthesize_unit(
                        session=session,
                        renderer=renderer,
                        text=unit,
                        reference_audio=processed,
                        expression=plan.resolved_expression,
                        sequence=index,
                        is_final=(index == len(units) - 1),
                    )

                if cancel_flag.is_set():
                    break

                collected.append(chunk)
                with self._lock:
                    self._chunk_stores[session.session_id].append(chunk)

                if session.metrics.first_audio_ready is None:
                    session.metrics.mark_first_audio()
                    self.events.emit(
                        FIRST_AUDIO_READY,
                        {
                            "session_id": session.session_id,
                            "ttfa_ms": session.metrics.ttfa_ms,
                            "sequence": index,
                        },
                    )

                session.metrics.chunk_count += 1
                session.metrics.generated_audio_seconds += chunk.duration_seconds
                try:
                    chunk_queue.put(chunk, timeout=30.0)
                except Exception as e:
                    raise RealtimeSessionError(
                        f"Chunk queue backpressure failure: {e}",
                        user_message="Real-time generation stalled on the playback buffer.",
                    ) from e

                self.events.emit(
                    PROGRESS,
                    {
                        "session_id": session.session_id,
                        "sequence": index,
                        "chunk_count": session.metrics.chunk_count,
                        "state": session.state.value,
                    },
                )

            chunk_queue.close()
            if playback_thread is not None:
                playback_thread.join(timeout=60.0)

            if cancel_flag.is_set():
                self._finalize_cancelled(session, sink, chunk_queue, collected)
                return

            final_path = None
            if save_final:
                final_path = self._write_final_audio(
                    session, identity, plan, collected, output_path, use_memory=use_memory
                )
                session.final_audio_path = final_path

            session.metrics.underrun_count = chunk_queue.underruns
            if hasattr(sink, "underruns"):
                session.metrics.underrun_count += int(getattr(sink, "underruns"))

            session.metrics.mark_complete()
            if session.state == SessionState.PLAYING:
                session.transition(SessionState.COMPLETED)
            elif session.state == SessionState.GENERATING:
                session.transition(SessionState.COMPLETED)

            self.events.emit(
                COMPLETED,
                {
                    "session_id": session.session_id,
                    "final_audio_path": final_path,
                    "metrics": session.metrics.to_dict(),
                },
            )
        except SessionCancelledError:
            self._finalize_cancelled(session, sink, chunk_queue, collected)
        except Exception as e:
            logger.exception("Real-time session %s failed", session.session_id)
            session.error_message = str(e)
            session.metrics.error = str(e)
            session.metrics.mark_complete()
            try:
                if not session.is_terminal():
                    session.transition(SessionState.FAILED)
            except ValueError:
                pass
            try:
                sink.flush()
                sink.stop()
            except Exception:
                pass
            chunk_queue.flush_and_close()
            self.events.emit(
                FAILED,
                {"session_id": session.session_id, "error": str(e)},
            )
        finally:
            try:
                sink.close()
            except Exception:
                logger.exception("Error closing sink for %s", session.session_id)
            with self._lock:
                self._workers.pop(session.session_id, None)
                self._cancel_flags.pop(session.session_id, None)

    def _synthesize_unit(
        self,
        *,
        session: RealtimeSession,
        renderer: VoiceRenderer,
        text: str,
        reference_audio: Path,
        expression: ExpressionProfile,
        sequence: int,
        is_final: bool,
    ) -> AudioChunk:
        """Synthesize one speech unit to memory (temp file cleaned immediately)."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                out = renderer.synthesize(
                    text,
                    reference_audio,
                    tmp_path,
                    expression=expression,
                )
            except Exception as e:
                raise VoiceRenderError(
                    f"Chunk synthesis failed: {e}",
                    user_message="Speech generation failed during real-time playback.",
                ) from e
            audio, sr = load_audio(out, sr=SAMPLE_RATE)
            samples = np.asarray(audio, dtype=np.float32).reshape(-1)
            return AudioChunk(
                session_id=session.session_id,
                sequence=sequence,
                samples=samples,
                sample_rate=sr,
                channels=1,
                dtype="float32",
                is_final=is_final,
                text=text,
            )
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _playback_loop(
        self,
        session: RealtimeSession,
        chunk_queue: AudioChunkQueue,
        sink: AudioSink,
        cancel_flag: threading.Event,
    ) -> None:
        while True:
            if cancel_flag.is_set():
                sink.flush()
                sink.stop()
                return
            item = chunk_queue.get(timeout=0.25)
            if item is None:
                if cancel_flag.is_set():
                    sink.flush()
                    sink.stop()
                    return
                # Closed + empty means producer finished.
                if chunk_queue._closed and chunk_queue.qsize() == 0:  # noqa: SLF001
                    return
                continue

            if session.state == SessionState.GENERATING:
                try:
                    session.transition(SessionState.PLAYING)
                except ValueError:
                    pass
            if session.metrics.playback_start is None:
                session.metrics.mark_playback_start()
                self.events.emit(
                    PLAYBACK_STARTED,
                    {
                        "session_id": session.session_id,
                        "ttfp_ms": session.metrics.ttfp_ms,
                    },
                )
            try:
                sink.enqueue(item)
            except Exception as e:
                logger.exception("Playback enqueue failed for %s", session.session_id)
                session.error_message = str(e)
                cancel_flag.set()
                return

            if item.is_final:
                # Drain any remaining (should be empty) then exit.
                continue

    def _write_final_audio(
        self,
        session: RealtimeSession,
        identity: VoiceIdentity,
        plan: FullRenderPlan,
        chunks: list[AudioChunk],
        output_path: str | Path | None,
        *,
        use_memory: bool,
    ) -> str:
        if output_path is None:
            out_dir = self._identity_service.repository.output_dir(identity.id)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"rt_{uuid.uuid4().hex[:8]}.wav"

        path, duration = save_assembled_wav(chunks, output_path)
        session.metrics.generated_audio_seconds = duration

        settings = map_expression_to_chatterbox(plan.resolved_expression)
        extra: dict[str, Any] = {
            "realtime_mode": True,
            "session_id": session.session_id,
            "streaming_mode": STREAMING_MODE,
            "chunking_strategy": session.chunking_strategy,
            "chunk_count": session.metrics.chunk_count,
            "ttfa_ms": session.metrics.ttfa_ms,
            "first_playback_ms": session.metrics.ttfp_ms,
            "generation_time_ms": session.metrics.total_generation_ms,
            "audio_duration_ms": round(duration * 1000.0, 3),
            "cancelled": False,
            "underrun_count": session.metrics.underrun_count,
            "target_first_audio_ms": self.target_first_audio_ms,
            "base_expression_name": plan.base_expression.versioned_name,
            "base_expression_profile": plan.base_expression.to_dict(),
            "context_name": plan.context.versioned_name,
            "context_version": plan.context.version,
            "context_profile": plan.context.to_dict(),
            "resolved_expression_name": plan.resolved_expression.versioned_name,
            "context_policy_version": plan.policy_version,
            "applied_context_rules": list(plan.applied_rules),
            "memory_items_used": list(plan.memory.memories_applied),
            "memory_items_consulted": list(plan.memory.memories_consulted),
            "memory_resolution_version": plan.memory.policy_version,
            "memory_resolution_time_ms": plan.memory.resolution_time_ms,
            "use_memory": use_memory,
            "render_text": plan.render_text,
            "identity_version": identity.schema_version,
        }
        save_render_metadata(
            path,
            identity_id=identity.id,
            identity_name=identity.name,
            expression=plan.resolved_expression,
            renderer=identity.renderer_model,
            renderer_version=get_renderer_version(),
            generation_parameters=settings.to_dict(),
            duration_seconds=duration,
            extra=extra,
        )
        return path

    def _finalize_cancelled(
        self,
        session: RealtimeSession,
        sink: AudioSink,
        chunk_queue: AudioChunkQueue,
        collected: list[AudioChunk] | None = None,
    ) -> None:
        try:
            sink.flush()
            sink.stop()
        except Exception:
            pass
        chunk_queue.flush_and_close()
        session.metrics.mark_cancel_completed()
        session.metrics.mark_complete()
        if collected is not None:
            session.metrics.chunk_count = len(collected)
            session.metrics.generated_audio_seconds = sum(c.duration_seconds for c in collected)
        if not session.is_terminal():
            try:
                session.transition(SessionState.CANCELLED)
            except ValueError:
                session.state = SessionState.CANCELLED
        self.events.emit(
            CANCELLED,
            {
                "session_id": session.session_id,
                "metrics": session.metrics.to_dict(),
            },
        )

    def _on_sink_event(self, session_id: str, event: str, payload: dict[str, Any]) -> None:
        if event == "playback_started":
            try:
                session = self.registry.get(session_id)
            except SessionNotFound:
                return
            if session.metrics.playback_start is None:
                session.metrics.mark_playback_start()
            self.events.emit(PLAYBACK_STARTED, {**payload, "session_id": session_id})

    def get_collected_chunks(self, session_id: str) -> list[AudioChunk]:
        with self._lock:
            return list(self._chunk_stores.get(session_id, []))

    def get_assembled_audio(self, session_id: str) -> tuple[np.ndarray, int]:
        return assemble_chunks(self.get_collected_chunks(session_id))
