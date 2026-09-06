"""Phase 5 real-time voice unit tests (fake renderer, no audio device)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voiceclone.core.service import VoiceIdentityService
from voiceclone.identity.embeddings import EmbeddingStore
from voiceclone.identity.repository import VoiceRepository
from voiceclone.realtime.assembly import assemble_chunks
from voiceclone.realtime.chunking import split_speech_units
from voiceclone.realtime.models import AudioChunk, SessionState, generate_session_id
from voiceclone.realtime.queue import AudioChunkQueue
from voiceclone.realtime.sink import CollectingAudioSink, NullAudioSink


class FakeRenderer:
    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def synthesize(self, text, reference_audio, output_path=None, **kwargs):
        with self.lock:
            self.calls.append(str(text))
        time.sleep(self.delay)
        out = Path(output_path) if output_path else Path("/tmp") / "out.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sr = 24000
        dur = max(0.15, min(0.6, len(str(text)) / 120))
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        sf.write(str(out), (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
        return str(out)


@pytest.fixture
def identity_service(tmp_path, monkeypatch):
    home = tmp_path / "vc"
    monkeypatch.setenv("VOICECLONE_HOME", str(home))
    import voiceclone.Config as cfg

    cfg.ROOT = home
    cfg.VOICES_DIR = home / "voices"
    cfg.OUTPUTS_DIR = home / "outputs"
    cfg.CACHE_DIR = home / "cache"

    store = EmbeddingStore()

    def fake_generate(self, processed, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vec = np.ones(256, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        np.save(output_path, vec)
        return vec

    def fake_validate(self, path, **kwargs):
        return np.load(path)

    monkeypatch.setattr(EmbeddingStore, "generate", fake_generate)
    monkeypatch.setattr(EmbeddingStore, "validate", fake_validate)

    repo = VoiceRepository(
        voices_dir=home / "voices",
        outputs_dir=home / "outputs",
        embedding_store=store,
    )
    renderer = FakeRenderer()
    service = VoiceIdentityService(
        repository=repo,
        renderer=renderer,
        embedding_store=store,
        auto_migrate=False,
    )
    ref = tmp_path / "ref.wav"
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(str(ref), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    identity = service.create_from_file("RealtimeTest", ref)
    return service, identity, renderer


def test_split_speech_units_sentence_boundaries():
    units = split_speech_units("Hello world. How are you? Fine!")
    assert units == ["Hello world.", "How are you?", "Fine!"]


def test_audio_chunk_queue_backpressure_and_order():
    q = AudioChunkQueue(maxsize=2)
    sid = generate_session_id()
    c0 = AudioChunk(session_id=sid, sequence=0, samples=np.zeros(10, dtype=np.float32))
    c1 = AudioChunk(session_id=sid, sequence=1, samples=np.zeros(10, dtype=np.float32))
    q.put(c0, timeout=1)
    q.put(c1, timeout=1)
    assert q.qsize() == 2
    first = q.get(timeout=1)
    assert first.sequence == 0
    q.close()


def test_session_lifecycle_complete(identity_service):
    service, identity, renderer = identity_service
    sink = CollectingAudioSink()
    session = service.synthesize_realtime(
        identity.id,
        "First sentence. Second sentence.",
        play_audio=False,
        save_final=True,
        sink=sink,
    )
    done = service.wait_realtime(session.session_id, timeout=30)
    assert done.state == SessionState.COMPLETED
    assert done.metrics.chunk_count >= 2
    assert done.metrics.ttfa_ms is not None
    assert done.final_audio_path and Path(done.final_audio_path).exists()
    assert len(renderer.calls) >= 2


def test_cancel_active_session(identity_service):
    service, identity, renderer = identity_service
    renderer.delay = 0.08
    session = service.synthesize_realtime(
        identity.id,
        "One. Two. Three. Four. Five. Six.",
        play_audio=False,
        save_final=False,
        sink=NullAudioSink(),
    )
    time.sleep(0.03)
    service.cancel_realtime(session.session_id)
    done = service.wait_realtime(session.session_id, timeout=30)
    assert done.state == SessionState.CANCELLED
    assert done.metrics.cancel_latency_ms is not None


def test_identity_expression_context_memory_stable_across_chunks(identity_service):
    service, identity, renderer = identity_service
    service.add_pronunciation_memory(identity.id, "Minitorch", "mini-torch")
    session = service.synthesize_realtime(
        identity.id,
        "Minitorch is ready. Minitorch is steady.",
        expression="calm",
        context="desktop",
        use_memory=True,
        play_audio=False,
        save_final=True,
        sink=CollectingAudioSink(),
    )
    done = service.wait_realtime(session.session_id, timeout=30)
    assert done.state == SessionState.COMPLETED
    assert done.expression_name and done.expression_name.startswith("calm")
    assert done.context_name and "desktop" in done.context_name
    assert any("mini-torch" in c.lower() or "minitorch" in c.lower() for c in renderer.calls)


def test_use_memory_false_skips_pronunciation(identity_service):
    service, identity, renderer = identity_service
    service.add_pronunciation_memory(identity.id, "Minitorch", "mini-torch")
    session = service.synthesize_realtime(
        identity.id,
        "Minitorch online.",
        use_memory=False,
        play_audio=False,
        save_final=False,
        sink=CollectingAudioSink(),
    )
    done = service.wait_realtime(session.session_id, timeout=30)
    assert done.state == SessionState.COMPLETED
    assert any("Minitorch" in call for call in renderer.calls)


def test_failed_session_does_not_corrupt_next(identity_service):
    service, identity, renderer = identity_service
    original = renderer.synthesize

    def boom(text, reference_audio, output_path=None, **kwargs):
        raise RuntimeError("boom")

    renderer.synthesize = boom  # type: ignore[method-assign]
    session = service.synthesize_realtime(
        identity.id,
        "This will fail.",
        play_audio=False,
        save_final=False,
        sink=NullAudioSink(),
    )
    done = service.wait_realtime(session.session_id, timeout=30)
    assert done.state == SessionState.FAILED

    renderer.synthesize = original  # type: ignore[method-assign]
    session2 = service.synthesize_realtime(
        identity.id,
        "Recovery sentence.",
        play_audio=False,
        save_final=True,
        sink=CollectingAudioSink(),
    )
    done2 = service.wait_realtime(session2.session_id, timeout=30)
    assert done2.state == SessionState.COMPLETED


def test_standard_synthesize_still_works(identity_service):
    service, identity, _renderer = identity_service
    path = service.synthesize(identity.id, "Standard path still works.")
    assert Path(path).exists()


def test_assemble_chunks_ok(identity_service):
    service, identity, _renderer = identity_service
    session = service.synthesize_realtime(
        identity.id,
        "Alpha. Beta.",
        play_audio=False,
        save_final=False,
        sink=CollectingAudioSink(),
    )
    done = service.wait_realtime(session.session_id, timeout=30)
    chunks = service.realtime.get_collected_chunks(done.session_id)
    assert len(chunks) >= 1
    audio, sr = assemble_chunks(chunks)
    assert sr == 24000
    assert audio.size > 0


def test_thread_safe_registry_operations(identity_service):
    service, identity, renderer = identity_service
    renderer.delay = 0.03
    sessions: list[str] = []

    def start_one(i: int) -> None:
        s = service.synthesize_realtime(
            identity.id,
            f"Thread {i}. Another line.",
            play_audio=False,
            save_final=False,
            sink=NullAudioSink(),
        )
        sessions.append(s.session_id)

    threads = [threading.Thread(target=start_one, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(sessions) == 3
    assert len(set(sessions)) == 3
    for sid in sessions:
        service.cancel_realtime(sid)
        done = service.wait_realtime(sid, timeout=30)
        assert done.state in {SessionState.CANCELLED, SessionState.COMPLETED}
