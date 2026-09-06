"""Evaluate real-time vs standard synthesis latency.

Defaults to a fake renderer/embeddings harness so CI can measure TTFA without
Chatterbox weights. Pass --real-models to use the live VoiceIdentityService.

Streaming capability is reported from STREAMING_MODE (this build:
chunked_generation — not native model streaming).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf


SHORT = "Hello there."
MEDIUM = (
    "Real-time voice should begin playback before the full utterance is finished. "
    "This medium sample spans a couple of sentences for chunking."
)
LONG = (
    "Phase five introduces a real-time runtime around the existing voice identity engine. "
    "Generation is split into speech units at sentence boundaries. "
    "Playback starts when the first audio chunk is ready. "
    "Cancellation stops further generation at the next unit boundary. "
    "Identity, expression, context, and memory remain stable across the session."
)


class _FakeRenderer:
    def synthesize(self, text, reference_audio, output_path=None, **kwargs):
        out = Path(output_path) if output_path else Path(tempfile.mkdtemp()) / "out.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sr = 24000
        dur = max(0.15, min(0.8, len(str(text)) / 100))
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        # tiny sleep so TTFA is measurable vs total
        time.sleep(0.01)
        sf.write(str(out), (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
        return str(out)


def _build_fake_service(home: Path):
    os.environ["VOICECLONE_HOME"] = str(home)
    import voiceclone.Config as cfg

    cfg.ROOT = home
    cfg.VOICES_DIR = home / "voices"
    cfg.OUTPUTS_DIR = home / "outputs"
    cfg.CACHE_DIR = home / "cache"

    from voiceclone.core.service import VoiceIdentityService
    from voiceclone.identity.embeddings import EmbeddingStore
    from voiceclone.identity.repository import VoiceRepository

    store = EmbeddingStore()

    def fake_generate(self, processed, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vec = np.ones(256, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        np.save(output_path, vec)
        return vec

    EmbeddingStore.generate = fake_generate  # type: ignore[method-assign]
    EmbeddingStore.validate = lambda self, path, **k: np.load(path)  # type: ignore[method-assign]

    repo = VoiceRepository(
        voices_dir=home / "voices",
        outputs_dir=home / "outputs",
        embedding_store=store,
    )
    return VoiceIdentityService(
        repository=repo,
        renderer=_FakeRenderer(),
        embedding_store=store,
        auto_migrate=False,
    )


def _ensure_identity(service, name: str, home: Path):
    existing = [i for i in service.list_identities() if i.name == name]
    if existing:
        return existing[0]
    ref = home / "ref.wav"
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sf.write(str(ref), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    return service.create_from_file(name, ref)


def _run_standard(service, identity_id: str, text: str) -> dict:
    start = time.perf_counter()
    path = service.synthesize(identity_id, text, save_metadata=True)
    total_ms = round((time.perf_counter() - start) * 1000.0, 3)
    info = sf.info(path)
    return {
        "mode": "standard",
        "path": path,
        "total_latency_ms": total_ms,
        "audio_duration_s": round(info.duration, 4),
        "ttfa_ms": total_ms,
    }


def _run_realtime(service, identity_id: str, text: str, play_audio: bool) -> dict:
    from voiceclone.realtime.models import STREAMING_MODE
    from voiceclone.realtime.sink import CollectingAudioSink

    sink = None if play_audio else CollectingAudioSink()
    session = service.synthesize_realtime(
        identity_id,
        text,
        play_audio=play_audio,
        save_final=True,
        sink=sink,
    )
    done = service.wait_realtime(session.session_id, timeout=600)
    metrics = done.metrics.to_dict()
    return {
        "mode": "realtime",
        "streaming_mode": STREAMING_MODE,
        "state": done.state.value,
        "path": done.final_audio_path,
        "metrics": metrics,
        "ttfa_ms": metrics.get("ttfa_ms"),
        "ttfp_ms": metrics.get("ttfp_ms"),
        "total_generation_ms": metrics.get("total_generation_ms"),
        "audio_duration_s": metrics.get("generated_audio_seconds"),
        "chunk_count": metrics.get("chunk_count"),
        "underrun_count": metrics.get("underrun_count"),
        "rtf": metrics.get("rtf"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play-audio", action="store_true")
    parser.add_argument("--real-models", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--identity-name", default="RealtimeEval")
    args = parser.parse_args()

    from voiceclone.core.service import VoiceIdentityService
    from voiceclone.realtime.models import STREAMING_MODE

    if args.real_models:
        service = VoiceIdentityService()
        home = Path(tempfile.mkdtemp())
    else:
        home = Path(tempfile.mkdtemp())
        service = _build_fake_service(home)

    identity = _ensure_identity(service, args.identity_name, home)

    report = {
        "streaming_mode": STREAMING_MODE,
        "backend": "real_models" if args.real_models else "fake_renderer",
        "streaming_mode_meaning": {
            "native_streaming": "Model emits audio tokens/waveform progressively",
            "chunked_generation": "App splits text and synthesizes units incrementally (this build)",
            "buffered_simulation": "Full render then fake progressive playback",
        },
        "identity_id": identity.id,
        "cases": {},
    }

    for label, text in (("short", SHORT), ("medium", MEDIUM), ("long", LONG)):
        standard = _run_standard(service, identity.id, text)
        realtime = _run_realtime(service, identity.id, text, play_audio=args.play_audio)
        report["cases"][label] = {
            "text": text,
            "standard": standard,
            "realtime": realtime,
            "ttfa_improvement_ms": (
                None
                if realtime.get("ttfa_ms") is None
                else round(standard["total_latency_ms"] - float(realtime["ttfa_ms"]), 3)
            ),
        }

    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
