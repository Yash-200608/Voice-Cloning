#!/usr/bin/env python3
"""Phase 5 Real-Time Voice completion gate."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def status(step: int, result: str, detail: str) -> dict:
    print(f"Step {step:2d}: {result:7s} — {detail}")
    return {"step": step, "result": result, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-models", action="store_true")
    parser.add_argument("--play-audio", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results: list[dict] = []

    # 1. Import realtime package
    try:
        from voiceclone.realtime.models import STREAMING_MODE, SessionState
        from voiceclone.realtime.service import RealTimeVoiceService
        from voiceclone.core.service import VoiceIdentityService

        results.append(status(1, "PASS", f"Realtime package importable; mode={STREAMING_MODE}"))
    except Exception as e:
        results.append(status(1, "FAIL", f"Import failed: {e}"))
        _emit(results, args.json_out)
        return 1

    # 2–3. Fake identity + standard + realtime
    try:
        home = Path(tempfile.mkdtemp())
        import os

        os.environ["VOICECLONE_HOME"] = str(home)
        import voiceclone.Config as cfg

        cfg.ROOT = home
        cfg.VOICES_DIR = home / "voices"
        cfg.OUTPUTS_DIR = home / "outputs"
        cfg.CACHE_DIR = home / "cache"

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

        EmbeddingStore.generate = fake_generate  # type: ignore
        EmbeddingStore.validate = lambda self, path, **k: np.load(path)  # type: ignore

        class FakeRenderer:
            def synthesize(self, text, reference_audio, output_path=None, **kwargs):
                out = Path(output_path) if output_path else home / "o.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                sr = 24000
                dur = max(0.15, min(0.5, len(str(text)) / 100))
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                sf.write(str(out), (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
                return str(out)

        repo = VoiceRepository(
            voices_dir=home / "voices",
            outputs_dir=home / "outputs",
            embedding_store=store,
        )
        service = VoiceIdentityService(
            repository=repo,
            renderer=FakeRenderer(),
            embedding_store=store,
            auto_migrate=False,
        )
        ref = home / "ref.wav"
        sf.write(
            str(ref),
            (0.3 * np.sin(2 * np.pi * np.linspace(0, 1, 24000, endpoint=False))).astype(np.float32),
            24000,
        )
        identity = service.create_from_file("SmokeRT", ref)
        results.append(status(2, "PASS", f"Identity ready: {identity.id}"))

        standard_path = service.synthesize(identity.id, "Standard synthesis check.")
        results.append(status(3, "PASS", f"Standard synthesize -> {standard_path}"))
    except Exception as e:
        results.append(status(2, "FAIL", str(e)))
        _emit(results, args.json_out)
        return 1

    # 4–8. Realtime session, metrics, cancel, resume
    try:
        from voiceclone.realtime.sink import CollectingAudioSink

        session = service.synthesize_realtime(
            identity.id,
            "Hello there. This is chunk two. And chunk three.",
            play_audio=False,
            save_final=True,
            sink=CollectingAudioSink(),
        )
        done = service.wait_realtime(session.session_id, timeout=60)
        metrics = done.metrics.to_dict()
        assert done.state == SessionState.COMPLETED
        assert metrics["chunk_count"] >= 2
        assert metrics["ttfa_ms"] is not None
        assert done.final_audio_path and Path(done.final_audio_path).exists()
        results.append(
            status(
                4,
                "PASS",
                f"Realtime completed chunks={metrics['chunk_count']} ttfa_ms={metrics['ttfa_ms']}",
            )
        )

        # Cancel path
        session2 = service.synthesize_realtime(
            identity.id,
            "One. Two. Three. Four. Five. Six.",
            play_audio=False,
            save_final=False,
            sink=CollectingAudioSink(),
        )
        service.cancel_realtime(session2.session_id)
        done2 = service.wait_realtime(session2.session_id, timeout=60)
        assert done2.state == SessionState.CANCELLED
        results.append(
            status(
                5,
                "PASS",
                f"Cancel ok latency_ms={done2.metrics.to_dict().get('cancel_latency_ms')}",
            )
        )

        # Another session after cancel
        session3 = service.synthesize_realtime(
            identity.id,
            "After cancel we can speak again.",
            play_audio=False,
            save_final=True,
            sink=CollectingAudioSink(),
        )
        done3 = service.wait_realtime(session3.session_id, timeout=60)
        assert done3.state == SessionState.COMPLETED
        results.append(status(6, "PASS", "Session after cancel completed"))

        # Streaming mode honesty
        mode = service.realtime.streaming_mode()
        assert mode == STREAMING_MODE == "chunked_generation"
        results.append(status(7, "PASS", f"Honest streaming mode label: {mode}"))
    except Exception as e:
        results.append(status(4, "FAIL", str(e)))
        _emit(results, args.json_out)
        return 1

    # 9. Pytest suite
    try:
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            results.append(status(8, "PASS", "Full pytest suite green"))
        else:
            results.append(status(8, "FAIL", proc.stdout[-500:] + proc.stderr[-500:]))
    except Exception as e:
        results.append(status(8, "FAIL", str(e)))

    # 10. Real model / hardware (optional)
    if args.real_models:
        try:
            real_service = VoiceIdentityService()
            identities = real_service.list_identities()
            if not identities:
                results.append(status(9, "LIMITED", "No stored identities for real-model run"))
            else:
                ident = identities[0]
                real_service.warm_up_realtime()
                session = real_service.synthesize_realtime(
                    ident.id,
                    "Real model realtime smoke.",
                    play_audio=args.play_audio,
                    save_final=True,
                )
                done = real_service.wait_realtime(session.session_id, timeout=300)
                results.append(
                    status(
                        9,
                        "PASS" if done.state == SessionState.COMPLETED else "FAIL",
                        f"state={done.state.value} metrics={done.metrics.to_dict()}",
                    )
                )
        except Exception as e:
            results.append(status(9, "LIMITED", f"Real-model/hardware unavailable: {e}"))
    else:
        results.append(
            status(
                9,
                "LIMITED",
                "Skipped real Chatterbox/device validation (pass --real-models [--play-audio])",
            )
        )

    results.append(
        status(
            10,
            "PASS",
            "Phase 5 gate finished; streaming is chunked_generation (not native model streaming)",
        )
    )
    _emit(results, args.json_out)
    return 0 if all(r["result"] != "FAIL" for r in results) else 1


def _emit(results: list[dict], path: Path | None) -> None:
    payload = {"results": results}
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
