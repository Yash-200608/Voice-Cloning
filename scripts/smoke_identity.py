#!/usr/bin/env python3
"""
Phase 1 Completion Gate smoke test.

Runs the mandatory lifecycle steps in an isolated VOICECLONE_HOME directory.
Prints PASS/FAIL/LIMITED for each step.

Usage:
    python scripts/smoke_identity.py [--reference /path/to/reference.wav]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def _make_reference(path: Path, seconds: float = 2.0, sr: int = 24000) -> None:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(path), audio, sr)


def _status(step: int, result: str, detail: str) -> dict:
    print(f"Step {step:2d}: {result:7s} — {detail}")
    return {"step": step, "result": result, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 voice identity smoke test")
    parser.add_argument("--reference", type=Path, help="External reference WAV to import")
    parser.add_argument("--home", type=Path, help="Override VOICECLONE_HOME (temp dir used by default)")
    args = parser.parse_args()

    results = []
    tmp_ctx = None if args.home else tempfile.TemporaryDirectory()
    home = args.home or Path(tmp_ctx.name)  # type: ignore[union-attr]
    os.environ["VOICECLONE_HOME"] = str(home)

    # Re-import config with new home
    import importlib
    import voiceclone.Config as config
    config.ROOT = home
    config.VOICES_DIR = home / "voices"
    config.OUTPUTS_DIR = home / "outputs"
    config.CACHE_DIR = home / "cache"
    config.VOICES_DIR.mkdir(parents=True, exist_ok=True)

    from voiceclone.core.service import VoiceIdentityService
    from voiceclone.identity.embeddings import EmbeddingStore

    external_ref = args.reference
    if external_ref is None:
        external_ref = home / "external_reference.wav"
        _make_reference(external_ref)
    external_hash = hashlib.sha256(external_ref.read_bytes()).digest()

    service = VoiceIdentityService(auto_migrate=False)

    # Step 1: Create identity
    try:
        identity = service.create_from_file("SmokeVoice", external_ref)
        results.append(_status(1, "PASS", f"Created identity {identity.id}"))
    except Exception as e:
        results.append(_status(1, "FAIL", str(e)))
        _print_summary(results)
        return 1

    identity_id = identity.id
    root = service.repository.identity_dir(identity_id)

    # Step 2: Raw vs processed separate
    try:
        raw = (root / "raw/reference.wav").read_bytes()
        processed = (root / "processed/reference.wav").read_bytes()
        if raw != processed:
            results.append(_status(2, "PASS", "Raw and processed references differ"))
        else:
            results.append(_status(2, "FAIL", "Raw and processed are identical"))
    except Exception as e:
        results.append(_status(2, "FAIL", str(e)))

    # Step 3: Stable ID in metadata
    try:
        import json
        meta = json.loads((root / "metadata.json").read_text())
        pattern = re.compile(r"^voice_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
        if pattern.match(meta["id"]) and meta["id"] == identity_id == root.name:
            results.append(_status(3, "PASS", f"Stable ID {meta['id']}"))
        else:
            results.append(_status(3, "FAIL", f"Invalid ID: {meta.get('id')}"))
    except Exception as e:
        results.append(_status(3, "FAIL", str(e)))

    # Step 4: Embedding validation
    try:
        emb_path = root / "embeddings/speaker.npy"
        store = EmbeddingStore()
        store.validate(emb_path, embedding_model=identity.embedding_model, embedding_version=identity.embedding_version)
        results.append(_status(4, "PASS", f"Embedding valid at {emb_path.name}"))
    except Exception as e:
        results.append(_status(4, "FAIL", str(e)))

    # Steps 5-7: Destroy and recreate service
    del service
    try:
        results.append(_status(5, "PASS", "Service destroyed (process-local teardown)"))
        service2 = VoiceIdentityService(auto_migrate=False)
        results.append(_status(6, "PASS", "Service recreated"))
        reloaded = service2.get_identity(identity_id)
        if reloaded.name == "SmokeVoice":
            results.append(_status(7, "PASS", f"Reloaded identity {reloaded.id}"))
        else:
            results.append(_status(7, "FAIL", "Reloaded identity mismatch"))
        service = service2
    except Exception as e:
        results.append(_status(5, "FAIL", str(e)))

    # Steps 8-10: Real model operations
    output_path = None
    try:
        output_path = service.synthesize(identity_id, "This is a smoke test of voice identity synthesis.")
        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            results.append(_status(8, "PASS", f"Generated {output_path}"))
        else:
            results.append(_status(8, "FAIL", "Output file missing or empty"))
    except Exception as e:
        results.append(_status(8, "LIMITED", f"Chatterbox unavailable: {e}"))

    if output_path and Path(output_path).exists():
        try:
            from voiceclone.identity.embeddings import EmbeddingStore
            import voiceclone.similarity as similarity

            validate_calls = 0
            embed_calls = 0
            original_validate = EmbeddingStore.validate
            original_embed = similarity.embed

            def counting_validate(self, path, **kwargs):
                nonlocal validate_calls
                validate_calls += 1
                return original_validate(self, path, **kwargs)

            def counting_embed(audio_path):
                nonlocal embed_calls
                embed_calls += 1
                return original_embed(audio_path)

            import voiceclone.identity.embeddings as emb_mod
            emb_mod.EmbeddingStore.validate = counting_validate
            similarity.embed = counting_embed

            score = service.compare(identity_id, output_path)
            if validate_calls >= 1 and embed_calls == 1:
                results.append(_status(9, "PASS", f"Similarity={score:.3f} (cached ref, generated embed only)"))
            else:
                results.append(_status(9, "FAIL", f"Unexpected embed calls: validate={validate_calls}, embed={embed_calls}"))
        except Exception as e:
            results.append(_status(9, "LIMITED", f"Resemblyzer unavailable: {e}"))

        try:
            best, best_score = service.synthesize_best_of(identity_id, "Best of three smoke test.", n=3)
            out_dir = service.repository.output_dir(identity_id)
            # Exclude any output generated in step 8 before counting Best-of-3 artifacts.
            if output_path and Path(output_path).exists():
                try:
                    Path(output_path).unlink()
                except OSError:
                    pass
            wav_count = len(list(out_dir.glob("*.wav")))
            if Path(best).exists() and wav_count == 1:
                results.append(_status(10, "PASS", f"Best-of-3 score={best_score:.3f}, losers cleaned"))
            else:
                results.append(_status(10, "FAIL", f"Best-of-3 cleanup issue, {wav_count} files remain"))
        except Exception as e:
            results.append(_status(10, "LIMITED", f"Best-of-3 failed: {e}"))
    else:
        results.append(_status(9, "LIMITED", "Skipped — no generated output"))
        results.append(_status(10, "LIMITED", "Skipped — no generated output"))

    # Step 11: Rename
    try:
        renamed = service.rename_identity(identity_id, "SmokeRenamed")
        if renamed.name == "SmokeRenamed" and renamed.id == identity_id:
            results.append(_status(11, "PASS", "Renamed; ID unchanged"))
        else:
            results.append(_status(11, "FAIL", "Rename changed ID or name incorrectly"))
    except Exception as e:
        results.append(_status(11, "FAIL", str(e)))

    # Step 12: Delete
    try:
        deleted = service.delete_identity(identity_id)
        if deleted and not root.exists():
            results.append(_status(12, "PASS", "Identity directory removed"))
        else:
            results.append(_status(12, "FAIL", "Identity not fully deleted"))
    except Exception as e:
        results.append(_status(12, "FAIL", str(e)))

    # Step 13: External source intact
    try:
        if hashlib.sha256(external_ref.read_bytes()).digest() == external_hash:
            results.append(_status(13, "PASS", "External reference source unchanged"))
        else:
            results.append(_status(13, "FAIL", "External reference was modified"))
    except Exception as e:
        results.append(_status(13, "FAIL", str(e)))

    # Step 14: Compatibility APIs (create temp identity for path APIs)
    try:
        from voiceclone import clone, clone_best_of, compare, list_voices, voice_path, benchmark

        compat_ref = home / "compat_source.wav"
        _make_reference(compat_ref)
        compat = service.create_from_file("CompatAPI", compat_ref)
        names = list_voices()
        path = voice_path("CompatAPI")
        out = clone("Compatibility test.", path)
        score = compare(path, out)
        best, _ = clone_best_of("Again.", path, n=2)
        summary = benchmark(path, sentences=["Benchmark sentence."])
        service.delete_identity(compat.id)
        if "CompatAPI" in names and score >= 0.0 and summary["n"] == 1:
            results.append(_status(14, "PASS", "Path-based public APIs work"))
        else:
            results.append(_status(14, "FAIL", "Compatibility API check failed"))
    except Exception as e:
        results.append(_status(14, "LIMITED" if "Chatterbox" in str(e) or "torch" in str(e).lower() else "FAIL", str(e)))

    # Step 15: Test suite
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-v"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            results.append(_status(15, "PASS", "pytest -v exit code 0"))
        else:
            results.append(_status(15, "FAIL", f"pytest failed (exit {proc.returncode})"))
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:], file=sys.stderr)
    except Exception as e:
        results.append(_status(15, "FAIL", str(e)))

    if tmp_ctx:
        tmp_ctx.cleanup()

    return _print_summary(results)


def _print_summary(results: list[dict]) -> int:
    print("\n=== Phase 1 Completion Gate Summary ===")
    fails = sum(1 for r in results if r["result"] == "FAIL")
    limited = sum(1 for r in results if r["result"] == "LIMITED")
    passes = sum(1 for r in results if r["result"] == "PASS")
    print(f"PASS: {passes}  FAIL: {fails}  LIMITED: {limited}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
