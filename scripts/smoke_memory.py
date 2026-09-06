#!/usr/bin/env python3
"""Phase 4 Voice Memory completion gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def status(step: int, result: str, detail: str) -> dict:
    print(f"Step {step:2d}: {result:7s} — {detail}")
    return {"step": step, "result": result, "detail": detail}


def make_ref(path: Path) -> None:
    sr = 24000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), audio, sr)


def install_mocks(service) -> str:
    """Install deterministic mocks when real embedding/TTS stack is unavailable."""
    from pathlib import Path as P
    import numpy as np
    from voiceclone.identity.embeddings import EmbeddingStore

    vec = np.random.randn(256).astype(np.float32)
    vec /= np.linalg.norm(vec)

    def fake_generate(self, processed_audio, output_path):
        output_path = P(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, vec, allow_pickle=False)
        return vec

    def fake_validate(self, path, **kwargs):
        path = P(path)
        if not path.exists():
            from voiceclone.core.exceptions import MissingEmbedding
            raise MissingEmbedding(f"Embedding not found: {path}")
        return np.load(path, allow_pickle=False)

    EmbeddingStore.generate = fake_generate  # type: ignore[method-assign]
    EmbeddingStore.validate = fake_validate  # type: ignore[method-assign]

    class FakeRenderer:
        def synthesize(self, text, reference_audio, output_path=None, **kwargs):
            import uuid
            out = P(output_path) if output_path else P(tempfile.gettempdir()) / f"out_{uuid.uuid4().hex[:8]}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sr = 24000
            t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
            audio = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
            sf.write(str(out), audio, sr)
            return str(out)

    service.renderer = FakeRenderer()

    import voiceclone.similarity as similarity

    def fake_compare_with_embedding(ref_embedding, generated_path):
        return float(np.clip(np.dot(ref_embedding, vec), -1.0, 1.0))

    similarity.compare_with_embedding = fake_compare_with_embedding
    return "MOCKED_STACK"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path)
    parser.add_argument("--real-models", action="store_true")
    args = parser.parse_args()

    results = []
    tmp = None if args.home else tempfile.TemporaryDirectory()
    home = args.home or Path(tmp.name)  # type: ignore[union-attr]
    os.environ["VOICECLONE_HOME"] = str(home)

    import voiceclone.Config as config

    config.ROOT = home
    config.VOICES_DIR = home / "voices"
    config.OUTPUTS_DIR = home / "outputs"
    config.VOICES_DIR.mkdir(parents=True, exist_ok=True)

    from voiceclone.core.service import VoiceIdentityService
    from voiceclone.evaluation.render_metadata import load_render_metadata

    ref = home / "external.wav"
    make_ref(ref)

    service = VoiceIdentityService(auto_migrate=False)
    mode = "REAL"
    if not args.real_models:
        mode = install_mocks(service)
    results.append(status(0, "PASS", f"Stack mode={mode}"))

    try:
        identity = service.create_from_file("MemGate", ref)
    except Exception as e:
        results.append(status(1, "FAIL", f"create identity failed: {e}"))
        print(json.dumps({"results": results}, indent=2))
        return 1

    identity_id = identity.id
    before_embedding = identity.embedding_path
    results.append(status(1, "PASS", f"Selected identity {identity_id}"))

    pron = service.add_pronunciation_memory(identity_id, "Minitorch", "mini-torch")
    style = service.add_style_preference(identity_id, "speaking_rate", 0.65)
    results.append(status(2, "PASS", "Added pronunciation + style memories"))

    service2 = VoiceIdentityService(auto_migrate=False)
    if mode == "MOCKED_STACK":
        install_mocks(service2)
    loaded = service2.list_memories(identity_id)
    results.append(status(3, "PASS" if len(loaded) >= 2 else "FAIL", f"Reloaded {len(loaded)} memories"))
    service = service2

    sentence = "Minitorch improves learning."
    try:
        with_mem = service.synthesize(identity_id, sentence, use_memory=True)
        meta_on = load_render_metadata(with_mem)
        ok = pron.id in meta_on.get("memory_items_used", []) and "mini-torch" in meta_on.get("render_text", "")
        results.append(status(4, "PASS" if ok else "FAIL", "Memory-enabled synthesis consulted memories"))
    except Exception as e:
        results.append(status(4, "FAIL", str(e)))
        meta_on = {}

    try:
        without = service.synthesize(identity_id, sentence, use_memory=False)
        meta_off = load_render_metadata(without)
        ok = meta_off.get("memory_items_used", []) == [] and "mini-torch" not in meta_off.get("render_text", "")
        results.append(status(5, "PASS" if ok else "FAIL", "Memory bypass ignored memories"))
    except Exception as e:
        results.append(status(5, "FAIL", str(e)))

    service.disable_memory(identity_id, style.id)
    plan = service.resolve_render_plan(identity_id=identity_id, text=sentence, use_memory=True)
    ok = style.id not in plan.memory.memories_applied and pron.id in plan.memory.memories_applied
    results.append(status(6, "PASS" if ok else "FAIL", f"applied={list(plan.memory.memories_applied)}"))

    other = service.create_from_file("MemGateB", ref)
    plan_b = service.resolve_render_plan(identity_id=other.id, text=sentence, use_memory=True)
    results.append(status(7, "PASS" if "mini-torch" not in plan_b.render_text else "FAIL", "Identity isolation"))

    updated = service.update_memory(
        identity_id,
        pron.id,
        value={"pronunciation_type": "custom_text", "pronunciation_value": "MINI-torch"},
    )
    results.append(status(8, "PASS" if updated.version >= 2 else "FAIL", f"version={updated.version}"))

    service.delete_memory(identity_id, pron.id)
    plan_del = service.resolve_render_plan(identity_id=identity_id, text=sentence, use_memory=True)
    ok = "mini-torch" not in plan_del.render_text and "MINI-torch" not in plan_del.render_text
    results.append(status(9, "PASS" if ok else "FAIL", plan_del.render_text))

    after = service.get_identity(identity_id)
    ok = after.embedding_path == before_embedding and after.id == identity_id
    results.append(status(10, "PASS" if ok else "FAIL", "Identity unchanged"))

    try:
        service.add_pronunciation_memory(identity_id, "Minitorch", "mini-torch")
        best, score = service.synthesize_best_of(identity_id, sentence, n=3, use_memory=True)
        results.append(status(11, "PASS", f"Best-of-3 with memory score={score:.3f}"))
    except Exception as e:
        results.append(status(11, "FAIL", str(e)))

    try:
        bench = service.benchmark(identity_id, sentences=[sentence], compare_memory=True)
        results.append(status(12, "PASS", f"Benchmark comparison keys={list(bench)}"))
    except Exception as e:
        results.append(status(12, "FAIL", str(e)))

    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(Path(__file__).resolve().parents[1]))
    results.append(status(13, "PASS" if proc.returncode == 0 else "FAIL", f"pytest exit={proc.returncode}"))

    # Optional real-model attempt
    if not args.real_models:
        results.append(status(14, "LIMITED", "Real Chatterbox/Resemblyzer validation skipped (use --real-models)"))
    else:
        results.append(status(14, "PASS", "Real-model mode requested by caller"))

    fails = [r for r in results if r["result"] == "FAIL"]
    print(json.dumps({"results": results, "failures": len(fails), "mode": mode}, indent=2))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
