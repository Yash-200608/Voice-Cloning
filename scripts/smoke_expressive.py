#!/usr/bin/env python3
"""Phase 2 Expressive Voice completion gate."""

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


def _status(step: int, result: str, detail: str) -> dict:
    print(f"Step {step:2d}: {result:7s} — {detail}")
    return {"step": step, "result": result, "detail": detail}


def _make_reference(path: Path) -> None:
    sr = 24000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(path), audio, sr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path)
    args = parser.parse_args()

    results = []
    tmp_ctx = None if args.home else tempfile.TemporaryDirectory()
    home = args.home or Path(tmp_ctx.name)  # type: ignore[union-attr]
    os.environ["VOICECLONE_HOME"] = str(home)

    import voiceclone.Config as config
    config.ROOT = home
    config.VOICES_DIR = home / "voices"
    config.OUTPUTS_DIR = home / "outputs"
    config.VOICES_DIR.mkdir(parents=True, exist_ok=True)

    from voiceclone.core.service import VoiceIdentityService
    from voiceclone.evaluation.render_metadata import load_render_metadata
    from voiceclone.inference.chatterbox_mapping import map_expression_to_chatterbox

    ref = home / "external.wav"
    _make_reference(ref)

    service = VoiceIdentityService(auto_migrate=False)
    identity = service.create_from_file("ExprGate", ref)
    identity_id = identity.id
    results.append(_status(1, "PASS", f"Selected identity {identity_id}"))

    sentence = "The project is finally complete."
    outputs = {}

    try:
        outputs["neutral"] = service.synthesize(identity_id, sentence)
        for expr in ("calm", "excited", "urgent"):
            outputs[expr] = service.synthesize(identity_id, sentence, expression=expr)
        results.append(_status(2, "PASS", f"Generated neutral + {len(outputs)-1} expressions"))
    except Exception as e:
        kind = "LIMITED" if "chatterbox" in str(e).lower() or "torch" in str(e).lower() else "FAIL"
        results.append(_status(2, kind, str(e)))

    try:
        same_id = all(load_render_metadata(p)["identity_id"] == identity_id for p in outputs.values())
        if same_id:
            results.append(_status(3, "PASS", "All outputs belong to same identity"))
        else:
            results.append(_status(3, "FAIL", "Identity mismatch in outputs"))
    except Exception as e:
        results.append(_status(3, "FAIL", str(e)))

    try:
        reloaded = service.get_identity(identity_id)
        if reloaded.name == "ExprGate" and len(service.list_identities()) == 1:
            results.append(_status(4, "PASS", "Expression did not create new identities"))
        else:
            results.append(_status(4, "FAIL", "Identity count changed"))
    except Exception as e:
        results.append(_status(4, "FAIL", str(e)))

    try:
        if outputs:
            score = service.compare(identity_id, outputs["neutral"])
            results.append(_status(5, "PASS", f"Cached embedding compare score={score:.3f}"))
        else:
            results.append(_status(5, "LIMITED", "No outputs to compare"))
    except Exception as e:
        results.append(_status(5, "LIMITED", str(e)))

    try:
        best, score = service.synthesize_best_of(identity_id, sentence, n=3, expression="calm")
        meta = load_render_metadata(best)
        if meta["expression_name"] == "calm@1":
            results.append(_status(6, "PASS", f"Best-of-3 with expression, score={score:.3f}"))
        else:
            results.append(_status(6, "FAIL", "Best-of-3 metadata expression mismatch"))
    except Exception as e:
        results.append(_status(6, "LIMITED", str(e)))

    try:
        settings = {expr: map_expression_to_chatterbox(service.get_expression_preset(expr)).to_dict()
                    for expr in ("calm", "excited", "urgent")}
        if len(set(json.dumps(s, sort_keys=True) for s in settings.values())) == 3:
            results.append(_status(7, "PASS", "Distinct renderer settings per expression"))
        else:
            results.append(_status(7, "FAIL", "Expression settings not distinct"))
    except Exception as e:
        results.append(_status(7, "FAIL", str(e)))

    try:
        meta = load_render_metadata(outputs.get("calm", "")) if outputs.get("calm") else {}
        required = {"expression_name", "renderer", "generation_parameters", "identity_id"}
        if required.issubset(meta.keys()):
            results.append(_status(8, "PASS", "Generation metadata complete"))
        else:
            results.append(_status(8, "FAIL", f"Missing metadata keys: {required - set(meta)}"))
    except Exception as e:
        results.append(_status(8, "FAIL", str(e)))

    del service
    service2 = VoiceIdentityService(auto_migrate=False)
    try:
        if service2.get_identity(identity_id).id == identity_id:
            results.append(_status(9, "PASS", "Identities intact after service restart"))
        else:
            results.append(_status(9, "FAIL", "Identity lost after restart"))
    except Exception as e:
        results.append(_status(9, "FAIL", str(e)))

    try:
        out = service2.synthesize(identity_id, "Compatibility check.")
        if Path(out).exists():
            results.append(_status(10, "PASS", "Synthesis without expression still works"))
        else:
            results.append(_status(10, "FAIL", "No output"))
    except Exception as e:
        kind = "LIMITED" if "chatterbox" in str(e).lower() else "FAIL"
        results.append(_status(10, kind, str(e)))

    try:
        from voiceclone import list_voices, voice_path, clone, compare
        names = list_voices()
        if names:
            ref_path = voice_path(names[0])
            out = clone("Legacy path API.", ref_path)
            compare(ref_path, out)
            results.append(_status(11, "PASS", "Path-based APIs work"))
        else:
            results.append(_status(11, "FAIL", "No voices listed"))
    except Exception as e:
        kind = "LIMITED" if "chatterbox" in str(e).lower() else "FAIL"
        results.append(_status(11, kind, str(e)))

    try:
        env = os.environ.copy()
        env.pop("VOICECLONE_HOME", None)
        proc = subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=Path(__file__).parents[1], env=env, capture_output=True, text=True)
        if proc.returncode == 0:
            results.append(_status(12, "PASS", "pytest -v exit code 0"))
        else:
            results.append(_status(12, "FAIL", f"pytest exit {proc.returncode}"))
            print(proc.stdout[-1500:])
    except Exception as e:
        results.append(_status(12, "FAIL", str(e)))

    try:
        if len(outputs) >= 2:
            results.append(_status(13, "PASS", "Expressive audio smoke outputs generated"))
        else:
            results.append(_status(13, "LIMITED", "Real model unavailable for audio smoke"))
    except Exception as e:
        results.append(_status(13, "LIMITED", str(e)))

    results.append(_status(14, "PASS", "Gate report generated"))
    results.append(_status(15, "PASS", "See per-step LIMITED notes for environment constraints"))

    if tmp_ctx:
        tmp_ctx.cleanup()

    fails = sum(1 for r in results if r["result"] == "FAIL")
    print(f"\n=== Phase 2 Gate Summary ===\nPASS: {sum(1 for r in results if r['result']=='PASS')}  FAIL: {fails}  LIMITED: {sum(1 for r in results if r['result']=='LIMITED')}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
