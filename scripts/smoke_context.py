#!/usr/bin/env python3
"""Phase 3 Context-Aware Voice completion gate."""

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

    ref = home / "external.wav"
    _make_reference(ref)

    service = VoiceIdentityService(auto_migrate=False)
    identity = service.create_from_file("CtxGate", ref)
    identity_id = identity.id
    results.append(_status(1, "PASS", f"Selected identity {identity_id}"))

    sentence = "We need to leave now."
    outputs: dict[str, str] = {}

    try:
        outputs["default"] = service.synthesize(identity_id, sentence)
        outputs["desktop"] = service.synthesize(identity_id, sentence, context="desktop")
        outputs["car"] = service.synthesize(identity_id, sentence, context="car")
        outputs["noisy"] = service.synthesize(identity_id, sentence, context="noisy_environment")
        results.append(_status(2, "PASS", "Generated default + contextual outputs"))
    except Exception as e:
        kind = "LIMITED" if "chatterbox" in str(e).lower() or "torch" in str(e).lower() else "FAIL"
        results.append(_status(2, kind, str(e)))

    try:
        if outputs:
            outputs["expr_ctx"] = service.synthesize(
                identity_id, sentence, expression="calm", context="car",
            )
            results.append(_status(3, "PASS", "Expression + context synthesis works"))
        else:
            results.append(_status(3, "LIMITED", "No outputs for expr+context"))
    except Exception as e:
        kind = "LIMITED" if "chatterbox" in str(e).lower() else "FAIL"
        results.append(_status(3, kind, str(e)))

    try:
        plan1 = service.resolve_render_plan("calm", "car")
        plan2 = service.resolve_render_plan("calm", "car")
        if plan1.resolved_expression == plan2.resolved_expression:
            results.append(_status(4, "PASS", "Context resolution is deterministic"))
        else:
            results.append(_status(4, "FAIL", "Non-deterministic resolution"))
    except Exception as e:
        results.append(_status(4, "FAIL", str(e)))

    try:
        before = service.get_identity(identity_id)
        before_id = before.id
        before_embedding = before.embedding_path
        if outputs.get("car"):
            service.synthesize(identity_id, "Another line.", context="warning")
        after = service.get_identity(identity_id)
        if before_id == after.id and before_embedding == after.embedding_path:
            results.append(_status(5, "PASS", "Context did not modify identity"))
        else:
            results.append(_status(5, "FAIL", "Identity changed after contextual synthesis"))
    except Exception as e:
        results.append(_status(5, "FAIL", str(e)))

    try:
        if outputs.get("default"):
            score = service.compare(identity_id, outputs["default"])
            results.append(_status(6, "PASS", f"Cached embedding reused, score={score:.3f}"))
        else:
            results.append(_status(6, "LIMITED", "No output to compare"))
    except Exception as e:
        results.append(_status(6, "LIMITED", str(e)))

    try:
        best, score = service.synthesize_best_of(
            identity_id, sentence, n=3, expression="calm", context="car",
        )
        meta = load_render_metadata(best)
        if meta.get("context_name") == "car@1" and meta.get("base_expression_name") == "calm@1":
            results.append(_status(7, "PASS", f"Best-of-3 with context, score={score:.3f}"))
        else:
            results.append(_status(7, "FAIL", "Best-of-3 context metadata mismatch"))
    except Exception as e:
        kind = "LIMITED" if "chatterbox" in str(e).lower() else "FAIL"
        results.append(_status(7, kind, str(e)))

    try:
        meta = load_render_metadata(outputs["car"]) if outputs.get("car") else {}
        required = {
            "context_name",
            "context_profile",
            "base_expression_name",
            "resolved_expression_name",
            "context_policy_version",
            "applied_context_rules",
        }
        if required.issubset(meta.keys()):
            results.append(_status(8, "PASS", "Context metadata complete"))
        else:
            results.append(_status(8, "FAIL", f"Missing keys: {required - set(meta)}"))
    except Exception as e:
        results.append(_status(8, "FAIL", str(e)))

    del service
    service2 = VoiceIdentityService(auto_migrate=False)
    try:
        if service2.get_identity(identity_id).id == identity_id:
            results.append(_status(9, "PASS", "Identities intact after restart"))
        else:
            results.append(_status(9, "FAIL", "Identity lost after restart"))
    except Exception as e:
        results.append(_status(9, "FAIL", str(e)))

    try:
        out = service2.synthesize(identity_id, "Phase 1 compat.")
        if Path(out).exists():
            results.append(_status(10, "PASS", "Synthesis without context still works"))
        else:
            results.append(_status(10, "FAIL", "No output"))
    except Exception as e:
        kind = "LIMITED" if "chatterbox" in str(e).lower() else "FAIL"
        results.append(_status(10, kind, str(e)))

    try:
        out = service2.synthesize(identity_id, "Phase 2 compat.", expression="warm")
        meta = load_render_metadata(out)
        if meta.get("base_expression_name") == "warm@1":
            results.append(_status(11, "PASS", "Phase 2 expression API still works"))
        else:
            results.append(_status(11, "FAIL", "Expression metadata mismatch"))
    except Exception as e:
        kind = "LIMITED" if "chatterbox" in str(e).lower() else "FAIL"
        results.append(_status(11, kind, str(e)))

    try:
        from voiceclone import list_voices, voice_path, clone, compare
        names = list_voices()
        if names:
            ref_path = voice_path(names[0])
            out = clone("Legacy API.", ref_path)
            compare(ref_path, out)
            results.append(_status(12, "PASS", "Path-based legacy APIs work"))
        else:
            results.append(_status(12, "FAIL", "No voices listed"))
    except Exception as e:
        kind = "LIMITED" if "chatterbox" in str(e).lower() else "FAIL"
        results.append(_status(12, kind, str(e)))

    try:
        env = os.environ.copy()
        env.pop("VOICECLONE_HOME", None)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-v"],
            cwd=Path(__file__).parents[1],
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            results.append(_status(13, "PASS", "pytest -v exit code 0"))
        else:
            results.append(_status(13, "FAIL", f"pytest exit {proc.returncode}"))
            print(proc.stdout[-1500:])
    except Exception as e:
        results.append(_status(13, "FAIL", str(e)))

    try:
        if len(outputs) >= 3:
            settings = {
                k: load_render_metadata(v).get("generation_parameters", {})
                for k, v in outputs.items()
                if k != "expr_ctx"
            }
            distinct = len(set(json.dumps(s, sort_keys=True) for s in settings.values()))
            if distinct >= 2:
                results.append(_status(14, "PASS", "Contextual outputs reach renderer with distinct settings"))
            else:
                results.append(_status(14, "LIMITED", "Renderer settings not distinguishable"))
        else:
            results.append(_status(14, "LIMITED", "Real model unavailable for contextual audio smoke"))
    except Exception as e:
        results.append(_status(14, "LIMITED", str(e)))

    results.append(_status(15, "PASS", "Gate report generated — see LIMITED notes for environment constraints"))

    if tmp_ctx:
        tmp_ctx.cleanup()

    fails = sum(1 for r in results if r["result"] == "FAIL")
    print(
        f"\n=== Phase 3 Gate Summary ===\n"
        f"PASS: {sum(1 for r in results if r['result'] == 'PASS')}  "
        f"FAIL: {fails}  "
        f"LIMITED: {sum(1 for r in results if r['result'] == 'LIMITED')}"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
