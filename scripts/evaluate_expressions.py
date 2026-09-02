#!/usr/bin/env python3
"""
Expressive voice evaluation script.

Generates a standard evaluation set for one identity across multiple expressions
and sentences, recording generation metadata for human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate expressive voice generation")
    parser.add_argument("--identity-id", required=True, help="Voice identity ID")
    parser.add_argument("--output-dir", type=Path, help="Directory for evaluation outputs")
    parser.add_argument("--home", type=Path, help="Override VOICECLONE_HOME")
    args = parser.parse_args()

    if args.home:
        os.environ["VOICECLONE_HOME"] = str(args.home)
        import voiceclone.Config as config
        config.ROOT = args.home
        config.VOICES_DIR = args.home / "voices"
        config.OUTPUTS_DIR = args.home / "outputs"

    from voiceclone.core.service import VoiceIdentityService
    from voiceclone.evaluation.render_metadata import load_render_metadata

    service = VoiceIdentityService(auto_migrate=False)
    identity = service.get_identity(args.identity_id)

    expressions = ["neutral", "calm", "warm", "professional", "serious", "excited", "concerned", "urgent"]
    sentences = [
        "The project is finally complete.",
        "Good morning, everyone.",
        "We need to discuss this immediately.",
    ]

    out_dir = args.output_dir or (
        Path(os.environ.get("VOICECLONE_HOME", Path.home() / ".voiceclone"))
        / "outputs"
        / identity.id
        / "expression_eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for expression in expressions:
        for sentence in sentences:
            t0 = time.time()
            try:
                path = service.synthesize(identity.id, sentence, expression=expression)
                elapsed = round(time.time() - t0, 2)
                meta = load_render_metadata(path)
                similarity = service.compare(identity.id, path)
                audio, sr = sf.read(path)
                duration = round(len(audio) / sr, 3)
                rows.append({
                    "identity_id": identity.id,
                    "expression": expression,
                    "sentence": sentence,
                    "output": path,
                    "similarity": round(similarity, 4),
                    "duration_s": duration,
                    "gen_time_s": elapsed,
                    "renderer_settings": json.dumps(meta.get("generation_parameters", {})),
                    "identity_fidelity": round(similarity, 4),
                    "intelligibility": "",
                    "expressive_distinction": "",
                    "naturalness": "",
                    "artifact_presence": "",
                })
                print(f"PASS {expression:12s} | sim={similarity:.3f} | {path}")
            except Exception as e:
                print(f"LIMITED {expression:12s} | {e}")
                rows.append({
                    "identity_id": identity.id,
                    "expression": expression,
                    "sentence": sentence,
                    "output": "",
                    "similarity": "",
                    "duration_s": "",
                    "gen_time_s": "",
                    "renderer_settings": "",
                    "identity_fidelity": "",
                    "intelligibility": "",
                    "expressive_distinction": "",
                    "naturalness": "",
                    "artifact_presence": f"error: {e}",
                })

    csv_path = out_dir / f"expression_eval_{int(time.time())}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nEvaluation CSV: {csv_path}")
    print("Human rating columns (expressive_distinction, naturalness, etc.) are left blank for manual review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
